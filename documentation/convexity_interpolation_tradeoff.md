# Trade-off Assessment: Convexity-Preserving Interpolation in Phase 4b

## 1. Problem Statement

Our Phase 4b calendar reconciliation uses **linear interpolation** on a moneyness grid. Analysis reveals this introduces significant convexity violations:

| Configuration                 | Convexity Violations | Calendar Violations |
| ----------------------------- | -------------------- | ------------------- |
| Phase 4a only                 | 732                  | 1,666               |
| Phase 4a + 4b (linear interp) | 2,629                | 2                   |
| **Difference**                | **+1,897**           | **-1,664**          |

**The linear interpolation in Phase 4b introduces ~1,900 convexity violations while eliminating calendar violations.**

---

## 2. Why Linear Interpolation Breaks Convexity

### 2.1 The Mathematical Issue

Convexity requires the call price curve to be convex in strike:
$$\frac{\partial^2 C}{\partial K^2} \geq 0$$

Linear interpolation creates **piecewise linear** price curves. At the interpolation nodes, the second derivative is a Dirac delta (infinite), and between nodes, it's zero. When mapping back to original strikes, small non-convexities emerge at:

1. **Grid-to-original mapping**: Original strikes may not align with grid points
2. **Averaging effects**: Isotonic regression averages adjacent prices, potentially flattening convex regions
3. **Clipping to bid-ask**: After isotonic adjustment, clamping can break convexity

### 2.2 Example

Consider three consecutive strikes with repaired prices:

- K=100: C = 10.0 (convex: second_diff = +0.2)
- K=105: C = 8.0
- K=110: C = 6.2

After calendar adjustment via interpolation:

- K=100: C = 10.05 (adjusted up for calendar)
- K=105: C = 8.0 (unchanged)
- K=110: C = 6.2 (unchanged)

New second_diff = 10.05 - 2(8.0) + 6.2 = 0.25 ✓

But if K=105 was also adjusted:

- K=105: C = 7.95 (adjusted down)

New second_diff = 10.05 - 2(7.95) + 6.2 = 0.35 ✓ (still convex)

However, if intermediate strike adjustments don't respect the convexity constraint:

- K=100: C = 10.0
- K=105: C = 8.2 (adjusted up for calendar on a different grid point)
- K=110: C = 6.2

New second_diff = 10.0 - 2(8.2) + 6.2 = -0.2 ✗ **Convexity violated!**

---

## 3. Potential Solutions

### 3.1 Option A: Convexity-Preserving Interpolation

**Approach**: Replace linear interpolation with a method that preserves convexity.

**Methods available**:

1. **Monotone Convex Spline (Hyman filter)**: Standard in fixed-income curve construction
2. **PCHIP with convexity enforcement**: Piecewise Cubic Hermite with constraints
3. **Quadratic splines with convexity constraints**: Natural for second-derivative preservation

**Implementation complexity**: HIGH

- Need to implement custom interpolation routines
- May conflict with calendar constraints (convex + monotone interpolation is non-trivial)
- Scipy's PCHIP is shape-preserving but not convexity-guaranteeing

**Expected improvement**: Could reduce convexity violations by 50-80%

### 3.2 Option B: Post-4b Convexity Cleanup

**Approach**: After calendar reconciliation, run a convexity-fixing pass per expiration.

**Method**:

```python
for each expiration:
    while convexity_violations_exist:
        for each violation triplet (i-1, i, i+1):
            # Raise C[i] to restore convexity
            target = (C[i-1] + C[i+1]) / 2
            C[i] = min(target, ask[i])  # Stay within bounds
```

**Problem**: This can re-introduce calendar violations! We already tried this and it made things worse.

**Mitigation**: Run iteratively until equilibrium (already implemented but limited by constraint conflicts).

### 3.3 Option C: Joint Constraint Enforcement (Full QP)

**Approach**: Formulate a single QP that enforces all constraints simultaneously on the interpolated grid.

**Problem**: This is exactly what we avoided due to scalability and solver issues.

### 3.4 Option D: Accept Trade-off for Heston

**Approach**: Keep current implementation; document that it's optimized for Heston calibration.

**Rationale**:

- Heston calibration doesn't directly use convexity
- Calendar consistency is critical for term structure
- Convexity violations are mostly small (89% have magnitude < 0.01)

---

## 4. Cost-Benefit Analysis

### 4.1 Implementing Convexity-Preserving Interpolation

| Factor              | Assessment                                        |
| ------------------- | ------------------------------------------------- |
| Development time    | 2-4 hours                                         |
| Testing complexity  | High (need to verify both convexity and calendar) |
| Risk of regression  | Medium (may break calendar or bid-ask compliance) |
| Benefit for Heston  | Low (Heston doesn't use convexity directly)       |
| Benefit for density | High (Breeden-Litzenberger requires convexity)    |

### 4.2 Doing Nothing (Accept Trade-off)

| Factor                    | Assessment                         |
| ------------------------- | ---------------------------------- |
| Development time          | 0                                  |
| Risk                      | None                               |
| Heston calibration impact | Minimal                            |
| Density extraction impact | Moderate (would need separate fix) |

### 4.3 Partial Fix: Re-run Phase 4a after 4b

**Approach**: After Phase 4b calendar fix, re-run Phase 4a per-expiration repair to restore convexity.

**Expected behavior**:

- Phase 4a will restore convexity (hard monotonicity, soft convexity)
- May slightly increase calendar violations
- Could iterate 4a → 4b → 4a until stable

**Implementation**: Simple modification to pipeline.

---

## 5. Recommendation

### For Heston Calibration (Current Goal)

**Accept the trade-off.** The current implementation is optimized for Heston:

- 99.9% calendar violation reduction is critical for term structure
- Heston parameters {κ, θ, ξ, ρ, v₀} are determined by global fit, not local convexity
- Residual convexity violations won't materially affect calibration quality

### For Future Density Extraction

If risk-neutral density extraction is needed later:

1. **Option A (Preferred)**: Implement monotone convex interpolation using established algorithms
2. **Option B (Quick)**: Re-run Phase 4a after 4b and accept slight calendar degradation

### Implementation Priority

1. **Now**: Document trade-off; proceed with Heston calibration
2. **Phase 5+**: If density extraction needed, implement Option A

---

## 6. Quantified Impact on Heston

To verify that convexity violations don't materially affect Heston, we can measure:

1. **Pre-calibration**: Fit Heston to surface with 2,629 convexity violations
2. **Post-calibration**: Check parameter plausibility (Feller condition, bounds)
3. **Comparison**: If we later implement convexity-preserving interpolation, compare parameters

**Expected outcome**: Parameters should be stable because Heston's objective:
$$\min_{\Theta} \sum_{i,j} \left( \sigma^{\text{Heston}}(K_i, T_j; \Theta) - \sigma^{\text{market}}(K_i, T_j) \right)^2$$

is a global least-squares fit that averages over local non-convexities.

---

## 7. Conclusion

The convexity-preserving interpolation trade-off is:

| Approach             | Calendar       | Convexity            | Heston Impact | Density Impact    | Complexity |
| -------------------- | -------------- | -------------------- | ------------- | ----------------- | ---------- |
| Current (linear)     | ✅ 99.9% fixed | ⚠️ +1,897 introduced | ✅ Minimal    | ⚠️ Would need fix | Low        |
| Convex interpolation | ✅ Same        | ✅ ~50-80% better    | ✅ Same       | ✅ Good           | High       |
| Accept for Heston    | ✅ 99.9% fixed | ⚠️ Accept            | ✅ Minimal    | ⚠️ Defer          | None       |

**Recommendation**: Proceed with current implementation for Heston calibration. Document as a known limitation. Implement convexity-preserving interpolation only if density extraction is added to scope.

---

_This analysis supports the decision to prioritize calendar consistency for Heston calibration while acknowledging the convexity trade-off as a deliberate engineering choice._
