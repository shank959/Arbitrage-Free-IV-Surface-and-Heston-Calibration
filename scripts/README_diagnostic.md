# Full Pipeline Diagnostic Investigation

This script implements comprehensive diagnostic checks to understand why:
1. Heston RMSE is 5% instead of 1-3%
2. Repair makes Heston calibration WORSE (-1.8% improvement)

## Usage

```bash
# Run diagnostic on default date (2025-10-15)
python3 scripts/diagnostic_pipeline.py

# Run on specific date
python3 scripts/diagnostic_pipeline.py --date 2025-10-20

# Custom paths
python3 scripts/diagnostic_pipeline.py \
    --input reports/phase4/1545/repaired_options.parquet \
    --repair-summary reports/phase4/1545/repair_summary.csv \
    --output reports/diagnostic \
    --max-options 150
```

## Diagnostic Phases

### Phase A: Data Integrity Checks

**A1: Raw Data Quality Check**
- Verifies raw prices are reasonable
- Checks bid-ask spreads
- Flags zeros, negatives, NaN values

**A2: Repair Adjustment Analysis**
- Measures magnitude of repair adjustments
- Flags if >20% of options changed by >10%

**A3: IV Computation Consistency**
- Compares Phase 3 vs Phase 5 IV functions
- Flags if difference > 0.1%

### Phase B: Trace Single Option

**B1: Pick Representative Option**
- Selects ATM option with ~3 months maturity

**B2: Compute IVs from Both Prices**
- Calculates IV from raw mid vs repaired price

**B3: Heston Fit Comparison**
- Calibrates Heston on both raw and repaired
- Compares model IV to market IV for test option

### Phase C: Calibration Sample Comparison

**C1: Sampling Consistency**
- Verifies raw and repaired get same options selected
- Flags if overlap < 80%

**C2: IV Distribution Comparison**
- Compares IV distributions
- Computes correlation for same options
- Flags if correlation < 0.95%

### Phase D: Repair Quality Metrics

**D1: Load Repair Summary**
- Analyzes repair summary statistics

**D2: Spot Check Arbitrage Violations**
- Checks if repair removed calendar arbitrage violations
- Compares violations in raw vs repaired

### Phase E: Model Mismatch Hypothesis

**E1: Measure Smile Characteristics**
- Compares smile slope (OTM - ATM) for raw vs repaired
- Flags if repaired slope is steeper

**E2: Check Term Structure**
- Analyzes ATM term structure changes
- Measures IV changes across maturities

## Output

The script generates:
- Console output with detailed findings
- `reports/diagnostic/diagnostic_results.json` - Complete results in JSON format

## Key Findings from Initial Run

Based on the first diagnostic run:

1. **⚠️ A3: IV Computation Inconsistency**
   - Max difference: 1.42% between Phase 3 and Phase 5 IV functions
   - This is a significant issue that could explain calibration problems

2. **⚠️ B3: Repaired Error > Raw Error**
   - For the test option, repaired Heston error (2.51%) > raw error (2.10%)
   - Suggests repair may be making fit worse for some options

3. **✓ C1: Sampling is Consistent**
   - 91% overlap between raw and repaired option sets
   - Sampling is working correctly

## Next Steps

Based on diagnostic findings:

1. **Fix IV Computation Inconsistency (Priority 1)**
   - Consolidate Phase 3 and Phase 5 IV functions
   - Phase 3 uses `minimize_scalar` with simple intrinsic
   - Phase 5 uses `brentq` with discounted intrinsic
   - These differences could cause calibration issues

2. **Investigate Repair Algorithm**
   - If repair makes fit worse, review Phase 4 repair logic
   - Check if adjustments are preserving smile characteristics

3. **Model Adequacy**
   - If all tests pass but RMSE stays high, consider:
     - Heston may be inadequate for this data
     - May need Bates model (Heston + jumps)
     - Accept 5% RMSE as best achievable

## Decision Tree

The diagnostic follows this decision tree:

```
Repair makes Heston worse
    ↓
A1: Data integrity → Issues? → Bad data quality
    ↓ OK
A2: Repair magnitude → >20% changed? → Too aggressive
    ↓ OK
A3: IV consistency → Diff > 0.1%? → IV function mismatch ⚠️ FOUND
    ↓ OK
C1: Sample consistency → <80% overlap? → Bad sampling
    ↓ OK
C2: IV correlation → <0.95? → Surface structure changed
    ↓ OK
E1: Smile steepness → Repaired steeper? → Paradox
    ↓ OK
E2: Term structure → Distorted? → TS issue
    ↓ OK
Model inadequacy → Need jumps
```

