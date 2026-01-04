#!/usr/bin/env python3
"""
Phase 4: Arbitrage-Free Surface Repair via Convex Optimization

CLI entrypoint for repairing option surfaces to eliminate static arbitrage
violations while respecting bid-ask bounds.

Usage:
    python scripts/phase4_repair.py \
        --input data/processed/1545/options_1545.parquet \
        --output-dir reports/phase4/1545 \
        --slack-penalty 1000 \
        --validate

Mathematical Background (from vol_fitter.pdf):
-----------------------------------------------
The repair is formulated as a Quadratic Program (QP) that projects observed
mid-prices onto an arbitrage-consistent polytope:

    min  Σ w[i,j] * (C_tilde[i,j] - C_mid[i,j])^2

Subject to:
    1. Bid-Ask Bounds:   bid <= C_tilde <= ask
    2. Monotonicity:     C_tilde[i,j] >= C_tilde[i+1,j]  (calls decrease with strike)
    3. Convexity:        C_tilde[i-1,j] - 2*C_tilde[i,j] + C_tilde[i+1,j] >= 0
    4. Calendar Spread:  C_tilde[i,j] <= C_tilde[i,j+1]

When constraints are infeasible, slack variables are introduced with penalty λ.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.phase4.repair import repair_surface, RepairResult
from pipelines.phase4.validate import validate_repaired_surface, generate_validation_report


def main(argv=None):
    """
    Main CLI entrypoint for Phase 4 repair.
    
    Workflow:
    1. Parse command line arguments
    2. Run surface repair via convex optimization
    3. Optionally validate repaired surface (re-run detection)
    4. Export results and reports
    """
    parser = argparse.ArgumentParser(
        description="Phase 4: Repair option surfaces to eliminate arbitrage violations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic repair
  python scripts/phase4_repair.py --input data/processed/1545/options_1545.parquet --output-dir reports/phase4/1545

  # Repair with validation
  python scripts/phase4_repair.py --input data/processed/1545/options_1545.parquet --output-dir reports/phase4/1545 --validate

  # Custom slack penalty (higher = stricter constraints)
  python scripts/phase4_repair.py --input data/processed/1545/options_1545.parquet --output-dir reports/phase4/1545 --slack-penalty 5000
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input parquet file (e.g., data/processed/1545/options_1545.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files (e.g., reports/phase4/1545)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--slack-penalty",
        type=float,
        default=1000.0,
        help="Penalty λ for slack variables when constraints are infeasible (default: 1000). "
             "Higher values enforce stricter adherence to no-arbitrage constraints.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run post-repair validation to verify zero violations.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["1545", "eod", "both"],
        default=None,
        help="Shorthand for dataset selection (overrides --input if provided).",
    )
    
    # Phase 4b calendar reconciliation options
    parser.add_argument(
        "--no-calendar-fix",
        action="store_true",
        help="Skip Phase 4b calendar reconciliation (only run Phase 4a per-expiration repair).",
    )
    parser.add_argument(
        "--calendar-grid-size",
        type=int,
        default=200,
        help="Number of moneyness grid points for calendar reconciliation (default: 200).",
    )
    parser.add_argument(
        "--calendar-moneyness-min",
        type=float,
        default=0.6,
        help="Minimum moneyness (K/S) for calendar grid (default: 0.6).",
    )
    parser.add_argument(
        "--calendar-moneyness-max",
        type=float,
        default=1.4,
        help="Maximum moneyness (K/S) for calendar grid (default: 1.4).",
    )
    parser.add_argument(
        "--convexity-cleanup",
        action="store_true",
        help="Run Phase 4c convexity cleanup after calendar fix. "
             "This reduces convexity violations but may increase calendar violations. "
             "Use for density extraction; skip for Heston calibration.",
    )
    
    args = parser.parse_args(argv)
    
    # =========================================================================
    # Handle dataset shorthand
    # =========================================================================
    if args.dataset:
        datasets = []
        if args.dataset in ["1545", "both"]:
            datasets.append(("1545", Path("data/processed/1545/options_1545.parquet")))
        if args.dataset in ["eod", "both"]:
            datasets.append(("eod", Path("data/processed/eod/options_eod.parquet")))
    else:
        datasets = [("custom", Path(args.input))]
    
    # =========================================================================
    # Run repair for each dataset
    # =========================================================================
    for name, input_path in datasets:
        print(f"\n{'='*70}")
        print(f"PHASE 4: ARBITRAGE REPAIR - {name.upper()}")
        print(f"{'='*70}")
        
        # Determine output directory
        if args.dataset:
            output_dir = Path(f"reports/phase4/{name}")
        else:
            output_dir = Path(args.output_dir)
        
        # Check input exists
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}")
            continue
        
        print(f"Input:  {input_path}")
        print(f"Output: {output_dir}")
        print(f"Slack penalty: {args.slack_penalty}")
        print(f"Calendar fix: {'disabled' if args.no_calendar_fix else 'enabled'}")
        print()
        
        # =====================================================================
        # Run surface repair (Phase 4a + optional Phase 4b)
        # =====================================================================
        result = repair_surface(
            input_path=input_path,
            output_dir=output_dir,
            slack_penalty=args.slack_penalty,
            apply_calendar_fix=not args.no_calendar_fix,
            calendar_grid_size=args.calendar_grid_size,
            calendar_moneyness_bounds=(args.calendar_moneyness_min, args.calendar_moneyness_max),
            apply_convexity_cleanup=args.convexity_cleanup,
        )
        
        # =====================================================================
        # Step 5: Validate repaired surface (optional)
        # =====================================================================
        if args.validate:
            print("\nRunning post-repair validation...")
            validation = validate_repaired_surface(result.repaired_df)
            generate_validation_report(validation, output_dir)
            
            if validation.is_valid:
                print("\n✓ Validation PASSED: All arbitrage violations eliminated.")
            else:
                print(f"\n✗ Validation FAILED: {validation.post_violations['total']} violations remain.")
        
        # =====================================================================
        # Print final summary
        # =====================================================================
        print(f"\nOutputs written to: {output_dir}/")
        print(f"  - repaired_options.parquet  (full repaired dataset)")
        print(f"  - repair_summary.csv        (per quote_date metrics)")
        print(f"  - adjustments.csv           (per-option adjustments)")
        print(f"  - adjustment_heatmap.png    (visualization)")
        if result.feasibility_df is not None and not result.feasibility_df.empty:
            print(f"  - feasibility_report.csv    (slack variable usage)")
        if args.validate:
            print(f"  - validation_summary.csv    (pre/post violation counts)")


if __name__ == "__main__":
    main()

