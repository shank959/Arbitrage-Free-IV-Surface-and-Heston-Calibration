"""
Phase 4: Arbitrage-Free Surface Repair via Convex Optimization

This module implements the repair of option prices to eliminate static arbitrage
violations while respecting bid-ask bounds.

Phase 4a (per-expiration repair):
- Repairs each expiration independently with hard monotonicity + convexity constraints
- Uses CVXPY QP solver for reliable results

Phase 4b (calendar reconciliation):
- Applies isotonic regression across maturities to enforce calendar monotonicity
- Lightweight O(grid_size * maturities) algorithm without joint QP

Key components:
- repair.py: Core QP construction, per-expiration repair
- calendar_reconcile.py: Isotonic regression for calendar consistency
- validate.py: Post-repair validation using Phase 2 detection logic
"""

from .repair import repair_surface, RepairResult
from .calendar_reconcile import apply_calendar_reconciliation, CalendarReconcileResult
from .validate import validate_repaired_surface

__all__ = [
    "repair_surface",
    "RepairResult",
    "apply_calendar_reconciliation",
    "CalendarReconcileResult",
    "validate_repaired_surface",
]
