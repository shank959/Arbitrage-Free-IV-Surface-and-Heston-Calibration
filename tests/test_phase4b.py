"""
Unit tests for Phase 4b: Calendar Reconciliation via Isotonic Regression
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, "src")

from pipelines.phase4.calendar_reconcile import (
    _isotonic_pav_clean,
    bounded_isotonic_regression,
    count_calendar_violations,
    direct_calendar_fix,
    fix_monotonicity_per_expiration,
)


class TestIsotonicPAV:
    """Tests for the Pool Adjacent Violators algorithm."""
    
    def test_already_monotonic(self):
        """Already non-decreasing sequence should remain unchanged."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = _isotonic_pav_clean(y, np.ones(len(y)))
        np.testing.assert_array_almost_equal(result, y)
    
    def test_strictly_decreasing(self):
        """Strictly decreasing sequence should become flat."""
        y = np.array([4.0, 3.0, 2.0, 1.0])
        result = _isotonic_pav_clean(y, np.ones(len(y)))
        expected = np.array([2.5, 2.5, 2.5, 2.5])  # Average of all
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_mixed_sequence(self):
        """Mixed sequence should be properly smoothed."""
        y = np.array([1.0, 4.0, 2.0, 5.0])
        result = _isotonic_pav_clean(y, np.ones(len(y)))
        # Check monotonicity
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-8
    
    def test_single_element(self):
        """Single element should return unchanged."""
        y = np.array([5.0])
        result = _isotonic_pav_clean(y, np.ones(len(y)))
        np.testing.assert_array_almost_equal(result, y)
    
    def test_empty_array(self):
        """Empty array should return empty."""
        y = np.array([])
        result = _isotonic_pav_clean(y, np.array([]))
        assert len(result) == 0


class TestBoundedIsotonic:
    """Tests for bounded isotonic regression."""
    
    def test_respects_bounds(self):
        """Result should respect bid-ask bounds."""
        y = np.array([5.0, 3.0, 4.0, 2.0])
        lower = np.array([4.0, 2.0, 3.0, 1.0])
        upper = np.array([6.0, 4.0, 5.0, 3.0])
        
        result = bounded_isotonic_regression(y, lower, upper, max_iters=10)
        
        # Check bounds
        np.testing.assert_array_less(lower - 1e-8, result)
        np.testing.assert_array_less(result, upper + 1e-8)
    
    def test_preserves_monotonicity_when_possible(self):
        """Should be monotonic when bounds allow."""
        y = np.array([1.0, 3.0, 2.0, 4.0])
        lower = np.array([0.0, 0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0, 10.0])
        
        result = bounded_isotonic_regression(y, lower, upper, max_iters=10)
        
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-8


class TestCalendarViolations:
    """Tests for calendar violation counting."""
    
    def test_no_violations(self):
        """Data with no violations should return 0."""
        df = pd.DataFrame({
            "quote_date": ["2025-01-01"] * 4,
            "strike": [100.0, 100.0, 100.0, 100.0],
            "expiration": ["2025-01-10", "2025-01-15", "2025-01-20", "2025-01-25"],
            "price_repaired": [5.0, 5.5, 6.0, 6.5],  # Increasing with maturity (correct)
        })
        
        violations = count_calendar_violations(df, "price_repaired")
        assert violations == 0
    
    def test_with_violations(self):
        """Data with violations should be counted correctly."""
        df = pd.DataFrame({
            "quote_date": ["2025-01-01"] * 4,
            "strike": [100.0, 100.0, 100.0, 100.0],
            "expiration": ["2025-01-10", "2025-01-15", "2025-01-20", "2025-01-25"],
            "price_repaired": [6.0, 5.0, 4.0, 7.0],  # Two violations: 6>5 and 5>4
        })
        
        violations = count_calendar_violations(df, "price_repaired")
        assert violations == 2


class TestDirectCalendarFix:
    """Tests for direct calendar fix."""
    
    def test_fixes_violations(self):
        """Should fix calendar violations."""
        df = pd.DataFrame({
            "quote_date": ["2025-01-01"] * 4,
            "strike": [100.0, 100.0, 100.0, 100.0],
            "expiration": ["2025-01-10", "2025-01-15", "2025-01-20", "2025-01-25"],
            "price_repaired": [6.0, 5.0, 4.0, 7.0],
            "mid": [5.5, 5.0, 4.5, 6.5],
            "bid": [5.0, 4.5, 4.0, 6.0],
            "ask": [6.0, 5.5, 5.0, 7.0],
        })
        
        before = count_calendar_violations(df, "price_repaired")
        fixed = direct_calendar_fix(df)
        after = count_calendar_violations(fixed, "price_repaired")
        
        assert after < before


class TestMonotonicityFix:
    """Tests for monotonicity fix."""
    
    def test_fixes_violations(self):
        """Should fix monotonicity violations."""
        df = pd.DataFrame({
            "quote_date": ["2025-01-01"] * 4,
            "expiration": ["2025-01-15"] * 4,
            "strike": [100.0, 105.0, 110.0, 115.0],
            "price_repaired": [8.0, 10.0, 7.0, 5.0],  # Violation: 8 < 10
            "mid": [9.0, 8.0, 7.0, 6.0],
            "bid": [7.0, 7.0, 6.0, 4.0],
            "ask": [11.0, 11.0, 8.0, 6.0],
        })
        
        fixed = fix_monotonicity_per_expiration(df)
        prices = fixed.sort_values("strike")["price_repaired"].to_numpy()
        
        # Should be non-increasing
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1] - 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

