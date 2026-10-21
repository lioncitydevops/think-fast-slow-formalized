import math
import pytest
from think_fast_slow.metacognition.effort_allocation import (
    compute_s_confidence,
    compute_closed_form_budget,
    get_decision_band,
    evaluate_return_curves_table,
    NonparametricBudgetTable,
)
from think_fast_slow.core.types import DecisionBand, ReturnCurveType

def test_prop_5_6_sensitivity_s_confidence():
    """25. Proposition 5.6: Effort sensitivity s(c) calculation and ambiguity suppression."""
    s_confident = compute_s_confidence(0.95, r=0.1, gamma=2.0)
    s_mid = compute_s_confidence(0.75, r=0.1, gamma=2.0)
    s_low = compute_s_confidence(0.50, r=0.1, gamma=2.0)
    # At c=0.50 (coin flip), d=1.0, 1 - d^2 = 0 -> s(c) = 0
    assert abs(s_low - 0.0) < 1e-6
    # s(c) peaks at intermediate confidence
    assert s_mid > s_confident
    assert s_mid > s_low

def test_prop_5_6_law_of_least_effort_cutoff():
    """26. Proposition 5.6: Law of least effort cutoff c >= (1 - lambda*beta)/(1 + r)."""
    r = 0.1
    lambda_beta = 0.01
    cutoff = (1.0 - lambda_beta) / (1.0 + r)  # 0.99 / 1.1 = 0.90
    b_at_cutoff = compute_closed_form_budget(cutoff + 0.01, r=r, lambda_beta=lambda_beta)
    assert b_at_cutoff == 0

def test_prop_5_6_closed_form_budget_formula():
    """27. Proposition 5.6: Optimal budget matches B*(c) = beta * ln(s / lambda*beta)."""
    beta = 1024
    lambda_beta = 0.01
    c = 0.75
    s = compute_s_confidence(c, r=0.1, gamma=2.0)
    expected_b = int(round(beta * math.log(s / lambda_beta)))
    actual_b = compute_closed_form_budget(c, r=0.1, gamma=2.0, lambda_beta=lambda_beta, beta=beta)
    assert actual_b == expected_b

def test_table_6_canonical_return_curves():
    """28. Table 6: Optimal budgets under five return curves."""
    curves = evaluate_return_curves_table()
    assert len(curves) == 5
    curve_names = [c["curve"] for c in curves]
    assert "saturating" in curve_names
    assert "log-linear" in curve_names
    assert "early-flat" in curve_names
    assert "threshold" in curve_names
    assert "inverted-U" in curve_names
    # Check saturating kept = 1.0, threshold kept = 0.45
    sat = next(c for c in curves if c["curve"] == "saturating")
    thresh = next(c for c in curves if c["curve"] == "threshold")
    assert sat["kept"] == 1.00
    assert thresh["kept"] == 0.45

def test_prop_5_7_threshold_minimum_effective_budget():
    """29. Proposition 5.7: Minimum effective budget on threshold curve."""
    curves = evaluate_return_curves_table()
    thresh = next(c for c in curves if c["curve"] == "threshold")
    assert thresh["min_B_pos"] >= 3000

def test_prop_5_8_nonparametric_budget_table_fit():
    """30. Proposition 5.8: Nonparametric task-conditioned budget table fitting."""
    table = NonparametricBudgetTable()
    cal_data = [
        {"confidence": 0.95, "task": "swe", "budget": 0, "accuracy_gain": 0.0},
        {"confidence": 0.95, "task": "swe", "budget": 1024, "accuracy_gain": -0.02},
        {"confidence": 0.75, "task": "swe", "budget": 2048, "accuracy_gain": 0.15},
        {"confidence": 0.75, "task": "swe", "budget": 0, "accuracy_gain": 0.0},
    ]
    table.fit(cal_data)
    assert table.get_budget(0.95, task="swe") == 0
    assert table.get_budget(0.75, task="swe") > 0
