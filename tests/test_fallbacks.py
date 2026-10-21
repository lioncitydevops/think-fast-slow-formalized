import pytest
from think_fast_slow.metacognition.fallbacks import compute_bayes_action, ConformalPredictor

def test_prop_5_9_calibrated_bayes_action_expected_loss():
    """31. Proposition 5.9: Calibrated Bayes action minimizes expected loss."""
    probs = {"cat": 0.80, "dog": 0.20}
    symmetric_loss = {
        "cat": {"cat": 0.0, "dog": 1.0, "hold": 0.5},
        "dog": {"cat": 1.0, "dog": 0.0, "hold": 0.5},
        "hold": {"cat": 0.5, "dog": 0.5, "hold": 0.0},
    }
    action, exp_losses = compute_bayes_action(probs, symmetric_loss)
    assert action == "cat"
    assert exp_losses["cat"] < exp_losses["dog"]

def test_bayes_action_asymmetric_loss_selects_hold():
    """32. Proposition 5.9: Asymmetric misclassification costs prefer safe hold under uncertainty."""
    # Diffuse distribution 0.45 vs 0.55
    probs = {"benign": 0.45, "malicious": 0.55}
    loss_matrix = {
        "benign": {"benign": 0.0, "malicious": 10.0, "hold": 0.3},
        "malicious": {"benign": 5.0, "malicious": 0.0, "hold": 0.3},
        "hold": {"benign": 0.2, "malicious": 0.2, "hold": 0.0},
    }
    action, exp_losses = compute_bayes_action(probs, loss_matrix)
    assert action == "hold"
    assert exp_losses["hold"] < exp_losses["benign"]
    assert exp_losses["hold"] < exp_losses["malicious"]

def test_prop_5_10_conformal_calibration_quantile():
    """33. Proposition 5.10: Conformal predictor calibrates finite-sample nonconformity score."""
    conformal = ConformalPredictor(alpha=0.10)
    cal_probs = [{"A": 0.9, "B": 0.1}, {"A": 0.8, "B": 0.2}, {"A": 0.7, "B": 0.3}]
    cal_golds = ["A", "A", "A"]
    q_hat = conformal.calibrate(cal_probs, cal_golds)
    assert 0.0 <= q_hat <= 1.0
    assert conformal.q_hat is not None

def test_conformal_prediction_set_coverage():
    """34. Proposition 5.10: Conformal set guarantees coverage P(y in C(x)) >= 1 - alpha."""
    conformal = ConformalPredictor(alpha=0.05)
    cal_probs = [{"yes": 0.95, "no": 0.05}] * 50
    cal_golds = ["yes"] * 50
    conformal.calibrate(cal_probs, cal_golds)
    eval_res = conformal.evaluate_coverage([{"yes": 0.90, "no": 0.10}] * 20, ["yes"] * 20)
    assert eval_res["coverage_guarantee_met"]
    assert eval_res["empirical_coverage"] >= 0.90
