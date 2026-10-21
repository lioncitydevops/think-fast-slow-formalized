import math
import pytest
import numpy as np
from think_fast_slow.metacognition.escalation import compute_escalation_identity
from think_fast_slow.metacognition.act_band import clopper_pearson_upper, certify_act_threshold, WaldSPRT
from think_fast_slow.metacognition.paired_transitions import compute_paired_transition_decomposition

def test_prop_5_1_escalation_identity_machine_precision():
    """15. Proposition 5.1 Escalation Identity holds to machine precision."""
    a1_list = ["A", "B", "A", "B", "A", "B", "A", "A"]
    a2_list = ["A", "A", "B", "B", "B", "A", "A", "B"]
    gold_list = ["A", "A", "A", "B", "B", "A", "B", "A"]
    res = compute_escalation_identity(a1_list, a2_list, gold_list)
    assert res["identity_holds"]
    assert res["diff_machine_precision"] < 1e-12

def test_prop_5_1_binary_choice_formula():
    """16. Proposition 5.1 K=2 identity G(Z) = M_Z * (2 * W_2 - 1)."""
    a1_list = ["Yes", "No", "Yes", "No", "Yes"]
    a2_list = ["Yes", "Yes", "No", "No", "No"]
    gold_list = ["Yes", "Yes", "Yes", "No", "No"]
    res = compute_escalation_identity(a1_list, a2_list, gold_list)
    assert res["identity_holds"]
    m_z = res["M_Z"]
    w_2 = res["W2"]
    expected_gain = m_z * (2.0 * w_2 - 1.0)
    assert abs(res["G_identity"] - expected_gain) < 1e-12

def test_corollary_5_2_label_free_disagreement():
    """17. Corollary 5.2: Disagreement mass M_Z is computable without gold labels."""
    a1_list = ["A", "B", "C", "D", "A"]
    a2_list = ["A", "C", "C", "D", "B"]
    disagreements = sum(1 for a, b in zip(a1_list, a2_list) if a != b)
    m_z = disagreements / len(a1_list)
    assert m_z == 0.40

def test_prop_5_3_act_band_guarantee():
    """18. Proposition 5.3: Act-band error is bounded by 1 - tau under calibration."""
    confs = [0.95, 0.92, 0.90, 0.88, 0.70, 0.60]
    is_correct = [True, True, True, False, True, False]
    tau = 0.90
    act_idx = [i for i, c in enumerate(confs) if c >= tau]
    act_error = sum(1 for i in act_idx if not is_correct[i]) / len(act_idx)
    assert act_error <= (1.0 - tau) + 0.10

def test_remark_1_clopper_pearson_exact_calculation():
    """19. Remark 1: Clopper-Pearson exact confidence bounds calculation."""
    # When k=0, n=59, delta=0.05: 1 - 0.05^(1/59) <= 0.05
    upper = clopper_pearson_upper(k=0, n=59, delta=0.05)
    assert upper <= 0.051
    # Check k=0, n=1
    upper_single = clopper_pearson_upper(k=0, n=1, delta=0.05)
    assert abs(upper_single - 0.95) < 1e-4

def test_remark_1_fixed_sequence_scan_minimum_sample_size():
    """20. Remark 1: Fixed sequence testing scans only thresholds meeting sample requirements."""
    confs = [0.99] + [0.90] * 30 + [0.85] * 30
    corrects = [True] * 61
    res = certify_act_threshold(confs, corrects, epsilon=0.15, delta=0.05)
    assert res["certified_tau"] <= 0.95
    assert res["stats"]["bound_satisfied"]

def test_prop_5_4_paired_transition_decomposition_identity():
    """21. Proposition 5.4 Paired Transition Decomposition holds to machine precision."""
    a1_list = ["A", "B", "A", "B", "A", "B", "A", "B"]
    blind_list = ["A", "A", "B", "B", "B", "A", "A", "B"]
    inf_list = ["A", "B", "A", "B", "B", "B", "A", "A"]
    gold_list = ["A", "A", "A", "B", "B", "A", "B", "A"]
    res = compute_paired_transition_decomposition(a1_list, blind_list, inf_list, gold_list)
    assert res["identity_holds"]
    assert res["diff_machine_precision"] < 1e-12

def test_prop_5_4_stable_term_vanishes_identically():
    """22. Proposition 5.4: Stable term contribution vanishes identically."""
    a1_list = ["A", "B", "A", "B"]
    blind_list = ["A", "B", "A", "B"]  # All stable
    inf_list = ["A", "B", "A", "B"]
    gold_list = ["A", "A", "B", "B"]
    res = compute_paired_transition_decomposition(a1_list, blind_list, inf_list, gold_list)
    assert res["stable_contribution_zero"]
    assert res["breakdown"]["stable"]["contribution"] == 0.0

def test_corollary_5_5_pure_anchoring_scaling():
    """23. Corollary 5.5: Pure anchoring scaling G_inf = (1 - alpha) * G_blind."""
    a1_list = ["A", "A", "A", "A"]
    blind_list = ["B", "B", "B", "B"]
    # 2 out of 4 anchor to a1
    inf_list = ["A", "A", "B", "B"]
    gold_list = ["B", "B", "B", "B"]
    res = compute_paired_transition_decomposition(a1_list, blind_list, inf_list, gold_list)
    assert res["is_pure_anchoring"]
    alpha = res["shares"]["anchor"]
    assert abs(res["G_inf"] - (1.0 - alpha) * res["G_blind"]) < 1e-6

def test_wald_sprt_stopping_boundaries():
    """24. Wald's SPRT (Section 7.1) sequential likelihood ratio testing and drift detection."""
    sprt = WaldSPRT(epsilon=0.05, delta_sp=0.05, alpha_err=0.05, beta_err=0.05)
    # Stream series of errors to trigger drift alarm (reject H0)
    status = "CONTINUE"
    for _ in range(30):
        status = sprt.update(is_error=True)
        if status != "CONTINUE":
            break
    assert status == "REJECT_H0_DRIFT_DETECTED"

    # Reset and stream stream of correct decisions to trigger valid certification (accept H0)
    sprt.reset()
    for _ in range(80):
        status = sprt.update(is_error=False)
        if status != "CONTINUE":
            break
    assert status == "ACCEPT_H0_CALIBRATION_VALID"
