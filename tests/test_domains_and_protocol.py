import pytest
from think_fast_slow.domains.applications import ALL_DOMAINS
from think_fast_slow.evaluation.replications import replicate_table_6, replicate_table_9
from think_fast_slow.evaluation.hypotheses import evaluate_table_11_hypotheses

def test_all_seven_table_8_application_domains():
    """40. Table 8: Verifies registration of all 7 application domains."""
    expected_domains = {
        "crm": "Customer Service / CRM",
        "swe": "Software Engineering",
        "requirements": "Requirements Engineering",
        "logistics": "Logistics / Port Operations",
        "consumer": "Consumer behaviour / e-commerce",
        "egov": "E-Government Services",
        "agentic": "Agentic Software",
    }
    assert len(ALL_DOMAINS) == 7
    for key, name in expected_domains.items():
        assert key in ALL_DOMAINS
        profile = ALL_DOMAINS[key]
        assert profile.domain_name == name
        assert len(profile.sample_queries) > 0
        assert profile.costly_error is not None
        assert profile.fallback_action is not None

def test_table_6_and_table_9_replications():
    """41. Verifies numerical replication of Table 6 and Table 9."""
    t6 = replicate_table_6()
    assert len(t6) == 5
    for row in t6:
        assert 0.80 <= row["act_c"] <= 0.95
        assert 0.50 <= row["give_up_c"] <= 0.65
        assert row["kept"] > 0.40

    t9 = replicate_table_9(n_samples=5000)
    assert len(t9) == 5
    assert t9[0]["tokens"] == 0
    assert t9[1]["tokens"] == 4096

def test_table_11_hypotheses_evaluation_suite():
    """42. Table 11: Verifies evaluation logic for all 9 pre-specified hypotheses."""
    mock_log = {
        "H1": {"errors": 1, "total": 35, "epsilon": 0.20},
        "H1_prime": {"error_rate": 0.38, "mean_one_minus_c": 0.08, "error_ci_lower": 0.26},
        "H2": {"W2_mid": 0.85, "W1_mid": 0.25, "W2_top": 0.40, "W1_top": 0.60},
        "H3": {"anchor_share": 0.25, "reactance_share": 0.05, "G_inf": 0.05, "G_blind": 0.15, "reconcile_anchor_share": 0.08},
        "H4": {"B_top": 0, "B_mid": 2048, "task_difference_detected": True},
        "H5": {"auroc": 0.78},
        "H6": {"harmful_cut_pct": 0.38, "beneficial_kept_pct": 0.85},
        "H7": {"conformal_coverage": 0.96, "target_coverage": 0.95, "bayes_loss": 0.15, "always_act_loss": 0.35},
        "H8": {"probe_catches": 18, "threshold_catches": 4},
        "H9": {"gain_noisy": +0.08, "gain_biased": -0.12},
    }
    hypotheses = evaluate_table_11_hypotheses(mock_log)
    assert len(hypotheses) == 10  # H1, H1', H2-H9
    falsified = [h["id"] for h in hypotheses if h["falsified"]]
    assert len(falsified) == 0, f"Expected all hypotheses supported, got falsified: {falsified}"
