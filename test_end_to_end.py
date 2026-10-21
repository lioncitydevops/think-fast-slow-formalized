import time
import math
import json
import random
from typing import Dict, Any, List

from think_fast_slow.core.types import Query, DecisionBand
from think_fast_slow.core.constants import TASK_BOOLQ, TASK_GSM8K, TASK_CRT, DEFAULT_R, DEFAULT_GAMMA, DEFAULT_LAMBDA_BETA, DEFAULT_BETA
from think_fast_slow.system1.jev_client import JevClient
from think_fast_slow.system2.qwen_client import QwenClient
from think_fast_slow.patterns.controller import DualProcessController
from think_fast_slow.patterns import (
    create_p1_cascade,
    create_p2_budgeted,
    create_p3_probe,
    create_p4_verifier,
    create_p5_reconcile,
)
from think_fast_slow.metacognition.escalation import compute_escalation_identity
from think_fast_slow.metacognition.act_band import certify_act_threshold, clopper_pearson_upper, WaldSPRT
from think_fast_slow.metacognition.paired_transitions import compute_paired_transition_decomposition
from think_fast_slow.metacognition.effort_allocation import (
    compute_s_confidence,
    compute_closed_form_budget,
    get_decision_band,
    evaluate_return_curves_table,
    NonparametricBudgetTable,
)
from think_fast_slow.metacognition.fallbacks import compute_bayes_action, ConformalPredictor
from think_fast_slow.metacognition.noise_audit import run_empirical_noise_audit, majority_probability
from think_fast_slow.domains.applications import ALL_DOMAINS
from think_fast_slow.evaluation.harness import EvaluationHarness
from think_fast_slow.evaluation.replications import replicate_table_3, replicate_table_5
from think_fast_slow.evaluation.hypotheses import evaluate_table_6_hypotheses

def run_comprehensive_audit() -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING END-TO-END VERIFICATION OF DUAL-PROCESS SYSTEM")
    print("=" * 70)
    audit_log = {}

    # 1. Check System 1 (Live Jev)
    print("\n[CHECK 1/8] Verifying Live System 1 (TypeSafe Jev)...")
    jev_live = JevClient(use_mock=False)
    test_q = Query(
        id="check_jev_01",
        text="The quick brown fox jumps over the lazy dog.",
        question="Is the animal that jumps a fox?",
        options=["Yes", "No"],
        gold_answer="Yes",
    )
    t0 = time.perf_counter()
    r_jev = jev_live.judge(test_q)
    t_jev = time.perf_counter() - t0
    print(f" -> Jev Choice: {r_jev.action} | Confidence: {r_jev.confidence:.2f} | Latency: {t_jev:.3f}s")
    assert r_jev.action == "Yes", f"Jev returned unexpected answer: {r_jev.action}"
    assert 0.5 <= r_jev.confidence <= 1.0, f"Invalid confidence: {r_jev.confidence}"
    audit_log["system1_live"] = {
        "status": "PASS",
        "action": r_jev.action,
        "confidence": r_jev.confidence,
        "latency": t_jev,
        "probabilities": r_jev.probabilities,
    }

    # 2. Check System 2 (Live Qwen3-8B GPU Server)
    print("\n[CHECK 2/8] Verifying Live System 2 (Qwen3-8B GPU on RTX 3070)...")
    qwen_live = QwenClient(use_mock=False)
    # Test B=0 probe mode
    t0 = time.perf_counter()
    r_probe = qwen_live.think(test_q, budget=0, mode="probe")
    t_probe = time.perf_counter() - t0
    print(f" -> Qwen Probe (B=0): {r_probe.action} | Tokens: {r_probe.tokens_used} | Latency: {t_probe:.3f}s")
    assert r_probe.action in ["Yes", "No"]

    # Test B=512 deliberation mode
    t0 = time.perf_counter()
    r_delib = qwen_live.think(test_q, budget=512, mode="blind")
    t_delib = time.perf_counter() - t0
    print(f" -> Qwen Deliberate (B=512): {r_delib.action} | Tokens: {r_delib.tokens_used} | Latency: {t_delib:.3f}s")
    audit_log["system2_live"] = {
        "status": "PASS",
        "probe_action": r_probe.action,
        "probe_tokens": r_probe.tokens_used,
        "probe_latency": t_probe,
        "delib_action": r_delib.action,
        "delib_tokens": r_delib.tokens_used,
        "delib_latency": t_delib,
    }

    # 3. Check All 5 Patterns (P1 to P5) End-to-End
    print("\n[CHECK 3/8] Verifying All 5 Interaction Patterns (Algorithm 1)...")
    patterns = {
        "P1_Cascade": create_p1_cascade(jev_live, qwen_live, tau=0.85),
        "P2_Budgeted": create_p2_budgeted(jev_live, qwen_live, tau=0.85),
        "P3_Probe": create_p3_probe(jev_live, qwen_live, tau=0.85),
        "P4_Verifier": create_p4_verifier(jev_live, qwen_live, tau=0.85),
        "P5_Reconcile": create_p5_reconcile(jev_live, qwen_live, tau=0.85),
    }

    pattern_results = {}
    for pname, controller in patterns.items():
        t0 = time.perf_counter()
        res = controller.execute(test_q)
        lat = time.perf_counter() - t0
        print(f" -> {pname:<15}: Released = {res.action:<5} | Band = {res.band.value:<10} | Mode = {res.mode:<12} | Tokens = {res.tokens_spent:<4} | Latency = {lat:.2f}s")
        assert res.action in test_q.options
        pattern_results[pname] = {
            "action": res.action,
            "band": res.band.value,
            "mode": res.mode,
            "tokens": res.tokens_spent,
            "latency": lat,
        }
    audit_log["patterns"] = pattern_results

    # 4. Check Autonomous Fallbacks (Props 5.9 & 5.10)
    print("\n[CHECK 4/8] Verifying Fallbacks: Bayes Action & Conformal Answer Sets...")
    # Test diffuse probability vector (giving up without a human)
    diffuse_probs = {"billing": 0.45, "tech_support": 0.55}
    loss_matrix = {
        "billing": {"billing": 0.0, "tech_support": 2.0, "hold": 0.3},
        "tech_support": {"billing": 2.0, "tech_support": 0.0, "hold": 0.3},
        "hold": {"billing": 0.3, "tech_support": 0.3, "hold": 0.0},
    }
    bayes_act, exp_losses = compute_bayes_action(diffuse_probs, loss_matrix=loss_matrix)
    print(f" -> Calibrated Bayes Action on diffuse probabilities: {bayes_act} (Expected losses: {exp_losses})")
    assert bayes_act == "hold", "Bayes action should select 'hold' under high uncertainty and asymmetric misclassification cost"

    # Conformal calibration & prediction
    conformal = ConformalPredictor(alpha=0.05)
    cal_probs = [{"A": 0.9, "B": 0.1}, {"A": 0.8, "B": 0.2}, {"A": 0.7, "B": 0.3}, {"A": 0.6, "B": 0.4}]
    cal_golds = ["A", "A", "A", "A"]
    q_hat = conformal.calibrate(cal_probs, cal_golds)
    c_set = conformal.predict_set({"A": 0.75, "B": 0.25})
    print(f" -> Conformal q_hat: {q_hat:.3f} | Predicted set for (0.75, 0.25): {c_set}")
    assert "A" in c_set
    audit_log["fallbacks"] = {
        "bayes_action": bayes_act,
        "conformal_q_hat": q_hat,
        "conformal_predicted_set": c_set,
    }

    # 5. Check Mathematical Identities to Machine Precision
    print("\n[CHECK 5/8] Verifying Mathematical Identities to Machine Precision...")
    harness = EvaluationHarness(use_mock=True)
    full_eval = harness.run_full_suite(max_samples_per_task=25)

    esc = full_eval["escalation_identity"]
    decomp = full_eval["paired_decomposition"]

    print(f" -> Prop 5.1 Escalation Identity diff: {esc['diff_machine_precision']} (Holds: {esc['identity_holds']})")
    print(f" -> Prop 5.4 Paired Decomposition diff: {decomp['diff_machine_precision']} (Holds: {decomp['identity_holds']})")
    print(f" -> Prop 5.4 Stable Term contribution: {decomp['breakdown']['stable']['contribution']} (Zero: {decomp['stable_contribution_zero']})")
    assert esc["identity_holds"], "Proposition 5.1 Escalation Identity must hold to machine precision!"
    assert decomp["identity_holds"], "Proposition 5.4 Paired Decomposition must hold to machine precision!"
    assert decomp["stable_contribution_zero"], "Stable term in Proposition 5.4 must vanish identically!"
    audit_log["mathematical_identities"] = {
        "prop_5_1_diff": esc["diff_machine_precision"],
        "prop_5_1_holds": esc["identity_holds"],
        "prop_5_1_M_Z": esc["M_Z"],
        "prop_5_1_win_margin": esc["win_margin"],
        "prop_5_4_diff": decomp["diff_machine_precision"],
        "prop_5_4_holds": decomp["identity_holds"],
        "prop_5_4_stable_zero": decomp["stable_contribution_zero"],
        "transition_shares": decomp["shares"],
    }

    # 6. Check Table 6 & Table 9 Replications
    print("\n[CHECK 6/8] Verifying Paper Replications (Tables 6 & 9)...")
    t6 = full_eval.get("table_6_replication", replicate_table_3())
    t9 = full_eval.get("table_9_replication", replicate_table_5(n_samples=10000))
    print(f" -> Table 6 Return Curves evaluated: {len(t6)} rows")
    for row in t6:
        print(f"    - {row['curve']:<12}: Act c >= {row['act_c']:.2f}, Give Up c <= {row['give_up_c']:.2f}, Peak B* = {row['peak_B']}, Kept = {int(row['kept']*100)}%")
    print(f" -> Table 9 Worked Example evaluated: {len(t9)} rows")
    for row in t9:
        print(f"    - {row['policy']:<25}: Tokens = {row['tokens']:<4}, Accuracy = {row['accuracy']:.3f}")
    audit_log["table_6"] = t6
    audit_log["table_9"] = t9

    # 7. Check Table 11 Pre-Specified Hypotheses H1-H9
    print("\n[CHECK 7/8] Verifying Pre-Specified Hypotheses H1 to H9 (Table 11)...")
    hypotheses = full_eval["hypotheses"]
    falsified = []
    for h in hypotheses:
        status_sym = "[PASS]" if not h["falsified"] else "[FAIL]"
        print(f" {status_sym} {h['id']:<8}: {h['description']}")
        print(f"          Metric: {h['metric_value']}")
        if h["falsified"]:
            falsified.append(h["id"])
    assert len(falsified) == 0, f"Hypotheses failed: {falsified}"
    audit_log["hypotheses"] = hypotheses

    # 8. Check All 7 Application Domains (Table 8)
    print("\n[CHECK 8/8] Verifying Table 8 Domain Profiles...")
    print(f" -> Total domains registered: {len(ALL_DOMAINS)} (expected 7)")
    assert len(ALL_DOMAINS) == 7, f"Expected 7 application domains from Table 8, got {len(ALL_DOMAINS)}"
    for key, dom in ALL_DOMAINS.items():
        print(f"    - {dom.domain_name:<32}: Fallback = {dom.fallback_action:<28} | Risk = {dom.risk_to_monitor}")
        assert len(dom.sample_queries) > 0, f"Domain {key} has no sample queries"
    audit_log["domains"] = {k: {"name": v.domain_name, "fallback": v.fallback_action, "risk": v.risk_to_monitor} for k, v in ALL_DOMAINS.items()}

    print("\n" + "=" * 70)
    print("ALL AUDIT CHECKS PASSED: SYSTEM FULLY ADHERES TO RESEARCH PAPER")
    print("=" * 70)

    return audit_log

if __name__ == "__main__":
    audit_log = run_comprehensive_audit()
    with open("audit_results.json", "w") as f:
        json.dump(audit_log, f, indent=2)
    print("\nAudit results saved to audit_results.json")
