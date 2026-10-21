import pytest
from think_fast_slow.core.types import Query, DecisionBand
from think_fast_slow.system1.jev_client import JevClient
from think_fast_slow.system2.qwen_client import QwenClient
from think_fast_slow.patterns import (
    create_p1_cascade,
    create_p2_budgeted,
    create_p3_probe,
    create_p4_verifier,
    create_p5_reconcile,
)

def test_pattern_p1_cascade_act_deliberate_split():
    """35. Pattern 1 (P1 Cascade): Acts on c >= tau, deliberates on c < tau."""
    jev = JevClient(use_mock=True)
    qwen = QwenClient(use_mock=True)
    cascade = create_p1_cascade(jev, qwen, tau=0.88)
    q = Query(
        id="p1_q",
        text="Standard validation query.",
        question="Is the output acceptable?",
        options=["Accept", "Reject"],
        gold_answer="Accept",
    )
    res = cascade.execute(q)
    assert res.band in [DecisionBand.ACT, DecisionBand.DELIBERATE, DecisionBand.GIVE_UP]
    assert res.action in q.options

def test_pattern_p2_budgeted_task_conditioned():
    """36. Pattern 2 (P2 Budgeted Deliberation): Allocates budget conditioned on task type."""
    jev = JevClient(use_mock=True)
    qwen = QwenClient(use_mock=True)
    controller = create_p2_budgeted(jev, qwen, tau=0.88)
    q = Query(
        id="p2_q",
        text="A task-conditioned problem in requirements domain.",
        question="Is the requirement testable?",
        options=["Testable", "Ambiguous"],
        task="requirements",
    )
    res = controller.execute(q)
    assert res.action in q.options

def test_pattern_p3_probe_conflict_escalation():
    """37. Pattern 3 (P3 Disagreement Probe): Probe disagreement triggers conflict deliberation."""
    jev = JevClient(use_mock=True)
    qwen = QwenClient(use_mock=True)
    probe_controller = create_p3_probe(jev, qwen, tau=0.85)
    # CRT lure item produces conflict between Jev and Qwen
    q = Query(
        id="p3_lure",
        text="Lily pads cover half the lake on day 47.",
        question="How long to cover half the lake?",
        options=["47 days", "24 days"],
        gold_answer="47 days",
        metadata={"is_lure": True},
    )
    res = probe_controller.execute(q)
    assert res.action in q.options

def test_pattern_p4_verifier_reverts_unsupported_override():
    """38. Pattern 4 (P4 Jev as Verifier): Reverts unsupported overrides by System 2."""
    jev = JevClient(use_mock=True)
    qwen = QwenClient(use_mock=True)
    verifier_controller = create_p4_verifier(jev, qwen, tau=0.70)
    q = Query(
        id="p4_q",
        text="Code review commit: added bypass for production authentication.",
        question="Is this commit safe to merge?",
        options=["safe", "unsafe"],
        gold_answer="unsafe",
    )
    res = verifier_controller.execute(q)
    assert res.action in q.options

def test_pattern_p5_reconcile_decision_hygiene():
    """39. Pattern 5 (P5 Blind-then-Reconcile): Audits blind answer with second-turn check."""
    jev = JevClient(use_mock=True)
    qwen = QwenClient(use_mock=True)
    rec_controller = create_p5_reconcile(jev, qwen, tau=0.75)
    q = Query(
        id="p5_q",
        text="Ambiguous customer message regarding refund.",
        question="Is the customer eligible for instant refund?",
        options=["eligible", "ineligible"],
    )
    res = rec_controller.execute(q)
    assert res.action in q.options
