import pytest
from think_fast_slow.core.types import Query, QwenResponse
from think_fast_slow.system2.qwen_client import QwenClient
from think_fast_slow.system2.parser import extract_answer

def test_qwen_probe_mode_zero_budget():
    """9. Non-thinking probe mode (B=0) runs with low token cost (~24 tokens)."""
    client = QwenClient(use_mock=True)
    query = Query(
        id="probe_q",
        text="Water is composed of hydrogen and oxygen.",
        question="Is this statement correct?",
        options=["Yes", "No"],
        gold_answer="Yes",
    )
    resp = client.think(query, budget=0, mode="probe")
    assert resp.mode == "probe"
    assert resp.budget_requested == 0
    assert resp.tokens_used <= 35
    assert resp.action in ["Yes", "No"]

def test_qwen_thinking_mode_budget_cap():
    """10. Thinking mode allocates tokens bounded by requested budget."""
    client = QwenClient(use_mock=True)
    query = Query(
        id="delib_q",
        text="A logic puzzle requiring step by step reasoning.",
        question="Select the correct outcome:",
        options=["True", "False"],
        gold_answer="True",
    )
    resp = client.think(query, budget=1024, mode="blind")
    assert resp.mode == "blind"
    assert resp.budget_requested == 1024
    assert resp.tokens_used <= 1024
    assert "Thinking Trace" in resp.reasoning_trace

def test_qwen_blind_mode_no_hint():
    """11. Blind deliberation executes independently without System 1 hints."""
    client = QwenClient(use_mock=True)
    query = Query(
        id="blind_q",
        text="Independent judgment check.",
        question="Choose A or B:",
        options=["A", "B"],
        gold_answer="A",
    )
    resp = client.think(query, budget=512, mode="blind")
    assert resp.action in ["A", "B"]

def test_qwen_informed_mode_with_hint():
    """12. Informed mode processes hint (a1, c) and models anchoring rationalization."""
    client = QwenClient(use_mock=True)
    query = Query(
        id="informed_q",
        text="Anchoring test query.",
        question="Choose Option 1 or Option 2:",
        options=["Option 1", "Option 2"],
        gold_answer="Option 1",
    )
    resp = client.think(query, budget=1024, mode="informed", hint_a1="Option 2", hint_c=0.90)
    assert resp.mode == "informed"
    assert resp.action in query.options

def test_qwen_reconcile_mode_two_turns():
    """13. Reconcile mode executes short second-turn decision hygiene."""
    client = QwenClient(use_mock=True)
    query = Query(
        id="rec_q",
        text="Decision hygiene review query.",
        question="Select valid conclusion:",
        options=["Valid", "Invalid"],
        gold_answer="Valid",
    )
    resp = client.think(query, budget=1024, mode="reconcile", hint_a1="Invalid", hint_c=0.70)
    assert resp.mode == "reconcile"
    assert resp.action in query.options

def test_qwen_response_parsing_answer_tag():
    """14. Answer extraction parser extracts choice from reasoning content and Answer: tags."""
    options = ["Alpha", "Beta", "Gamma"]
    raw_text = "<think>Let me evaluate options. Option Beta is inconsistent with premise 1.</think>\nAnswer: Alpha"
    action = extract_answer(raw_text, options)
    assert action == "Alpha"

    # Fallback to last line
    raw_text2 = "Reasoning concluding with selection.\nBeta"
    action2 = extract_answer(raw_text2, options)
    assert action2 == "Beta"
