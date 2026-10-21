import pytest
import numpy as np
from typesafe_sdk import Choice
from think_fast_slow.core.types import Query, JevResponse
from think_fast_slow.system1.jev_client import JevClient

def test_jev_request_schema_choice():
    """1. Test request building for Choice question against vendor documentation."""
    query = Query(
        id="q_choice",
        text="Applicant has 3 years of engineering experience.",
        question="Select job qualification tier:",
        options=["senior", "mid", "junior"],
    )
    choice_obj = Choice(
        instructions=query.question,
        criteria={opt: None for opt in query.options},
    )
    assert choice_obj.instructions == "Select job qualification tier:"
    assert set(choice_obj.criteria.keys()) == {"senior", "mid", "junior"}

def test_jev_request_schema_yes_no():
    """2. Test request building for yes/no question against vendor documentation."""
    query = Query(
        id="q_yn",
        text="The quick brown fox jumps over the lazy dog.",
        question="Is the animal that jumps a fox?",
        options=["Yes", "No"],
    )
    choice_obj = Choice(
        instructions=query.question,
        criteria={"Yes": None, "No": None},
    )
    assert len(choice_obj.criteria) == 2
    assert "Yes" in choice_obj.criteria

def test_jev_auxiliary_task_question():
    """3. Test auxiliary question bundle in same single API call."""
    client = JevClient(use_mock=True)
    query = Query(
        id="q_aux",
        text="Database connection timed out during peak query traffic.",
        question="Is this a critical production incident?",
        options=["Yes", "No"],
    )
    aux_q = ("Classify service domain:", ["database", "network", "auth"])
    resp = client.judge(query, auxiliary_task_question=aux_q)
    assert resp.action in ["Yes", "No"]
    assert resp.task_label in ["database", "network", "auth", "general"]

def test_jev_choice_response_parsing():
    """4. Test response parsing of choice, confidence, and probability map."""
    client = JevClient(use_mock=True)
    query = Query(
        id="q_parse",
        text="Test passage",
        question="Test question",
        options=["A", "B", "C"],
        gold_answer="A",
    )
    resp = client.judge(query)
    assert resp.action in ["A", "B", "C"]
    assert 0.5 <= resp.confidence <= 1.0
    assert isinstance(resp.probabilities, dict)
    assert len(resp.probabilities) == 3

def test_jev_mock_confidence_calibration():
    """5. Statistical check that P(a1 = y | c) = c under calibration."""
    client = JevClient(use_mock=True)
    n_samples = 200
    queries = [
        Query(
            id=f"cal_{i}",
            text=f"Calibration passage {i}",
            question="Select truth value:",
            options=["True", "False"],
            gold_answer="True",
        )
        for i in range(n_samples)
    ]
    responses = [client.judge(q) for q in queries]
    confs = [r.confidence for r in responses]
    accs = [r.action == q.gold_answer for r, q in zip(responses, queries)]
    # Binned calibration: high confidence bin (c >= 0.80) should have high accuracy
    high_idx = [i for i, c in enumerate(confs) if c >= 0.80]
    high_acc = np.mean([accs[i] for i in high_idx])
    high_conf = np.mean([confs[i] for i in high_idx])
    assert abs(high_acc - high_conf) < 0.15

def test_jev_mock_wysiati_lure_vulnerability():
    """6. Test that CRT lure queries elicit confident wrong answers (WYSIATI)."""
    client = JevClient(use_mock=True)
    lure_query = Query(
        id="crt_lure_test",
        text="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball.",
        question="How much does the ball cost?",
        options=["$0.10", "$0.05"],
        gold_answer="$0.05",
        metadata={"is_lure": True},
    )
    resp = client.judge(lure_query)
    assert resp.confidence >= 0.80
    assert resp.action == "$0.10"  # Intuitive lure answer

def test_jev_error_handling_fail_closed():
    """7. Test error handling fail closed on missing/invalid client setup."""
    client = JevClient(use_mock=False, api_key="invalid_dummy_key", base_url="http://127.0.0.1:9999/v1")
    query = Query(
        id="q_err",
        text="Test fail closed",
        question="Will this fail?",
        options=["Yes", "No"],
    )
    with pytest.raises(Exception):
        client.judge(query)

def test_jev_classwise_probability_vector():
    """8. Test probability vector normalization."""
    client = JevClient(use_mock=True)
    query = Query(
        id="q_norm",
        text="Normalizing probability vector test",
        question="Pick a category",
        options=["Cat1", "Cat2", "Cat3", "Cat4"],
    )
    resp = client.judge(query)
    total_p = sum(resp.probabilities.values())
    assert abs(total_p - 1.0) < 1e-4
