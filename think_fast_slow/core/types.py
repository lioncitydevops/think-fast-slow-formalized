from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

class DecisionBand(str, Enum):
    ACT = "ACT"                  # High confidence: law of least effort (release a1)
    DELIBERATE = "DELIBERATE"    # Intermediate confidence: allocate thinking budget B*
    GIVE_UP = "GIVE_UP"          # Low confidence / ambiguous: hopeless, trigger Fallback(p)

class TransitionType(str, Enum):
    STABLE = "stable"            # i == b (informed matches blind)
    ANCHOR = "anchor"            # b != a1, i == a1 (hint pulls System 2 to System 1)
    REACTANCE = "reactance"      # b == a1, i != a1 (hint pushes System 2 away)
    DRIFT = "drift"              # b != a1, i not in {a1, b} (hint derails reasoning to 3rd option)

class ReturnCurveType(str, Enum):
    SATURATING = "saturating"    # h(B) = 1 - exp(-B/beta)
    LOG_LINEAR = "log_linear"    # h(B) = min(1, ln(1 + B/beta) / ln(1 + Bmax/beta))
    EARLY_FLAT = "early_flat"    # Quick saturation then flat plateau
    THRESHOLD = "threshold"      # S-shaped curve with inflection point
    INVERTED_U = "inverted_u"    # Inverse scaling: accuracy drops after peak

@dataclass
class Query:
    id: str
    text: str                               # Text state x
    question: str                           # Typed question instruction
    options: List[str]                      # Option space A (|A| = K)
    gold_answer: Optional[str] = None       # Gold label y
    task: Optional[str] = "general"         # Task category
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class JevResponse:
    action: str                             # a1
    confidence: float                       # c = max_k p_k
    probabilities: Dict[str, float]         # p vector
    latency: float = 0.0
    task_label: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

@dataclass
class QwenResponse:
    action: str                             # a2 (or b or a0)
    reasoning_trace: str                    # <think> reasoning tokens
    budget_requested: int                   # B (0 for non-thinking probe)
    tokens_used: int                        # Total tokens consumed
    latency: float = 0.0
    mode: str = "blind"                     # blind, informed, reconcile, probe
    raw_content: Optional[str] = None

@dataclass
class ControllerResult:
    query_id: str
    action: str                             # Final released action
    band: DecisionBand                      # ACT, DELIBERATE, GIVE_UP
    mode: str                               # ACT, PROBE_ACT, DELIBERATE, GIVE_UP_FALLBACK, VERIFIER_REVERT, RECONCILED
    tokens_spent: int
    total_latency: float
    jev: JevResponse
    qwen: Optional[QwenResponse] = None
    conformal_set: Optional[List[str]] = None
    is_correct: Optional[bool] = None
    notes: Dict[str, Any] = field(default_factory=dict)
