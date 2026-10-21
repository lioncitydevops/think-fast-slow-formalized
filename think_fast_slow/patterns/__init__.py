from typing import Optional
from .controller import DualProcessController
from ..system1.jev_client import JevClient
from ..system2.qwen_client import QwenClient
from ..core.types import Query, ControllerResult
from ..metacognition.effort_allocation import NonparametricBudgetTable

def create_p1_cascade(
    jev: JevClient,
    qwen: QwenClient,
    tau: float = 0.90,
    fixed_budget: int = 1024,
) -> DualProcessController:
    """
    Pattern 1 (P1 Cascade): Default-interventionist.
    Deliberates at a fixed budget when c < tau.
    """
    controller = DualProcessController(
        jev_client=jev,
        qwen_client=qwen,
        tau=tau,
        probe=False,
        reconcile=False,
        verify=False,
    )
    return controller

def create_p2_budgeted(
    jev: JevClient,
    qwen: QwenClient,
    tau: float = 0.90,
    budget_table: Optional[NonparametricBudgetTable] = None,
) -> DualProcessController:
    """
    Pattern 2 (P2 Budgeted Deliberation):
    Deliberates with a task-conditioned budget estimated nonparametrically.
    """
    return DualProcessController(
        jev_client=jev,
        qwen_client=qwen,
        tau=tau,
        budget_table=budget_table,
        probe=False,
        reconcile=False,
        verify=False,
    )

def create_p3_probe(
    jev: JevClient,
    qwen: QwenClient,
    tau: float = 0.90,
) -> DualProcessController:
    """
    Pattern 3 (P3 Disagreement Probe): Parallel-competitive.
    Runs Qwen3-8B in non-thinking mode (B=0) alongside Jev, and thinks
    only on disagreement or when c < tau.
    """
    return DualProcessController(
        jev_client=jev,
        qwen_client=qwen,
        tau=tau,
        probe=True,
        reconcile=False,
        verify=False,
    )

def create_p4_verifier(
    jev: JevClient,
    qwen: QwenClient,
    tau: float = 0.90,
) -> DualProcessController:
    """
    Pattern 4 (P4 Jev as Verifier):
    When Qwen overturns Jev, Jev judges whether the new answer is supported.
    If not, the override is reverted.
    """
    return DualProcessController(
        jev_client=jev,
        qwen_client=qwen,
        tau=tau,
        probe=False,
        reconcile=False,
        verify=True,
    )

def create_p5_reconcile(
    jev: JevClient,
    qwen: QwenClient,
    tau: float = 0.90,
) -> DualProcessController:
    """
    Pattern 5 (P5 Blind-then-Reconcile):
    System 2 reasons blind, then sees (a1, c) in a short second turn
    and may revise only on finding a concrete error.
    """
    return DualProcessController(
        jev_client=jev,
        qwen_client=qwen,
        tau=tau,
        probe=False,
        reconcile=True,
        verify=False,
    )
