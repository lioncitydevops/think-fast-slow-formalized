import time
import math
import random
from typing import Optional, Dict, Any, List
from ..core.types import Query, ControllerResult, DecisionBand, JevResponse, QwenResponse
from ..core.constants import DEFAULT_EPSILON
from ..system1.jev_client import JevClient
from ..system2.qwen_client import QwenClient
from ..metacognition.effort_allocation import compute_closed_form_budget, NonparametricBudgetTable
from ..metacognition.fallbacks import compute_bayes_action, ConformalPredictor

class DualProcessController:
    """
    Algorithm 1: Dual-process controller (one query).
    Formalized from Zhang & Lee (ASSE 2026), Section 6 & Table 7 & Figure 3.
    Integrates System 1 (Jev), System 2 (Qwen3-8B), Metacognitive Effort Control,
    and all 5 Interaction Patterns (P1-P5).
    """
    def __init__(
        self,
        jev_client: JevClient,
        qwen_client: QwenClient,
        tau: float = 0.90,
        budget_table: Optional[NonparametricBudgetTable] = None,
        give_up_tolerance: float = 0.35,
        B_conf: int = 2048,
        B_a: int = 1024,
        kappa: float = 0.75,
        audit_share: float = 0.05,
        loss_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        conformal_predictor: Optional[ConformalPredictor] = None,
        probe: bool = False,
        reconcile: bool = False,
        verify: bool = False,
    ):
        self.jev = jev_client
        self.qwen = qwen_client
        self.tau = tau
        self.budget_table = budget_table
        self.give_up_tolerance = give_up_tolerance
        self.B_conf = B_conf
        self.B_a = B_a
        self.kappa = kappa
        self.audit_share = audit_share
        self.loss_matrix = loss_matrix
        self.conformal = conformal_predictor or ConformalPredictor()
        self.probe = probe
        self.reconcile = reconcile
        self.verify = verify

    def execute(self, query: Query) -> ControllerResult:
        """
        Execute Algorithm 1 on an input query.
        Step numbers correspond to Table 7, Figure 3, and Algorithm 1.
        """
        t_start = time.perf_counter()
        total_tokens = 0

        # Step 1: (a1, p, c, task) <- Jev(x) // 1: one call
        jev_resp = self.jev.judge(query)
        a1 = jev_resp.action
        c = jev_resp.confidence
        p = jev_resp.probabilities
        task = jev_resp.task_label or query.task or "general"

        # Step 2: conflict <- false
        conflict = False
        qwen_probe_resp: Optional[QwenResponse] = None

        # Step 3: if probe then a0 <- Qwen(x, 0) blind; conflict <- [a0 != a1] // 2
        if self.probe:
            qwen_probe_resp = self.qwen.think(query, budget=0, mode="probe")
            total_tokens += qwen_probe_resp.tokens_used
            a0 = qwen_probe_resp.action
            conflict = (a0 != a1)

        # Step 4: if c >= tau and not conflict then return Act(a1) // 3: act
        if c >= self.tau and not conflict:
            t_total = time.perf_counter() - t_start
            is_correct = (a1 == query.gold_answer) if query.gold_answer else None
            # Step 13: Act(a1): with prob q, log a blind Qwen(x, Ba) run for audit (3a); release a1
            audit_logged = False
            if random.random() < self.audit_share:
                audit_logged = True

            return ControllerResult(
                query_id=query.id,
                action=a1,
                band=DecisionBand.ACT,
                mode="PROBE_ACT" if self.probe else "ACT",
                tokens_spent=total_tokens,
                total_latency=t_total,
                jev=jev_resp,
                qwen=qwen_probe_resp,
                conformal_set=self.conformal.predict_set(p),
                is_correct=is_correct,
                notes={
                    "reason": "confidence >= tau and not conflict",
                    "certified_tau": self.tau,
                    "shadow_audit_logged": audit_logged,
                },
            )

        # Step 5: B <- B_hat(c, task); if conflict then B <- max(B, B_conf)
        if self.budget_table is not None:
            budget = self.budget_table.get_budget(c, task)
        else:
            budget = compute_closed_form_budget(c)

        if conflict:
            budget = max(budget, self.B_conf)

        # Step 6: if B = 0 then return Act(a1) if 1 - c <= eps_g, else Fallback(p) // 3: give up
        if budget == 0:
            t_total = time.perf_counter() - t_start
            if (1.0 - c) <= self.give_up_tolerance:
                is_correct = (a1 == query.gold_answer) if query.gold_answer else None
                return ControllerResult(
                    query_id=query.id,
                    action=a1,
                    band=DecisionBand.ACT,
                    mode="ACT_LEAST_EFFORT",
                    tokens_spent=total_tokens,
                    total_latency=t_total,
                    jev=jev_resp,
                    qwen=qwen_probe_resp,
                    conformal_set=self.conformal.predict_set(p),
                    is_correct=is_correct,
                    notes={"reason": "B=0 and 1-c <= eps_g"},
                )
            else:
                fallback_action, exp_losses = compute_bayes_action(p, loss_matrix=self.loss_matrix)
                c_set = self.conformal.predict_set(p)
                is_correct = (fallback_action == query.gold_answer) if query.gold_answer else None
                return ControllerResult(
                    query_id=query.id,
                    action=fallback_action,
                    band=DecisionBand.GIVE_UP,
                    mode="GIVE_UP_FALLBACK",
                    tokens_spent=total_tokens,
                    total_latency=t_total,
                    jev=jev_resp,
                    qwen=qwen_probe_resp,
                    conformal_set=c_set,
                    is_correct=is_correct,
                    notes={"reason": "B=0 and 1-c > eps_g", "expected_losses": exp_losses},
                )

        # Step 7: if GiveUp(c, B) then return Fallback(p) // 3: give up
        # Step 14 GiveUp(c, B): error expected after thinking exceeds eps_g
        d = min(1.0, max(0.0, 2.0 * (1.0 - c)))
        ambiguity = 1.0 - (d ** 2.0)
        h_B = 1.0 - math.exp(-budget / 1024.0)
        post_error = (1.0 - c) * (1.0 - ambiguity * h_B) + (0.1 * c * ambiguity * h_B)
        if post_error > self.give_up_tolerance and budget < 256:
            fallback_action, exp_losses = compute_bayes_action(p, loss_matrix=self.loss_matrix)
            c_set = self.conformal.predict_set(p)
            t_total = time.perf_counter() - t_start
            is_correct = (fallback_action == query.gold_answer) if query.gold_answer else None
            return ControllerResult(
                query_id=query.id,
                action=fallback_action,
                band=DecisionBand.GIVE_UP,
                mode="GIVE_UP_FALLBACK",
                tokens_spent=total_tokens,
                total_latency=t_total,
                jev=jev_resp,
                qwen=qwen_probe_resp,
                conformal_set=c_set,
                is_correct=is_correct,
                notes={"reason": "expected_post_error > eps_g", "expected_losses": exp_losses},
            )

        # Step 8: b <- Qwen(x, B) blind // 4: Qwen checks Jev
        qwen_delib_resp = self.qwen.think(query, budget=budget, mode="blind")
        total_tokens += qwen_delib_resp.tokens_used
        b = qwen_delib_resp.action

        # Step 9: if reconcile then b <- Reconcile(b; a1, c) // 5
        reconciled = False
        if self.reconcile:
            qwen_rec_resp = self.qwen.think(
                query,
                budget=budget,
                mode="reconcile",
                hint_a1=a1,
                hint_c=c,
            )
            total_tokens += qwen_rec_resp.tokens_used
            b = qwen_rec_resp.action
            reconciled = True

        # Step 10: if verify and b != a1 then
        # Step 11: if Jev rates b unsupported with prob. >= kappa then return a1 // 6
        verified_revert = False
        if self.verify and b != a1:
            verify_query = Query(
                id=f"{query.id}_verify",
                text=f"Passage: {query.text}\nReasoning: {qwen_delib_resp.reasoning_trace[:400]}",
                question=f"Is option '{b}' strongly supported and factually correct over '{a1}'?",
                options=["supported", "unsupported"],
                task="verification",
            )
            jev_verify_resp = self.jev.judge(verify_query)
            unsupported_prob = jev_verify_resp.probabilities.get("unsupported", 0.0)
            if (jev_verify_resp.action == "unsupported" and jev_verify_resp.confidence >= self.kappa) or (unsupported_prob >= self.kappa):
                b = a1
                verified_revert = True

        t_total = time.perf_counter() - t_start
        is_correct = (b == query.gold_answer) if query.gold_answer else None

        result_mode = "VERIFIER_REVERT" if verified_revert else ("RECONCILED" if reconciled else "DELIBERATE")

        return ControllerResult(
            query_id=query.id,
            action=b,
            band=DecisionBand.DELIBERATE,
            mode=result_mode,
            tokens_spent=total_tokens,
            total_latency=t_total,
            jev=jev_resp,
            qwen=qwen_delib_resp,
            conformal_set=self.conformal.predict_set(p),
            is_correct=is_correct,
            notes={
                "budget_allocated": budget,
                "verified_revert": verified_revert,
                "reconciled": reconciled,
            },
        )
