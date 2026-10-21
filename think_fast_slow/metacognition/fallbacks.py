import math
from typing import Dict, List, Optional, Any, Set, Tuple
import numpy as np

Tuple_Action = Tuple[str, Dict[str, float]]

def compute_bayes_action(
    probabilities: Dict[str, float],
    loss_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    hold_cost: float = 0.40,
    action_space: Optional[List[str]] = None,
) -> Tuple_Action:
    """
    Proposition 5.9 (Calibrated Bayes Action).
    a*(p) = argmin_{a in A U {hold}} sum_k p_k * L(a, k).
    Minimizes expected loss among all rules that depend on the query only through p.
    """
    options = list(probabilities.keys())
    if action_space is None:
        actions = list(options) + ["hold"]
    else:
        actions = list(action_space)
        if "hold" not in actions:
            actions.append("hold")

    # Default 0-1 loss with hold penalty if no matrix provided
    if loss_matrix is None:
        loss_matrix = {}
        for a in actions:
            loss_matrix[a] = {}
            for k in options:
                if a == "hold":
                    loss_matrix[a][k] = hold_cost
                elif a == k:
                    loss_matrix[a][k] = 0.0
                else:
                    loss_matrix[a][k] = 1.0  # Misclassification cost

    best_action = actions[0]
    min_expected_loss = float("inf")
    expected_losses = {}

    for a in actions:
        exp_loss = sum(probabilities.get(k, 0.0) * loss_matrix[a].get(k, 1.0) for k in options)
        expected_losses[a] = float(exp_loss)
        if exp_loss < min_expected_loss:
            min_expected_loss = exp_loss
            best_action = a

    return best_action, expected_losses

class ConformalPredictor:
    """
    Proposition 5.10 (Conformal Answer Sets - Sadinle et al. 2019).
    With n exchangeable labelled items and q_hat the ceil((n+1)(1-alpha))-th smallest
    score 1 - p_y, the set C(x) = {k : 1 - p_k(x) <= q_hat} satisfies P(y in C(x)) >= 1 - alpha.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = 1.0
        self.calibration_scores: List[float] = []

    def calibrate(
        self,
        prob_vectors: List[Dict[str, float]],
        gold_labels: List[str],
    ) -> float:
        """
        Compute conformal threshold q_hat on calibration set.
        """
        n = len(gold_labels)
        if n == 0:
            self.q_hat = 1.0
            return 1.0

        scores = []
        for probs, gold in zip(prob_vectors, gold_labels):
            p_y = probs.get(gold, 0.0)
            score = 1.0 - p_y
            scores.append(score)

        scores.sort()
        self.calibration_scores = scores

        # Index ceil((n + 1) * (1 - alpha)) - 1 (0-indexed)
        k_idx = math.ceil((n + 1) * (1.0 - self.alpha)) - 1
        k_idx = min(n - 1, max(0, k_idx))
        self.q_hat = float(scores[k_idx])
        return self.q_hat

    def predict_set(self, probabilities: Dict[str, float]) -> List[str]:
        """
        Return conformal answer set C(x) = {k : 1 - p_k(x) <= q_hat}.
        """
        ans_set = [
            k for k, p in probabilities.items()
            if (1.0 - p) <= self.q_hat
        ]
        # In case set is empty, return argmax
        if not ans_set:
            best_opt = max(probabilities.keys(), key=lambda k: probabilities[k])
            ans_set = [best_opt]
        return ans_set

    def evaluate_coverage(
        self,
        test_probs: List[Dict[str, float]],
        test_golds: List[str],
    ) -> Dict[str, Any]:
        """
        Compute empirical coverage and average set size on test items.
        """
        n = len(test_golds)
        if n == 0:
            return {"coverage": 1.0, "avg_set_size": 1.0}

        covered = 0
        total_size = 0

        for probs, gold in zip(test_probs, test_golds):
            c_set = self.predict_set(probs)
            if gold in c_set:
                covered += 1
            total_size += len(c_set)

        coverage = covered / n
        avg_size = total_size / n

        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(coverage),
            "coverage_guarantee_met": bool(coverage >= (1.0 - self.alpha - 0.02)),
            "average_set_size": float(avg_size),
            "q_hat": float(self.q_hat),
        }

Tuple_Action = Tuple[str, Dict[str, float]]
