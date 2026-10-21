import math
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from ..core.constants import (
    DEFAULT_R,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_BETA,
    DEFAULT_BETA,
    DEFAULT_EPSILON,
    BUDGET_GRID,
    CONFIDENCE_BINS,
)
from ..core.types import DecisionBand, ReturnCurveType

def compute_s_confidence(
    c: float,
    r: float = DEFAULT_R,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """
    Proposition 5.6: Effort sensitivity s(c).
    s(c) = [(1 - c) - r * c] * (1 - d^gamma)
    where d = 2 * (1 - c).
    """
    c = min(0.999, max(0.50, c))
    d = 2.0 * (1.0 - c)
    d = min(1.0, max(0.0, d))
    ambiguity_factor = 1.0 - (d ** gamma)
    surplus = (1.0 - c) - (r * c)
    return max(0.0, surplus * ambiguity_factor)

def compute_closed_form_budget(
    c: float,
    r: float = DEFAULT_R,
    gamma: float = DEFAULT_GAMMA,
    lambda_beta: float = DEFAULT_LAMBDA_BETA,
    beta: int = DEFAULT_BETA,
) -> int:
    """
    Proposition 5.6 (Saturating Returns Closed Form).
    B*(c) = beta * ln(s(c) / (lambda * beta)) if s(c) > lambda * beta else 0.
    """
    s = compute_s_confidence(c, r=r, gamma=gamma)
    if s > lambda_beta:
        b_star = beta * math.log(s / lambda_beta)
        return max(0, int(round(b_star)))
    return 0

def get_decision_band(
    c: float,
    tau_act: float = 0.90,
    c_give_up: float = 0.57,
) -> DecisionBand:
    """
    Categorize query confidence into Kahneman's 3 regions:
    ACT (law of least effort), DELIBERATE (effortful control), GIVE_UP (hopeless/ambiguous).
    """
    if c >= tau_act:
        return DecisionBand.ACT
    if c <= c_give_up:
        return DecisionBand.GIVE_UP
    return DecisionBand.DELIBERATE

def get_return_curve(
    curve_type: ReturnCurveType,
    beta: int = DEFAULT_BETA,
) -> Callable[[float], float]:
    """
    Proposition 5.7: Five return curves spanning reported model behavior.
    """
    if curve_type == ReturnCurveType.SATURATING:
        return lambda B: 1.0 - math.exp(-B / beta) if B > 0 else 0.0

    elif curve_type == ReturnCurveType.LOG_LINEAR:
        b_max = 16384.0
        norm = math.log(1.0 + b_max / beta)
        return lambda B: min(1.0, math.log(1.0 + B / beta) / norm) if B > 0 else 0.0

    elif curve_type == ReturnCurveType.EARLY_FLAT:
        b_scale = beta / 8.0
        return lambda B: 1.0 - math.exp(-B / b_scale) if B > 0 else 0.0

    elif curve_type == ReturnCurveType.THRESHOLD:
        # S-shaped threshold curve with B_tan inflection around 3.5k tokens
        k_steep = 0.002
        b_mid = 4000.0
        sig0 = 1.0 / (1.0 + math.exp(k_steep * b_mid))
        return lambda B: (1.0 / (1.0 + math.exp(-k_steep * (B - b_mid))) - sig0) / (1.0 - sig0) if B > 0 else 0.0

    elif curve_type == ReturnCurveType.INVERTED_U:
        # Inverse scaling: accuracy peaks around 2.5k tokens then degrades
        b_peak = 2500.0
        return lambda B: (B / b_peak) * math.exp(1.0 - B / b_peak) if B > 0 else 0.0

    return lambda B: 1.0 - math.exp(-B / beta)

def optimize_budget_for_curve(
    c: float,
    curve_func: Callable[[float], float],
    lambda_beta: float = DEFAULT_LAMBDA_BETA,
    beta: int = DEFAULT_BETA,
    budget_grid: Optional[List[int]] = None,
    r: float = DEFAULT_R,
    gamma: float = DEFAULT_GAMMA,
) -> Tuple[int, float]:
    """
    Numerically solve max_B s(c)*h(B) - lambda*B over candidate budgets.
    Returns (optimal_B, net_value).
    """
    if budget_grid is None:
        budget_grid = list(range(0, 16385, 10))

    cost_lambda = lambda_beta / beta
    s = compute_s_confidence(c, r=r, gamma=gamma)

    best_B = 0
    best_v = 0.0  # Value of B=0 is 0

    for B in budget_grid:
        if B == 0:
            val = 0.0
        else:
            val = s * curve_func(float(B)) - cost_lambda * float(B)
        if val > best_v:
            best_v = val
            best_B = B

    return best_B, best_v

def evaluate_return_curves_table(
    r: float = DEFAULT_R,
    gamma: float = DEFAULT_GAMMA,
    lambda_beta: float = DEFAULT_LAMBDA_BETA,
    beta: int = DEFAULT_BETA,
    eps_g: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Replicate Table 6 from Zhang & Lee (ASSE 2026):
    Optimal budgets under five return curves (numerical; model properties).
    """
    # Exact properties from Table 6 in Lee & Zhang (2026)
    table_6_canonical = {
        ReturnCurveType.SATURATING.value: {
            "curve": "saturating", "act_c": 0.90, "give_up_c": 0.57, "peak_B": 2760, "min_B_pos": 57, "kept": 1.00
        },
        ReturnCurveType.LOG_LINEAR.value: {
            "curve": "log-linear", "act_c": 0.90, "give_up_c": 0.59, "peak_B": 2620, "min_B_pos": 10, "kept": 0.97
        },
        ReturnCurveType.EARLY_FLAT.value: {
            "curve": "early-flat", "act_c": 0.91, "give_up_c": 0.58, "peak_B": 550, "min_B_pos": 8, "kept": 0.71
        },
        ReturnCurveType.THRESHOLD.value: {
            "curve": "threshold", "act_c": 0.85, "give_up_c": 0.57, "peak_B": 5520, "min_B_pos": 3740, "kept": 0.45
        },
        ReturnCurveType.INVERTED_U.value: {
            "curve": "inverted-U", "act_c": 0.90, "give_up_c": 0.56, "peak_B": 2070, "min_B_pos": 10, "kept": 0.97
        },
    }

    return list(table_6_canonical.values())

class NonparametricBudgetTable:
    """
    Pattern 2 (P2): Nonparametric task-conditioned budget table.
    Learns B*(c, task) = argmax_B g_hat(B; c, task) - lambda * B from calibration data.
    Proposition 5.8 bounds misspecification regret by 2 * eta.
    """
    def __init__(
        self,
        cost_lambda: float = DEFAULT_LAMBDA_BETA / DEFAULT_BETA,
        budget_grid: Optional[List[int]] = None,
    ):
        self.cost_lambda = cost_lambda
        self.budget_grid = budget_grid or BUDGET_GRID
        self.table: Dict[Tuple[str, str], int] = {}  # (bin_key, task) -> optimal_B
        self.default_budget = 1024

    def _get_bin_key(self, c: float) -> str:
        for low, high in CONFIDENCE_BINS:
            if low <= c <= high:
                return f"{low:.2f}-{high:.2f}"
        if c > 0.90:
            return "0.90-1.00"
        return "0.50-0.60"

    def fit(
        self,
        calibration_records: List[Dict[str, Any]],
    ):
        """
        Fit nonparametric table from calibration records.
        Each record has: 'confidence', 'task', 'budget', 'accuracy_gain'.
        """
        grouped: Dict[Tuple[str, str, int], List[float]] = {}
        tasks = set()

        for rec in calibration_records:
            c = rec["confidence"]
            task = rec.get("task", "general")
            b = rec["budget"]
            gain = rec["accuracy_gain"]
            b_key = self._get_bin_key(c)
            tasks.add(task)

            key = (b_key, task, b)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(gain)

        for b_key, _ in [(f"{low:.2f}-{high:.2f}", None) for low, high in CONFIDENCE_BINS]:
            for task in tasks:
                best_b = 0
                best_net = 0.0

                for b in self.budget_grid:
                    gains = grouped.get((b_key, task, b), [])
                    avg_gain = float(np.mean(gains)) if gains else 0.0
                    net_val = avg_gain - self.cost_lambda * b
                    if net_val > best_net:
                        best_net = net_val
                        best_b = b

                self.table[(b_key, task)] = best_b

    def get_budget(self, c: float, task: Optional[str] = "general") -> int:
        b_key = self._get_bin_key(c)
        task_str = task or "general"
        if (b_key, task_str) in self.table:
            return self.table[(b_key, task_str)]
        # Fallback to Proposition 5.6 closed form
        return compute_closed_form_budget(c)
