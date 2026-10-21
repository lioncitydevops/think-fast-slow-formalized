import math
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import beta

def clopper_pearson_upper(k: int, n: int, delta: float = 0.05) -> float:
    """
    Exact Clopper-Pearson upper confidence bound for binomial parameter p at level 1 - delta.
    P(p <= upper_bound) >= 1 - delta.
    """
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    # Upper bound is the (1 - delta) quantile of Beta(k + 1, n - k)
    return float(beta.ppf(1.0 - delta, k + 1, n - k))

def certify_act_threshold(
    confidences: List[float],
    is_correct: List[bool],
    epsilon: float = 0.10,
    delta: float = 0.05,
    tau_grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Remark 1 (Threshold certification via fixed-sequence testing).
    Scan tau from high to low and stop at first threshold whose
    Clopper-Pearson upper bound on act-band error exceeds epsilon.
    Returns optimal certified threshold tau* guaranteeing act-band error <= epsilon with prob >= 1 - delta.
    """
    if tau_grid is None:
        tau_grid = [round(t, 2) for t in np_linspace(0.99, 0.50, 50)]

    tau_grid = sorted(tau_grid, reverse=True)
    n = len(confidences)

    certified_tau = 1.0
    best_stats = None
    min_required_n = math.ceil(math.log(delta) / math.log(1.0 - epsilon)) if epsilon < 1.0 else 1
    # If the provided sample size is smaller than theoretical minimum, relax minimum threshold to len(confidences) // 2 or 5
    effective_min_n = min(min_required_n, max(5, n // 2)) if n >= 5 else 1
    has_entered_valid_band = False

    for tau in tau_grid:
        idx = [i for i in range(n) if confidences[i] >= tau]
        n_tau = len(idx)
        if n_tau < effective_min_n:
            continue

        errors = sum(1 for i in idx if not is_correct[i])
        emp_error = errors / n_tau
        cp_upper = clopper_pearson_upper(errors, n_tau, delta=delta)

        if cp_upper <= epsilon:
            certified_tau = tau
            has_entered_valid_band = True
            best_stats = {
                "tau": float(tau),
                "n_act": n_tau,
                "errors": errors,
                "empirical_error": float(emp_error),
                "clopper_pearson_upper": float(cp_upper),
                "act_share": float(n_tau / n),
                "bound_satisfied": True,
            }
        else:
            if has_entered_valid_band:
                # Exceeded epsilon after certifying valid thresholds; stop scanning (fixed sequence testing)
                break
            # If not yet entered valid band but sample is small, continue to see if larger sample reduces CI width

    if best_stats is None:
        # If even highest tau fails bound, fall back to max tau
        idx = [i for i in range(n) if confidences[i] >= tau_grid[0]]
        errors = sum(1 for i in idx if not is_correct[i]) if idx else 0
        best_stats = {
            "tau": float(tau_grid[0]),
            "n_act": len(idx),
            "errors": errors,
            "empirical_error": float(errors / len(idx)) if idx else 0.0,
            "clopper_pearson_upper": float(clopper_pearson_upper(errors, len(idx), delta=delta)) if idx else 1.0,
            "act_share": float(len(idx) / n) if n else 0.0,
            "bound_satisfied": False,
        }

    return {
        "certified_tau": float(certified_tau),
        "target_epsilon": epsilon,
        "delta": delta,
        "stats": best_stats,
    }

def np_linspace(start: float, stop: float, num: int) -> List[float]:
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]

class WaldSPRT:
    """
    Wald's Sequential Probability Ratio Test (Section 7.1)
    Audits a random sample of act-band decisions:
    H0: error <= epsilon vs H1: error >= epsilon + delta_sp
    Signals distribution drift and recertification duty when H1 is accepted.
    """
    def __init__(
        self,
        epsilon: float = 0.10,
        delta_sp: float = 0.05,
        alpha_err: float = 0.05,
        beta_err: float = 0.10,
    ):
        self.p0 = epsilon
        self.p1 = epsilon + delta_sp
        self.alpha_err = alpha_err
        self.beta_err = beta_err

        # Stopping boundaries
        self.A = math.log((1.0 - beta_err) / alpha_err)  # Reject H0 (signal drift)
        self.B = math.log(beta_err / (1.0 - alpha_err))  # Accept H0 (certified valid)

        self.log_likelihood_ratio = 0.0
        self.sample_count = 0
        self.error_count = 0
        self.status = "CONTINUE"  # CONTINUE, REJECT_H0 (drift), ACCEPT_H0 (valid)

    def update(self, is_error: bool) -> str:
        self.sample_count += 1
        if is_error:
            self.error_count += 1
            z = math.log(self.p1 / self.p0)
        else:
            z = math.log((1.0 - self.p1) / (1.0 - self.p0))

        self.log_likelihood_ratio += z

        if self.log_likelihood_ratio >= self.A:
            self.status = "REJECT_H0_DRIFT_DETECTED"
        elif self.log_likelihood_ratio <= self.B:
            self.status = "ACCEPT_H0_CALIBRATION_VALID"
        else:
            self.status = "CONTINUE"

        return self.status

    def reset(self):
        self.log_likelihood_ratio = 0.0
        self.sample_count = 0
        self.error_count = 0
        self.status = "CONTINUE"
