import math
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import binom
import numpy as np

def majority_probability(p: float, k: int = 5) -> float:
    """
    Proposition 5.11: Probability that majority vote over k independent runs is correct.
    M_k(p) = P(Bin(k, p) > k/2) = sum_{j = (k+1)/2}^k C(k, j) * p^j * (1-p)^(k-j).
    """
    if k % 2 == 0:
        raise ValueError(f"k must be odd, got {k}")
    threshold = (k // 2) + 1
    # 1 - CDF(threshold - 1)
    return float(1.0 - binom.cdf(threshold - 1, k, p))

def compute_majority_gain_curve(
    k_values: Optional[List[int]] = None,
    p_grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute M_k(p) - p for Figure 2 in Zhang & Lee (2026).
    Demonstrates that gain > 0 for p > 0.5 (noise reduction),
    and gain < 0 for p < 0.5 (bias entrenchment).
    """
    if k_values is None:
        k_values = [3, 5, 9, 25]
    if p_grid is None:
        p_grid = [round(x, 3) for x in np.linspace(0.0, 1.0, 101)]

    curve_data = {}
    for k in k_values:
        gains = []
        for p in p_grid:
            m_k = majority_probability(p, k=k)
            gain = m_k - p
            gains.append(gain)
        curve_data[f"k={k}"] = {
            "p_values": p_grid,
            "gains": gains,
            "max_gain": max(gains),
            "min_gain": min(gains),
        }

    return curve_data

def run_empirical_noise_audit(
    repeated_runs: List[List[str]],
    gold_answers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Occasion Noise Audit (Section 5.6 & 8).
    Takes repeated answers from System 2 across k runs for each query.
    Computes:
    - Pairwise disagreement rate (occasion noise)
    - Majority vote answer per query
    - Empirical voting gain if gold labels are provided
    """
    n_queries = len(repeated_runs)
    if n_queries == 0:
        return {"pairwise_disagreement": 0.0, "majority_gain": 0.0}

    total_pairs = 0
    disagreeing_pairs = 0
    majority_answers = []

    for runs in repeated_runs:
        k = len(runs)
        for i in range(k):
            for j in range(i + 1, k):
                total_pairs += 1
                if runs[i] != runs[j]:
                    disagreeing_pairs += 1

        # Majority vote
        counts: Dict[str, int] = {}
        for ans in runs:
            counts[ans] = counts.get(ans, 0) + 1
        maj_ans = max(counts.keys(), key=lambda x: counts[x])
        majority_answers.append(maj_ans)

    disagreement_rate = disagreeing_pairs / total_pairs if total_pairs > 0 else 0.0

    eval_results = {
        "disagreement_rate": float(disagreement_rate),
        "total_queries": n_queries,
        "runs_per_query": len(repeated_runs[0]) if repeated_runs else 0,
        "majority_answers": majority_answers,
    }

    if gold_answers is not None:
        single_run_accuracies = []
        maj_correct = 0
        est_p_hats = []

        for idx, (runs, gold) in enumerate(zip(repeated_runs, gold_answers)):
            k = len(runs)
            if k >= 10:
                # Section 8.2 & Table 11: 5 runs vote, and 5 more estimate accuracy independently
                voting_runs = runs[:5]
                estimation_runs = runs[5:10]
            else:
                voting_runs = runs
                estimation_runs = runs

            # Voting runs accuracy and majority
            counts_v: Dict[str, int] = {}
            for ans in voting_runs:
                counts_v[ans] = counts_v.get(ans, 0) + 1
            maj_v = max(counts_v.keys(), key=lambda x: counts_v[x])

            acc_single = sum(1 for a in voting_runs if a == gold) / len(voting_runs)
            single_run_accuracies.append(acc_single)
            if maj_v == gold:
                maj_correct += 1

            # Independent estimation of p_hat_i
            p_hat = sum(1 for a in estimation_runs if a == gold) / len(estimation_runs)
            est_p_hats.append(p_hat)

        overall_single_acc = float(np.mean(single_run_accuracies))
        overall_maj_acc = float(maj_correct / n_queries)
        emp_gain = overall_maj_acc - overall_single_acc

        # Separate items with p_hat_i > 0.5 vs p_hat_i < 0.5 using independent estimation
        high_p_items = [i for i, p in enumerate(est_p_hats) if p > 0.5]
        low_p_items = [i for i, p in enumerate(est_p_hats) if p < 0.5]

        gain_high_p = 0.0
        if high_p_items:
            acc_single_h = np.mean([single_run_accuracies[i] for i in high_p_items])
            acc_maj_h = np.mean([1.0 if majority_answers[i] == gold_answers[i] else 0.0 for i in high_p_items])
            gain_high_p = float(acc_maj_h - acc_single_h)

        gain_low_p = 0.0
        if low_p_items:
            acc_single_l = np.mean([single_run_accuracies[i] for i in low_p_items])
            acc_maj_l = np.mean([1.0 if majority_answers[i] == gold_answers[i] else 0.0 for i in low_p_items])
            gain_low_p = float(acc_maj_l - acc_single_l)

        eval_results.update({
            "single_run_accuracy": overall_single_acc,
            "majority_accuracy": overall_maj_acc,
            "overall_voting_gain": float(emp_gain),
            "gain_on_noisy_items_p_gt_half": gain_high_p,
            "gain_on_biased_items_p_lt_half": gain_low_p,
            "hypothesis_H9_holds": bool(gain_high_p >= 0 and gain_low_p <= 0),
        })

    return eval_results
