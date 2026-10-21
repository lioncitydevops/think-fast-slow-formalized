import math
import random
from typing import Dict, List, Any
import numpy as np
from ..metacognition.effort_allocation import (
    evaluate_return_curves_table,
    compute_s_confidence,
    compute_closed_form_budget,
    get_decision_band,
)
from ..metacognition.noise_audit import compute_majority_gain_curve
from ..core.types import DecisionBand
from ..core.constants import DEFAULT_R, DEFAULT_GAMMA, DEFAULT_BETA, DEFAULT_LAMBDA_BETA

def replicate_table_6() -> List[Dict[str, Any]]:
    """
    Replicate Table 6 from Zhang & Lee (ASSE 2026):
    Optimal budgets under five return curves (numerical; model properties).
    """
    return evaluate_return_curves_table()

# Backward compatibility alias
replicate_table_3 = replicate_table_6

def replicate_table_9(
    n_samples: int = 50000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Replicate Table 9 from Zhang & Lee (ASSE 2026):
    Worked example: shares, mean thinking tokens per query, and accuracy
    on decided queries under three prices of thought.
    Confidence distribution: c = 0.5 + 0.5 * U, U ~ Beta(3, 1).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Generate synthetic queries matching worked example distribution
    u_vals = np.random.beta(3, 1, size=n_samples)
    confidences = 0.5 + 0.5 * u_vals

    r = 0.1
    gamma = 2.0
    beta = 1024

    policies = [
        {"name": "System 1 only", "type": "s1_only"},
        {"name": "Think on all, B=4096", "type": "think_all", "budget": 4096},
        {"name": "Budgeted, lambda*beta=0.003", "type": "budgeted", "lambda_beta": 0.003},
        {"name": "Budgeted, lambda*beta=0.01", "type": "budgeted", "lambda_beta": 0.01},
        {"name": "Budgeted, lambda*beta=0.03", "type": "budgeted", "lambda_beta": 0.03},
    ]

    table_rows = []

    for pol in policies:
        ptype = pol["type"]
        act_count = 0
        delib_count = 0
        giveup_count = 0
        total_tokens = 0
        decided_correct = 0
        decided_total = 0

        for c in confidences:
            # System 1 is correct with probability c (calibration)
            s1_correct = (random.random() < c)
            d = 2.0 * (1.0 - c)
            ambiguity = 1.0 - (d ** gamma)

            if ptype == "s1_only":
                act_count += 1
                decided_total += 1
                if s1_correct:
                    decided_correct += 1

            elif ptype == "think_all":
                B = pol["budget"]
                delib_count += 1
                total_tokens += B
                decided_total += 1
                h_B = 1.0 - math.exp(-B / beta)
                # Overturns error with prob (1 - d^gamma)*h(B), overturns correct with prob r*h(B)
                if not s1_correct:
                    is_correct = (random.random() < ambiguity * h_B)
                else:
                    is_correct = not (random.random() < r * h_B)
                if is_correct:
                    decided_correct += 1

            elif ptype == "budgeted":
                lb = pol["lambda_beta"]
                b_star = compute_closed_form_budget(c, r=r, gamma=gamma, lambda_beta=lb, beta=beta)
                s = compute_s_confidence(c, r=r, gamma=gamma)

                # Law of least effort act band
                tau_act = (1.0 - lb) / (1.0 + r)

                if c >= tau_act or (b_star == 0 and c > 0.70):
                    act_count += 1
                    decided_total += 1
                    if s1_correct:
                        decided_correct += 1
                elif b_star == 0 and c <= 0.60:
                    # Give up band
                    giveup_count += 1
                else:
                    delib_count += 1
                    total_tokens += b_star
                    decided_total += 1
                    h_B = 1.0 - math.exp(-b_star / beta)
                    if not s1_correct:
                        is_correct = (random.random() < ambiguity * h_B)
                    else:
                        is_correct = not (random.random() < r * h_B)
                    if is_correct:
                        decided_correct += 1

        act_share = act_count / n_samples
        delib_share = delib_count / n_samples
        giveup_share = giveup_count / n_samples
        mean_tokens = total_tokens / n_samples
        accuracy = (decided_correct / decided_total) if decided_total > 0 else 0.0

        table_rows.append({
            "policy": pol["name"],
            "act_share": f"{act_share * 100:.1f}%" if act_share > 0 else "-",
            "deliberate_share": f"{delib_share * 100:.1f}%" if delib_share > 0 else "-",
            "give_up_share": f"{giveup_share * 100:.1f}%" if giveup_share > 0 else "-",
            "tokens": int(round(mean_tokens)),
            "accuracy": round(accuracy, 3),
            "raw": {
                "act_pct": act_share * 100,
                "delib_pct": delib_share * 100,
                "giveup_pct": giveup_share * 100,
                "tokens": mean_tokens,
                "accuracy": accuracy,
            }
        })

    return table_rows

# Backward compatibility alias
replicate_table_5 = replicate_table_9

def generate_figure_1_data() -> Dict[str, Any]:
    """
    Figure 1 data: Optimal thinking budget B*(c) under five return curves (log scale).
    """
    c_grid = [round(x, 3) for x in np.linspace(0.50, 0.98, 60)]
    from ..core.types import ReturnCurveType
    from ..metacognition.effort_allocation import get_return_curve, optimize_budget_for_curve

    curves = [
        ReturnCurveType.SATURATING,
        ReturnCurveType.LOG_LINEAR,
        ReturnCurveType.EARLY_FLAT,
        ReturnCurveType.THRESHOLD,
        ReturnCurveType.INVERTED_U,
    ]

    series = {}
    for ctype in curves:
        h = get_return_curve(ctype)
        b_vals = []
        for c in c_grid:
            opt_b, _ = optimize_budget_for_curve(c, h)
            # Cap minimum positive value at 1 for clean log scale plotting
            b_vals.append(max(1, opt_b))
        series[ctype.value] = b_vals

    return {
        "confidence_grid": c_grid,
        "series": series,
    }

def generate_figure_2_data() -> Dict[str, Any]:
    """
    Figure 2 data: Gain from majority voting over k System-2 runs.
    """
    return compute_majority_gain_curve()

def generate_figure_4_data() -> Dict[str, Any]:
    """
    Figure 4 data: Price of thought frontier.
    Shows tokens per query, act share, and delib share as lambda*beta varies from 1e-3 to 2e-1.
    """
    lb_grid = np.logspace(-3, -0.6, 25)
    points = []

    for lb in lb_grid:
        u_vals = np.random.beta(3, 1, size=5000)
        confs = 0.5 + 0.5 * u_vals
        tokens = []
        acts = 0
        delibs = 0

        for c in confs:
            b = compute_closed_form_budget(c, lambda_beta=lb)
            tau_act = (1.0 - lb) / (1.0 + DEFAULT_R)
            if c >= tau_act or (b == 0 and c > 0.70):
                acts += 1
                tokens.append(0)
            elif b == 0 and c <= 0.60:
                tokens.append(0)
            else:
                delibs += 1
                tokens.append(b)

        points.append({
            "lambda_beta": float(lb),
            "mean_tokens": float(np.mean(tokens)),
            "act_pct": float(acts / 5000 * 100),
            "delib_pct": float(delibs / 5000 * 100),
        })

    return {"frontier": points}
