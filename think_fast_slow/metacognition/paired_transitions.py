from typing import List, Dict, Any, Optional
from ..core.types import TransitionType

def classify_transition(
    a1: str,
    b: str,
    i: str,
    K: int = 2,
) -> TransitionType:
    """
    Classify transition between blind (b) and informed (i) deliberation
    given System 1 answer (a1).
    """
    if i == b:
        return TransitionType.STABLE
    elif b != a1 and i == a1:
        return TransitionType.ANCHOR
    elif b == a1 and i != a1:
        return TransitionType.REACTANCE
    else:
        return TransitionType.DRIFT

def compute_paired_transition_decomposition(
    a1_list: List[str],
    blind_list: List[str],
    informed_list: List[str],
    gold_list: List[str],
) -> Dict[str, Any]:
    """
    Proposition 5.4 (Paired Transition Decomposition).
    Partitions escalated queries into Stable, Anchor, Reactance, and Drift.
    Verifies that:
        G_inf - G_blind = sum_T P(T) * [ P(i=y | T) - P(b=y | T) ]
    with the Stable term vanishing identically.
    """
    n = len(gold_list)
    if n == 0:
        return {"diff": 0.0, "identity_holds": True}

    # Identify transition types
    types = [
        classify_transition(a1, b, i)
        for a1, b, i in zip(a1_list, blind_list, informed_list)
    ]

    # Overall gains relative to a1
    blind_correct = sum(1 for b, y in zip(blind_list, gold_list) if b == y)
    inf_correct = sum(1 for i, y in zip(informed_list, gold_list) if i == y)
    a1_correct = sum(1 for a1, y in zip(a1_list, gold_list) if a1 == y)

    G_blind = (blind_correct - a1_correct) / n
    G_inf = (inf_correct - a1_correct) / n
    delta_G_direct = G_inf - G_blind

    # Decomposition by transition type T
    type_stats = {}
    sum_decomposed = 0.0

    for t_type in TransitionType:
        indices = [idx for idx, t in enumerate(types) if t == t_type]
        count = len(indices)
        p_T = count / n

        if count > 0:
            p_inf_correct = sum(1 for idx in indices if informed_list[idx] == gold_list[idx]) / count
            p_blind_correct = sum(1 for idx in indices if blind_list[idx] == gold_list[idx]) / count
        else:
            p_inf_correct = 0.0
            p_blind_correct = 0.0

        diff_T = p_inf_correct - p_blind_correct
        term_contribution = p_T * diff_T
        sum_decomposed += term_contribution

        type_stats[t_type.value] = {
            "count": count,
            "share_P_T": float(p_T),
            "p_inf_correct": float(p_inf_correct),
            "p_blind_correct": float(p_blind_correct),
            "diff": float(diff_T),
            "contribution": float(term_contribution),
        }

    # Verify stable term is exactly 0
    stable_contribution = type_stats[TransitionType.STABLE.value]["contribution"]
    identity_diff = abs(delta_G_direct - sum_decomposed)

    # Corollary 5.5: Pure anchoring check
    anchor_share = type_stats[TransitionType.ANCHOR.value]["share_P_T"]
    reactance_share = type_stats[TransitionType.REACTANCE.value]["share_P_T"]
    drift_share = type_stats[TransitionType.DRIFT.value]["share_P_T"]

    is_pure_anchoring = (reactance_share == 0 and drift_share == 0)
    alpha = anchor_share
    corollary_5_5_holds = abs(G_inf - (1.0 - alpha) * G_blind) < 1e-6 if is_pure_anchoring else None

    return {
        "G_blind": float(G_blind),
        "G_inf": float(G_inf),
        "delta_G_direct": float(delta_G_direct),
        "sum_decomposed": float(sum_decomposed),
        "diff_machine_precision": float(identity_diff),
        "identity_holds": bool(identity_diff < 1e-12),
        "stable_contribution_zero": bool(abs(stable_contribution) < 1e-12),
        "shares": {
            "stable": float(type_stats[TransitionType.STABLE.value]["share_P_T"]),
            "anchor": float(anchor_share),
            "reactance": float(reactance_share),
            "drift": float(drift_share),
        },
        "breakdown": type_stats,
        "is_pure_anchoring": is_pure_anchoring,
        "corollary_5_5_holds": corollary_5_5_holds,
    }
