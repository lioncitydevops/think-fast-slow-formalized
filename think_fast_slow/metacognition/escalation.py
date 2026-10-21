from typing import List, Dict, Any, Optional
import numpy as np

def compute_escalation_identity(
    a1_list: List[str],
    a2_list: List[str],
    gold_list: Optional[List[str]] = None,
    in_Z: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """
    Proposition 5.1 (Escalation Identity) & Corollary 5.2 (Where vs Whether).
    Accuracy gained on Z by releasing a2 instead of a1 is:
        G(Z) = M_Z * (W2 - W1)
    where:
        M_Z = P(x in Z, a1 != a2) [observable on unlabelled traffic]
        W_k = P(a_k = y | x in Z, a1 != a2) [requires labels]
    """
    n = len(a1_list)
    if n == 0:
        return {"M_Z": 0.0, "W1": 0.0, "W2": 0.0, "G_identity": 0.0, "G_direct": 0.0, "diff": 0.0}

    if in_Z is None:
        in_Z = [True] * n

    # Count escalated items
    z_indices = [i for i in range(n) if in_Z[i]]
    n_Z = len(z_indices)
    if n_Z == 0:
        return {"M_Z": 0.0, "W1": 0.0, "W2": 0.0, "G_identity": 0.0, "G_direct": 0.0, "diff": 0.0}

    # Disagreement indicator within Z
    disagreements = [i for i in z_indices if a1_list[i] != a2_list[i]]
    n_disagree = len(disagreements)

    # M_Z: probability of being in Z AND disagreeing
    M_Z = n_disagree / n

    if gold_list is None:
        # Corollary 5.2: Label-free computation of M_Z
        return {
            "M_Z": float(M_Z),
            "disagreement_count": n_disagree,
            "escalated_count": n_Z,
            "total_count": n,
            "W1": None,
            "W2": None,
            "win_margin": None,
            "G_identity": None,
        }

    # With labels: compute W1, W2 and check identity
    if n_disagree == 0:
        W1 = 0.0
        W2 = 0.0
    else:
        w1_correct = sum(1 for i in disagreements if a1_list[i] == gold_list[i])
        w2_correct = sum(1 for i in disagreements if a2_list[i] == gold_list[i])
        W1 = w1_correct / n_disagree
        W2 = w2_correct / n_disagree

    win_margin = W2 - W1
    G_identity = M_Z * win_margin

    # Direct computation: E[ 1_Z * ( 1{a2=y} - 1{a1=y} ) ]
    direct_gain = sum((1 if a2_list[i] == gold_list[i] else 0) - (1 if a1_list[i] == gold_list[i] else 0) for i in z_indices) / n

    diff = abs(G_identity - direct_gain)

    return {
        "M_Z": float(M_Z),
        "W1": float(W1),
        "W2": float(W2),
        "win_margin": float(win_margin),
        "G_identity": float(G_identity),
        "G_direct": float(direct_gain),
        "diff_machine_precision": float(diff),
        "identity_holds": bool(diff < 1e-12),
        "disagreement_count": n_disagree,
        "escalated_count": n_Z,
        "total_count": n,
    }
