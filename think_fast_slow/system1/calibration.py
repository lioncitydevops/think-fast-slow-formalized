import numpy as np
from typing import List, Dict, Tuple, Any

def compute_calibration_curve(
    confidences: List[float],
    accuracies: List[bool],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute reliability diagram bins, Expected Calibration Error (ECE),
    and Maximum Calibration Error (MCE).
    """
    conf = np.array(confidences)
    acc = np.array(accuracies, dtype=float)
    n = len(conf)

    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    bin_edges = np.linspace(0.5, 1.0, num_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            idx = np.where((conf >= low) & (conf <= high))[0]
        else:
            idx = np.where((conf >= low) & (conf < high))[0]

        count = len(idx)
        if count > 0:
            avg_conf = float(np.mean(conf[idx]))
            avg_acc = float(np.mean(acc[idx]))
            gap = abs(avg_acc - avg_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
            bin_data.append({
                "bin_low": float(low),
                "bin_high": float(high),
                "count": int(count),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "gap": gap,
            })
        else:
            bin_data.append({
                "bin_low": float(low),
                "bin_high": float(high),
                "count": 0,
                "avg_confidence": float((low + high) / 2.0),
                "avg_accuracy": float((low + high) / 2.0),
                "gap": 0.0,
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins": bin_data,
        "sample_size": n,
    }

def compute_brier_score(
    prob_vectors: List[Dict[str, float]],
    gold_labels: List[str],
) -> float:
    """
    Compute multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2.
    """
    n = len(gold_labels)
    if n == 0:
        return 0.0

    total_loss = 0.0
    for probs, gold in zip(prob_vectors, gold_labels):
        for opt, p in probs.items():
            y_k = 1.0 if opt == gold else 0.0
            total_loss += (p - y_k) ** 2

    return float(total_loss / n)

def compute_classwise_calibration(
    prob_vectors: List[Dict[str, float]],
    gold_labels: List[str],
    num_bins: int = 5,
) -> Dict[str, float]:
    """
    Compute classwise ECE across individual classes (required for Prop 5.9).
    """
    all_classes = sorted(list(set(gold_labels)))
    class_eces = {}

    for c in all_classes:
        confs = [probs.get(c, 0.0) for probs in prob_vectors]
        accs = [gold == c for gold in gold_labels]
        curve = compute_calibration_curve(confs, accs, num_bins=num_bins)
        class_eces[c] = curve["ece"]

    return class_eces
