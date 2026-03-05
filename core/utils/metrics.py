"""Additional evaluation metrics for MBRDL experiments."""

import numpy as np
from sklearn.metrics import accuracy_score


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error.

    Args:
        y_true: True labels [N]
        y_prob: Predicted probabilities [N, num_classes]
        n_bins: Number of bins for calibration (default: 10)

    Returns:
        ECE score (float)
    """
    # Get predicted class and confidence
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins[1:-1])

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_size = mask.sum() / len(y_true)
            ece += bin_size * abs(bin_acc - bin_conf)

    return ece


def compute_mean_reciprocal_rank(algorithm_results):
    """Mean Reciprocal Rank for algorithm comparison.

    Args:
        algorithm_results: Dict mapping algorithm name to accuracy

    Returns:
        Dict mapping algorithm name to MRR score

    Example:
        >>> results = {'erm': 0.70, 'mda': 0.75, 'mrat': 0.82}
        >>> compute_mean_reciprocal_rank(results)
        {'mrat': 1.0, 'mda': 0.5, 'erm': 0.333}
    """
    # Sort by accuracy (descending)
    sorted_algs = sorted(algorithm_results.items(), key=lambda x: x[1], reverse=True)

    # Compute reciprocal ranks
    mrr_scores = {}
    for rank, (alg_name, _) in enumerate(sorted_algs, start=1):
        mrr_scores[alg_name] = 1.0 / rank

    return mrr_scores


def compute_all_metrics(y_true, y_prob):
    """Compute all evaluation metrics.

    Args:
        y_true: True labels [N]
        y_prob: Predicted probabilities [N, num_classes]

    Returns:
        Dictionary with all metrics
    """
    y_pred = np.argmax(y_prob, axis=1)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'ece': compute_ece(y_true, y_prob),
    }
