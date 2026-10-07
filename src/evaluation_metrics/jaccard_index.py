from evaluation_metrics.metric_counts import get_true_positives, get_false_negatives, get_false_positives

def calculate_jaccard_index(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the TP/TP+FN+FP ratio (same as threat score). Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tp = get_true_positives(inferred_probabilities, truth_labels, cutoff)[0]
    fn = get_false_negatives(inferred_probabilities, truth_labels, cutoff)[0]
    fp = get_false_positives(inferred_probabilities, truth_labels, cutoff)[0]

    if (tp+fn+fp) == 0:
        return 0

    return tp/(tp+fn+fp)
