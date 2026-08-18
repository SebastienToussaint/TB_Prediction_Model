from .metric_counts import get_true_positives, get_false_positives, get_false_negatives

def calculate_f1_score(inferred_labels: list, truth_labels: list):
    """
    Determines the 2TP/(2TP+FP+FN) ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    tp = get_true_positives(inferred_labels, truth_labels)[0]
    fp = get_false_positives(inferred_labels, truth_labels)[0]
    fn = get_false_negatives(inferred_labels, truth_labels)[0]

    top = 2*tp
    bottom = 2*tp + fp + fn

    return top/bottom
