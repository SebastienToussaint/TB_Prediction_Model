from .metric_counts import get_true_positives, get_false_positives, get_false_negatives

def calculate_f1_score(inferred_probabilities: list, truth_labels: list):
    """
    Determines the 2TP/(2TP+FP+FN) ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tp = get_true_positives(inferred_probabilities, truth_labels)[0]
    fp = get_false_positives(inferred_probabilities, truth_labels)[0]
    fn = get_false_negatives(inferred_probabilities, truth_labels)[0]

    top = 2*tp
    bottom = 2*tp + fp + fn

    return top/bottom
