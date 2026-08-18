from .metric_counts import get_true_positives, get_true_negatives
from .core_metrics import calculate_true_positive_rate, calculate_true_negative_rate

def calculate_accuracy(inferred_labels: list, truth_labels: list):
    """
    Determines the TP+TN/P+N ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    tp = get_true_positives(inferred_labels, truth_labels)[0]
    tn = get_true_negatives(inferred_labels, truth_labels)[0]

    return (tp+tn)/(len(inferred_labels))


def calculate_balanced_accuracy(inferred_labels: list, truth_labels: list):
    """
    Determines the TPR+TNR/2 ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    tpr = calculate_true_positive_rate(inferred_labels, truth_labels)
    tnr = calculate_true_negative_rate(inferred_labels, truth_labels)

    return (tpr+tnr)/2
