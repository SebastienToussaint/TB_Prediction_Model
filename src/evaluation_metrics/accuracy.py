from evaluation_metrics.metric_counts import get_true_positives, get_true_negatives
from evaluation_metrics.core_metrics import calculate_true_positive_rate, calculate_true_negative_rate

def calculate_accuracy(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the TP+TN/P+N ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tp = get_true_positives(inferred_probabilities, truth_labels, cutoff)[0]
    tn = get_true_negatives(inferred_probabilities, truth_labels, cutoff)[0]

    return (tp+tn)/(len(inferred_probabilities))


def calculate_balanced_accuracy(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the TPR+TNR/2 ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tpr = calculate_true_positive_rate(inferred_probabilities, truth_labels, cutoff)
    tnr = calculate_true_negative_rate(inferred_probabilities, truth_labels, cutoff)

    return (tpr+tnr)/2
