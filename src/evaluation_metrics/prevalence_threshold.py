from evaluation_metrics.core_metrics import calculate_true_positive_rate, calculate_false_positive_rate
from math import sqrt

def calculate_prevalence_threshold(inferred_probabilities: list, truth_labels: list):
    """
    Determines the (sqrt(TPRxFPR)-FPR)/(TPR-FPR) value. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tpr = calculate_true_positive_rate(inferred_probabilities, truth_labels)
    fpr = calculate_false_positive_rate(inferred_probabilities, truth_labels)

    top = sqrt(tpr*fpr) - fpr
    bottom = tpr - fpr

    if bottom == 0:
        return 0

    return top/bottom
