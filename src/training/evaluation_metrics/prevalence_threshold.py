from .core_metrics import calculate_true_positive_rate, calculate_false_positive_rate
from math import sqrt

def calculate_informedness(inferred_labels: list, truth_labels: list):
    """
    Determines the (sqrt(TPRxFPR)-FPR)/(TPR-FPR) value. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    tpr = calculate_true_positive_rate(inferred_labels, truth_labels)
    fpr = calculate_false_positive_rate(inferred_labels, truth_labels)

    top = sqrt(tpr*fpr) - fpr
    bottom = tpr - fpr

    return top/bottom
