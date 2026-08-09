from core_metrics import calculate_positive_predictive_value, calculate_true_positive_rate
from math import sqrt

def calculate_fowlkes_mallows_index(inferred_labels: list, truth_labels: list):
    """
    Determines sqrt(PPVxTPR). Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    ppv = calculate_positive_predictive_value(inferred_labels, truth_labels)
    tpr = calculate_true_positive_rate(inferred_labels, truth_labels)

    return sqrt(ppv*tpr)
