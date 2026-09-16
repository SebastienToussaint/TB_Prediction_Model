from evaluation_metrics.core_metrics import calculate_positive_predictive_value, calculate_true_positive_rate
from math import sqrt

def calculate_fowlkes_mallows_index(inferred_probabilities: list, truth_labels: list):
    """
    Determines sqrt(PPVxTPR). Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    ppv = calculate_positive_predictive_value(inferred_probabilities, truth_labels)
    tpr = calculate_true_positive_rate(inferred_probabilities, truth_labels)

    return sqrt(ppv*tpr)
