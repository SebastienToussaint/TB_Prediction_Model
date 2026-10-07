from math import sqrt
from evaluation_metrics.core_metrics import *

def calculate_matthews_correlation_coefficient(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tpr = calculate_true_positive_rate(inferred_probabilities, truth_labels, cutoff)
    tnr = calculate_true_negative_rate(inferred_probabilities, truth_labels, cutoff)
    ppv = calculate_positive_predictive_value(inferred_probabilities, truth_labels, cutoff)
    npv = calculate_negative_predictive_value(inferred_probabilities, truth_labels, cutoff)
    fnr = calculate_false_negative_rate(inferred_probabilities, truth_labels, cutoff)
    fpr = calculate_false_positive_rate(inferred_probabilities, truth_labels, cutoff)
    _for = calculate_false_omission_rate(inferred_probabilities, truth_labels, cutoff)
    fdr = calculate_false_discovery_rate(inferred_probabilities, truth_labels, cutoff)

    return sqrt(tpr*tnr*ppv*npv) - sqrt(fnr*fpr*_for*fdr)

