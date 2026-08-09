from math import sqrt
from core_metrics import *

def calculate_matthews_correlation_coefficient(inferred_labels: list, truth_labels: list):
    """
    Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    tpr = calculate_true_positive_rate(inferred_labels, truth_labels)
    tnr = calculate_true_negative_rate(inferred_labels, truth_labels)
    ppv = calculate_positive_predictive_value(inferred_labels, truth_labels)
    npv = calculate_negative_predictive_value(inferred_labels, truth_labels)
    fnr = calculate_false_negative_rate(inferred_labels, truth_labels)
    fpr = calculate_false_positive_rate(inferred_labels, truth_labels)
    _for = calculate_false_omission_rate(inferred_labels, truth_labels)
    fdr = calculate_false_discovery_rate(inferred_labels, truth_labels)

    return sqrt(tpr*tnr*ppv*npv) - sqrt(fnr*fpr*_for*fdr)

