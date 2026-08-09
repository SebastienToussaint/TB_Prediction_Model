from core_metrics import calculate_true_positive_rate, calculate_false_positive_rate
from core_metrics import calculate_true_negative_rate, calculate_false_negative_rate

def calculate_positive_likelihood_ratio(inferred_labels: list, truth_labels: list):
    """
    Determines the TPR/FPR ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    tpr = calculate_true_positive_rate(inferred_labels, truth_labels)
    fpr = calculate_false_positive_rate(inferred_labels, truth_labels)

    return tpr/fpr

def calculate_negative_likelihood_ratio(inferred_labels: list, truth_labels: list):
    """
    Determines the FNR/TNR. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    fnr = calculate_false_negative_rate(inferred_labels, truth_labels)
    tnr = calculate_true_negative_rate(inferred_labels, truth_labels)

    return fnr/tnr
