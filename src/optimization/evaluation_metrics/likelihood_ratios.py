from .core_metrics import calculate_true_positive_rate, calculate_false_positive_rate
from .core_metrics import calculate_true_negative_rate, calculate_false_negative_rate

def calculate_positive_likelihood_ratio(inferred_probabilities: list, truth_labels: list):
    """
    Determines the TPR/FPR ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tpr = calculate_true_positive_rate(inferred_probabilities, truth_labels)
    fpr = calculate_false_positive_rate(inferred_probabilities, truth_labels)

    if fpr == 0:
        return 0

    return tpr/fpr

def calculate_negative_likelihood_ratio(inferred_probabilities: list, truth_labels: list):
    """
    Determines the FNR/TNR. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    fnr = calculate_false_negative_rate(inferred_probabilities, truth_labels)
    tnr = calculate_true_negative_rate(inferred_probabilities, truth_labels)

    if tnr == 0:
        return 0

    return fnr/tnr
