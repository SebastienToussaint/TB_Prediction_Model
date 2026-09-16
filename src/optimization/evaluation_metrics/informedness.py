from .core_metrics import calculate_true_positive_rate, calculate_true_negative_rate

def calculate_informedness(inferred_probabilities: list, truth_labels: list):
    """
    Determines the TPR + TNR - 1 value. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tpr = calculate_true_positive_rate(inferred_probabilities, truth_labels)
    tnr = calculate_true_negative_rate(inferred_probabilities, truth_labels)

    return tpr + tnr - 1