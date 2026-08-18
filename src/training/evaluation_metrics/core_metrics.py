from .metric_counts import get_true_positives, get_false_negatives, get_true_negatives, get_false_positives

def calculate_true_positive_rate(inferred_labels: list, truth_labels: list):
    """
    Determines the TP/TP+FN ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    true_positives = get_true_positives(inferred_labels, truth_labels)[0]
    false_negatives = get_false_negatives(inferred_labels, truth_labels)[0]

    return true_positives/(true_positives+false_negatives)


def calculate_recall(inferred_labels: list, truth_labels: list): # For ease of use
    return calculate_true_positive_rate(inferred_labels, truth_labels)


def calculate_false_positive_rate(inferred_labels: list, truth_labels: list):
    """
    Determines the FP/FP+TN ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    return 1-calculate_true_negative_rate(inferred_labels, truth_labels)


def calculate_true_negative_rate(inferred_labels: list, truth_labels: list):
    """
    Determines the TN/TN+FP ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    true_negatives = get_true_negatives(inferred_labels, truth_labels)[0]
    false_positives = get_false_positives(inferred_labels, truth_labels)[0]

    return true_negatives/(true_negatives+false_positives)


def calculate_specificity(inferred_labels: list, truth_labels: list): # For ease of use
    return 1-calculate_false_positive_rate(inferred_labels, truth_labels)


def calculate_false_negative_rate(inferred_labels: list, truth_labels: list):
    """
    Determines the FN/FN+TP ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    return 1-calculate_true_positive_rate(inferred_labels, truth_labels)


def calculate_positive_predictive_value(inferred_labels: list, truth_labels: list):
    """
    Determines the TP/TP+FP ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    true_positives = get_true_positives(inferred_labels, truth_labels)[0]
    false_positives = get_false_positives(inferred_labels, truth_labels)[0]

    return true_positives/(true_positives+false_positives)


def calculate_precision(inferred_labels: list, truth_labels: list): # For ease of use
    return calculate_positive_predictive_value(inferred_labels, truth_labels)


def calculate_negative_predictive_value(inferred_labels: list, truth_labels: list):
    """
    Determines the TN/TN+FN ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    true_negatives = get_true_negatives(inferred_labels, truth_labels)[0]
    false_negatives = get_false_negatives(inferred_labels, truth_labels)[0]

    return true_negatives/(true_negatives+false_negatives)


def calculate_false_omission_rate(inferred_labels: list, truth_labels: list):
    """
    Determines the FN/TN+FN ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    return 1-calculate_negative_predictive_value(inferred_labels, truth_labels)


def calculate_false_discovery_rate(inferred_labels: list, truth_labels: list):
    """
    Determines the FP/TP+FP ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    return 1-calculate_positive_predictive_value(inferred_labels, truth_labels)

