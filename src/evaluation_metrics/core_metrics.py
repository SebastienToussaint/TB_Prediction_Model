from evaluation_metrics.metric_counts import get_true_positives, get_false_negatives, get_true_negatives, get_false_positives

def calculate_true_positive_rate(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the TP/TP+FN ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    true_positives = get_true_positives(inferred_probabilities, truth_labels, cutoff)[0]
    false_negatives = get_false_negatives(inferred_probabilities, truth_labels, cutoff)[0]

    if true_positives+false_negatives == 0:
        return 0
    return true_positives/(true_positives+false_negatives)


def calculate_sensitivity(inferred_probabilities: list, truth_labels: list, cutoff=0.5): # For ease of use
    return calculate_true_positive_rate(inferred_probabilities, truth_labels, cutoff)


def calculate_false_positive_rate(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the FP/FP+TN ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    return 1-calculate_true_negative_rate(inferred_probabilities, truth_labels, cutoff)


def calculate_true_negative_rate(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the TN/TN+FP ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    true_negatives = get_true_negatives(inferred_probabilities, truth_labels, cutoff)[0]
    false_positives = get_false_positives(inferred_probabilities, truth_labels, cutoff)[0]

    if true_negatives+false_positives == 0:
        return 0

    return true_negatives/(true_negatives+false_positives)


def calculate_specificity(inferred_probabilities: list, truth_labels: list, cutoff=0.5): # For ease of use
    return 1-calculate_false_positive_rate(inferred_probabilities, truth_labels, cutoff)


def calculate_false_negative_rate(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the FN/FN+TP ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    return 1-calculate_true_positive_rate(inferred_probabilities, truth_labels, cutoff)


def calculate_positive_predictive_value(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the TP/TP+FP ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    true_positives = get_true_positives(inferred_probabilities, truth_labels, cutoff)[0]
    false_positives = get_false_positives(inferred_probabilities, truth_labels, cutoff)[0]

    if true_positives+false_positives == 0:
        return 0

    return true_positives/(true_positives+false_positives)


def calculate_precision(inferred_probabilities: list, truth_labels: list, cutoff=0.5): # For ease of use
    return calculate_positive_predictive_value(inferred_probabilities, truth_labels, cutoff)


def calculate_negative_predictive_value(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the TN/TN+FN ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    true_negatives = get_true_negatives(inferred_probabilities, truth_labels, cutoff)[0]
    false_negatives = get_false_negatives(inferred_probabilities, truth_labels, cutoff)[0]

    if true_negatives+false_negatives == 0:
        return 0

    return true_negatives/(true_negatives+false_negatives)


def calculate_false_omission_rate(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the FN/TN+FN ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    return 1-calculate_negative_predictive_value(inferred_probabilities, truth_labels, cutoff)


def calculate_false_discovery_rate(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the FP/TP+FP ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    return 1-calculate_positive_predictive_value(inferred_probabilities, truth_labels, cutoff)

