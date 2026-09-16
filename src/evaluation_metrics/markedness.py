from evaluation_metrics.core_metrics import calculate_positive_predictive_value, calculate_negative_predictive_value

def calculate_markedness(inferred_probabilities: list, truth_labels: list):
    """
    Determines the PPV + NPV - 1 value. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    ppv = calculate_positive_predictive_value(inferred_probabilities, truth_labels)
    npv = calculate_negative_predictive_value(inferred_probabilities, truth_labels)

    return ppv + npv - 1