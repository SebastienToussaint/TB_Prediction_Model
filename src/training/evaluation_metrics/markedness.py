from .core_metrics import calculate_positive_predictive_value, calculate_negative_predictive_value

def calculate_markedness(inferred_labels: list, truth_labels: list):
    """
    Determines the PPV + NPV - 1 value. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    ppv = calculate_positive_predictive_value(inferred_labels, truth_labels)
    npv = calculate_negative_predictive_value(inferred_labels, truth_labels)

    return ppv + npv - 1