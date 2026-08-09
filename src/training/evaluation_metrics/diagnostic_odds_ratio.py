from likelihood_ratios import calculate_positive_likelihood_ratio, calculate_negative_likelihood_ratio

def calculate_diagnostic_odds_ratio(inferred_labels: list, truth_labels: list):
    """
    Determines the LR+/LR- ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """

    lr_positive = calculate_positive_likelihood_ratio(inferred_labels, truth_labels)
    lr_negative = calculate_negative_likelihood_ratio(inferred_labels, truth_labels)

    return lr_positive/lr_negative
