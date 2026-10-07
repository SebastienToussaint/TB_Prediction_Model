from math import sqrt
from evaluation_metrics.metric_counts import get_true_positives, get_true_negatives, get_false_positives, get_false_negatives


def calculate_prevalence_threshold(inferred_probabilities: list, truth_labels: list, cutoff=0.5):
    """
    Determines the (sqrt(TPRxFPR)-FPR)/(TPR-FPR) value. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tp = get_true_positives(inferred_probabilities, truth_labels, cutoff)[0] + 0.5
    tn = get_true_negatives(inferred_probabilities, truth_labels, cutoff)[0] + 0.5
    fp = get_false_positives(inferred_probabilities, truth_labels, cutoff)[0] + 0.5
    fn = get_false_negatives(inferred_probabilities, truth_labels, cutoff)[0] + 0.5

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    top = sqrt(tpr*fpr) - fpr
    bottom = tpr - fpr

    if bottom == 0:
        print("Warning in prevalence threshold calculation. TPR-FPR = 0 which would lead to division by 0 error.\n"
              "Returning 0.5 as the prevalence threshold value.")
        return 0.5

    return top/bottom
