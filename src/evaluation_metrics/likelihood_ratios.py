from evaluation_metrics.metric_counts import get_true_positives, get_true_negatives, get_false_positives, get_false_negatives

def calculate_positive_likelihood_ratio(inferred_probabilities: list, truth_labels: list):
    """
    Determines the TPR/FPR ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """


    tp = get_true_positives(inferred_probabilities, truth_labels)[0] + 0.5 # Pseudocounts to avoid divide by 0
    tn = get_true_negatives(inferred_probabilities, truth_labels)[0] + 0.5
    fp = get_false_positives(inferred_probabilities, truth_labels)[0] + 0.5
    fn = get_false_negatives(inferred_probabilities, truth_labels)[0] + 0.5

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    return tpr/fpr

def calculate_negative_likelihood_ratio(inferred_probabilities: list, truth_labels: list):
    """
    Determines the FNR/TNR. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """

    tp = get_true_positives(inferred_probabilities, truth_labels)[0] + 0.5
    tn = get_true_negatives(inferred_probabilities, truth_labels)[0] + 0.5
    fp = get_false_positives(inferred_probabilities, truth_labels)[0] + 0.5
    fn = get_false_negatives(inferred_probabilities, truth_labels)[0] + 0.5

    fnr = fn/(fn+tp)
    tnr = tn/(tn+fp)

    return fnr/tnr
