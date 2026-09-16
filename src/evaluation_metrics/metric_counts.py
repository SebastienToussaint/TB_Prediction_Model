def get_true_positives(inferred_probabilities: list, truth_labels: list):
    """
    Calculates number of true positives. Assumes that 1 is Positive and 0 is Negative.
    Returns the number of TPs as well as the indices of the TP labels
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    """
    if len(inferred_probabilities) != len(truth_labels):
        print(f"ERROR - number of inferred labels ({len(inferred_probabilities)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    true_positives = 0
    true_positive_indices = []

    for i in range(len(inferred_probabilities)):
        if round(inferred_probabilities[i]) == 1 and truth_labels[i] == 1:
            true_positives += 1
            true_positive_indices.append(i)


    return true_positives, true_positive_indices

def get_false_positives(inferred_probabilities: list, truth_labels: list):
    """
    Calculates number of false positives. Assumes that 1 is Positive and 0 is Negative.
    Returns the number of FPs as well as the indices of the FP labels
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """
    if len(inferred_probabilities) != len(truth_labels):
        print(f"ERROR - number of inferred labels ({len(inferred_probabilities)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    false_positives = 0
    false_positive_indices = []

    for i in range(len(inferred_probabilities)):
        if round(inferred_probabilities[i]) == 1 and truth_labels[i] == 0:
            false_positives += 1
            false_positive_indices.append(i)

    return false_positives, false_positive_indices

def get_true_negatives(inferred_probabilities: list, truth_labels: list):
    """
    Calculates number of true negatives. Assumes that 1 is Positive and 0 is Negative.
    Returns the number of TNs as well as the indices of the TN labels
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """
    if len(inferred_probabilities) != len(truth_labels):
        print(f"ERROR - number of inferred labels ({len(inferred_probabilities)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    true_negatives = 0
    true_negative_indices = []

    for i in range(len(inferred_probabilities)):
        if round(inferred_probabilities[i]) == 0 and truth_labels[i] == 0:
            true_negatives += 1
            true_negative_indices.append(i)


    return true_negatives, true_negative_indices

def get_false_negatives(inferred_probabilities: list, truth_labels: list):
    """
    Calculates number of false negatives. Assumes that 1 is Positive and 0 is Negative.
    Returns the number of FNs as well as the indices of the FN labels
    :param inferred_probabilities: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_probabilities
    """
    if len(inferred_probabilities) != len(truth_labels):
        print(f"ERROR - number of inferred labels ({len(inferred_probabilities)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    false_negatives = 0
    false_negative_indices = []

    for i in range(len(inferred_probabilities)):
        if round(inferred_probabilities[i]) == 0 and truth_labels[i] == 1:
            false_negatives += 1
            false_negative_indices.append(i)


    return false_negatives, false_negative_indices
