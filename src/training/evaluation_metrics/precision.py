def calculate_precision(inferred_labels: list, truth_labels: list):
    """
    Determines the TP/TP+FP ratio. Assumes that 1 is Positive and 0 is Negative.
    Returns the precision as well as the indices of False Positive labels
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    :return:
    """
    if len(inferred_labels) != len(truth_labels):
        print(f"ERROR in `calculate_precision()` - number of inferred labels ({len(inferred_labels)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    true_positives = 0
    false_positives = 0
    false_positive_indices= []

    for i in range(len(inferred_labels)):
        if inferred_labels[i] == 1: # Since we only care about positives
            if truth_labels[i] == 1:
                true_positives += 1
            else:
                false_positives += 1
                false_positive_indices.append(i)

    return true_positives/(true_positives+false_positives), false_positive_indices
