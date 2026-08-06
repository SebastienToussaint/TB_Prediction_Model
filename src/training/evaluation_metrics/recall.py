def calculate_recall(inferred_labels: list, truth_labels: list):
    """
    Determines the TP/TP+FN ratio. Assumes that 1 is Positive and 0 is Negative.
    Returns the recall as well as the indices of the False Negative labels
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    :return:
    """
    if len(inferred_labels) != len(truth_labels):
        print(f"ERROR in `calculate_recall()` - number of inferred labels ({len(inferred_labels)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    true_positives = 0
    false_negatives = 0
    false_negative_indices= []

    for i in range(len(inferred_labels)):
        if inferred_labels[i] == 1 and truth_labels[i] == 1:
            true_positives += 1

        elif inferred_labels[i] == 0 and truth_labels[i] == 1:
            false_negatives += 1
            false_negative_indices.append(i)

    return true_positives/(true_positives+false_negatives), false_negative_indices
