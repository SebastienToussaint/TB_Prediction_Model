def calculate_accuracy(inferred_labels: list, truth_labels: list):
    """
    Determines the ratio of how many labels were correctly predicted
    Returns the accuracy as well as the indices of incorrect labels
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    :return:
    """
    if len(inferred_labels) != len(truth_labels):
        print(f"ERROR in `calculate_accuracy()` - number of inferred labels ({len(inferred_labels)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    correctly_predicted_labels = 0
    incorrect_label_indices = []

    for i in range(len(inferred_labels)):
        if inferred_labels[i] == truth_labels[i]:
            correctly_predicted_labels += 1
        else:
            incorrect_label_indices.append(i)

    return correctly_predicted_labels/len(truth_labels), incorrect_label_indices
