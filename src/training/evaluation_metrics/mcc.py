from math import sqrt

def calculate_matthews_correlation_coefficient(inferred_labels: list, truth_labels: list):
    """
    :param inferred_labels:
    :param truth_labels:
    :return:
    """

    if len(inferred_labels) != len(truth_labels):
        print(f"ERROR in `calculate_matthews_correlation_coefficient()` - number of inferred labels ({len(inferred_labels)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(len(inferred_labels)):
        if inferred_labels[i] == 1:
            # TP
            if truth_labels[i] == 1:
                true_positives += 1
            # FP
            elif truth_labels[i] == 0:
                false_positives += 1

        elif inferred_labels[i] == 0:
            # TN
            if truth_labels[i] == 0:
                true_negatives += 1
            # FN
            elif truth_labels[i] == 1:
                false_negatives += 1

    top = (true_positives*true_negatives)-(false_positives*false_negatives)
    bottom = sqrt((true_positives+false_positives)*(true_positives+false_negatives)*(true_negatives+false_positives)*(true_negatives+false_negatives))

    return top/bottom
