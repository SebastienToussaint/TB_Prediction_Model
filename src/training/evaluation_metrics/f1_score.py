from precision import calculate_precision
from recall import calculate_recall

def calculate_f1_score(inferred_labels: list, truth_labels: list):
    """
    Determines the ratio of 2(Precision x Recall)/(Precision + Recall) ratio. Assumes that 1 is Positive and 0 is Negative.
    :param inferred_labels: Predictions of model
    :param truth_labels: Ground truth labels. Indices must match inferred_labels
    :return:
    """
    if len(inferred_labels) != len(truth_labels):
        print(f"ERROR in `calculate_f1_score()` - number of inferred labels ({len(inferred_labels)}) does "
              f"not equal number of truth labels ({len(truth_labels)})")
        quit()

    recall, buffer = calculate_recall(inferred_labels, truth_labels)
    precision, buffer = calculate_precision(inferred_labels, truth_labels)

    return (2*(precision*recall))/(precision + recall)
