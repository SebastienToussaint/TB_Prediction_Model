import numpy as np
from evaluation_metrics.core_metrics import calculate_true_positive_rate, calculate_false_positive_rate
from matplotlib import pyplot as plt


def calculate_roc_values(inferred_probabilities: list, truth_labels: list):

    cutoff_value = 0.01

    roc_coords = []
    reached_one = False

    while cutoff_value <= 1:
        inferred_labels = []

        for probability in inferred_probabilities:
            if probability > cutoff_value:
                inferred_labels.append(1)
            else:
                inferred_labels.append(0)

        tpr = calculate_true_positive_rate(inferred_labels, truth_labels)
        fpr = calculate_false_positive_rate(inferred_labels, truth_labels)

        roc_coords.append([fpr, tpr])

        cutoff_value += 0.01

        if not reached_one:
            cutoff_value = min(cutoff_value, 1)
            reached_one = True

    return roc_coords


def draw_roc_curve(inferred_probabilities: list, truth_labels: list, save_directory):
    x = []
    y = []
    coordinates = calculate_roc_values(inferred_probabilities, truth_labels)

    for coord in coordinates: #TODO: do this with array slicing
        x.append(coord[0])
        y.append(coord[1])

    plt.plot(x, y)
    plt.fill_between(x,y, alpha=0.4)
    plt.savefig(str(save_directory) + "/roc_curve.svg")


def calculate_roc_auc(inferred_probabilities: list, truth_labels: list):
    coordinates = calculate_roc_values(inferred_probabilities, truth_labels)
    x = []
    y = []

    for coord in coordinates:  # TODO: do this with array slicing
        x.append(coord[0])
        y.append(coord[1])

    return float(-np.trapezoid(y, x))
