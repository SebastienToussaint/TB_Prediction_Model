import numpy as np
from core_metrics import calculate_recall, calculate_precision
from matplotlib import pyplot as plt


def calculate_pr_values(inferred_probabilities: list, truth_labels: list):

    cutoff_value = 0.01

    pr_coords = []
    reached_one = False

    while cutoff_value <= 1:
        inferred_labels = []

        for probability in inferred_probabilities:
            if probability > cutoff_value:
                inferred_labels.append(1)
            else:
                inferred_labels.append(0)

        recall = calculate_recall(inferred_labels, truth_labels)
        precision = calculate_precision(inferred_labels, truth_labels)

        pr_coords.append([recall, precision])

        cutoff_value += 0.01

        if not reached_one:
            cutoff_value = min(cutoff_value, 1)
            reached_one = True

    return pr_coords


def draw_pr_curve(coordinates: list):
    x = []
    y = []

    for coord in coordinates: #TODO: do this with array slicing
        x.append(coord[0])
        y.append(coord[1])



    plt.plot(x, y)
    plt.fill_between(x,y, alpha=0.4)
    plt.show()


def calculate_pr_auc(inferred_probabilities: list, truth_labels: list):
    coordinates = calculate_pr_values(inferred_probabilities, truth_labels)
    x = []
    y = []

    for coord in coordinates:  # TODO: do this with array slicing
        x.append(coord[0])
        y.append(coord[1])

    return float(-np.trapezoid(y, x))



