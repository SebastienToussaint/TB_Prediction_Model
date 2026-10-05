import numpy as np
from evaluation_metrics.core_metrics import calculate_recall, calculate_precision
from matplotlib import pyplot as plt


def calculate_pr_values(inferred_probabilities: list, truth_labels: list):

    thresholds = (sorted(set(inferred_probabilities), reverse=True))

    pr_coords = [[0.0, 1.0]]

    for cutoff_value in thresholds:
        inferred_labels = []

        for probability in inferred_probabilities:
            if probability >= cutoff_value:
                inferred_labels.append(1)
            else:
                inferred_labels.append(0)

        recall = calculate_recall(inferred_labels, truth_labels)
        precision = calculate_precision(inferred_labels, truth_labels)

        pr_coords.append([recall, precision])

    return pr_coords


def draw_pr_curve(inferred_probabilities: list, truth_labels: list, save_directory):
    x = []
    y = []

    coordinates = calculate_pr_values(inferred_probabilities, truth_labels)

    for coord in coordinates:
        x.append(coord[0])
        y.append(coord[1])

    plt.plot(x, y)
    plt.fill_between(x,y, alpha=0.4)
    plt.title("PR CURVE")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(str(save_directory) + "/pr_curve.svg")
    plt.clf()


def calculate_pr_auc(inferred_probabilities: list, truth_labels: list):
    coordinates = calculate_pr_values(inferred_probabilities, truth_labels)
    x = []
    y = []

    coordinates = sorted(coordinates, key=lambda coord: coord[0])

    for coord in coordinates:
        x.append(coord[0])
        y.append(coord[1])

    return float(np.trapezoid(y, x))



