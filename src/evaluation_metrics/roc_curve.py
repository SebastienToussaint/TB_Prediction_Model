import numpy as np
from evaluation_metrics.core_metrics import calculate_true_positive_rate, calculate_false_positive_rate
from matplotlib import pyplot as plt


def calculate_roc_values(inferred_probabilities: list, truth_labels: list):

    thresholds = ([1]+sorted(set(inferred_probabilities), reverse=True)+[-1])

    roc_coords = []

    for cutoff_value in thresholds:
        inferred_labels = []

        for probability in inferred_probabilities:
            if probability > cutoff_value:
                inferred_labels.append(1)
            else:
                inferred_labels.append(0)

        tpr = calculate_true_positive_rate(inferred_labels, truth_labels)
        fpr = calculate_false_positive_rate(inferred_labels, truth_labels)

        roc_coords.append([fpr, tpr])

    return roc_coords


def draw_roc_curve(inferred_probabilities: list, truth_labels: list, save_directory):
    x = []
    y = []
    coordinates = calculate_roc_values(inferred_probabilities, truth_labels)

    for coord in coordinates:
        x.append(coord[0])
        y.append(coord[1])

    plt.plot(x, y)

    plt.fill_between(x,y, alpha=0.4)
    # plt.plot([0, 1], linestyle="dashed")
    plt.title("ROC CURVE")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(str(save_directory) + "/roc_curve.svg")
    plt.clf()


def calculate_roc_auc(inferred_probabilities: list, truth_labels: list):
    coordinates = calculate_roc_values(inferred_probabilities, truth_labels)
    x = []
    y = []

    for coord in coordinates:
        x.append(coord[0])
        y.append(coord[1])

    return float(np.trapezoid(y, x))

def draw_multiple_model_roc_curve(inferred_probabilities: list, truth_labels: list, model_names: list, save_directory):
    plt.plot([0, 1], linestyle="dashed", color="gray", linewidth=1)

    for i, model_probabilities in enumerate(inferred_probabilities):

        x = []
        y = []
        coordinates = calculate_roc_values(model_probabilities, truth_labels[i])

        for coord in coordinates:
            x.append(coord[0])
            y.append(coord[1])

        plt.plot(x, y, linewidth=2)

    plt.legend(["Random classifier"] + model_names, frameon=False)
    plt.title("ROC curve comparison of multiple models")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(str(save_directory) + "/comparative_roc_curve.svg")
    plt.clf()
