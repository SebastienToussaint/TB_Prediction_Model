import numpy as np
from evaluation_metrics.core_metrics import calculate_true_positive_rate, calculate_false_positive_rate, calculate_sensitivity, calculate_specificity
from matplotlib import pyplot as plt


def get_youdens_j_point(inferred_probabilities: list, truth_labels: list):
    thresholds = ([1] + sorted(set(inferred_probabilities), reverse=True) + [-1])

    max_j_value = -1
    j_point = 0.5

    for cutoff_value in thresholds:
        # inferred_labels = []

        # for probability in inferred_probabilities:
            # if probability > cutoff_value:
            #     inferred_labels.append(1)
            # else:
            #     inferred_labels.append(0)

        tpr = calculate_true_positive_rate(inferred_probabilities, truth_labels, cutoff_value)
        fpr = calculate_false_positive_rate(inferred_probabilities, truth_labels, cutoff_value)

        if tpr-fpr > max_j_value:
            max_j_value = tpr-fpr
            j_point = cutoff_value

    return j_point

def get_threshold_for_sensitivity(inferred_probabilities: list, truth_labels: list, target_sensitivity):

    if not 0 <= target_sensitivity <= 1:
        print("Error while calculating threshold for sensitivity.\n"
              f"Target sensitivity needs to be between 0 and 1 (given: {target_sensitivity})")
        quit()

    thresholds = ([1] + sorted(set(inferred_probabilities), reverse=True) + [-1])

    distance_to_target = 1
    best_cutoff = 0.5

    for cutoff_value in thresholds:
        inferred_labels = []

        # for probability in inferred_probabilities:
        #     if probability > cutoff_value:
        #         inferred_labels.append(1)
        #     else:
        #         inferred_labels.append(0)

        sensitivity = calculate_sensitivity(inferred_probabilities, truth_labels,cutoff_value)

        if abs(sensitivity-target_sensitivity) < distance_to_target:
            best_cutoff = cutoff_value
            distance_to_target = abs(sensitivity-target_sensitivity)

            print(f"{best_cutoff} cutoff for sensitivity: {sensitivity} and specificity: {calculate_specificity(inferred_probabilities, truth_labels, cutoff_value)}")

    return best_cutoff


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

        tpr = calculate_true_positive_rate(inferred_labels, truth_labels, cutoff_value)
        fpr = calculate_false_positive_rate(inferred_labels, truth_labels, cutoff_value)

        roc_coords.append([fpr, tpr])

    return roc_coords


def draw_roc_curve(inferred_probabilities: list, truth_labels: list, save_directory):
    x = []
    y = []
    coordinates = calculate_roc_values(inferred_probabilities, truth_labels)
    youdens_j_point = get_youdens_j_point(inferred_probabilities, truth_labels)

    for coord in coordinates:
        x.append(coord[0])
        y.append(coord[1])

    plt.plot(x, y)

    # plt.fill_between(x,y, alpha=0.4)
    plt.plot([0, 1], linestyle="dashed")

    plt.axvline(youdens_j_point, linestyle="--")
    plt.title("ROC CURVE")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(["ROC-Curve", "Random model", "Youdens J point"])
    plt.savefig(str(save_directory) + "/roc_curve.svg")
    plt.clf()


def calculate_roc_auc(inferred_probabilities: list, truth_labels: list):
    coordinates = calculate_roc_values(inferred_probabilities, truth_labels)
    x = []
    y = []



    get_threshold_for_sensitivity(inferred_probabilities, truth_labels, 0.8)

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
