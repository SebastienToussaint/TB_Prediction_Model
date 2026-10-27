import random
import numpy as np
import torch
from pympler.asizeof import asizeof
from torch import nn
import utilities.string_manipulation
from  shared_objects.data_handling import CustomDataset, DataStructure
from evaluation_metrics import accuracy, core_metrics, diagnostic_odds_ratio, f1_score, fowlkes_mallows_index
from evaluation_metrics import informedness, jaccard_index, markedness, mcc, prevalence_threshold
from evaluation_metrics import roc_curve, pr_curve
from shared_objects.models import CNN
from utilities.directory_handling import verify_directory
from pathlib import Path
import csv
from utilities.array_handling import flatten_list
import matplotlib.pyplot as plt


def write_evaluations(inferred_probabilities, truth_labels, cutoff_value,output_directory, round_values=4, _no_output=False):

    if not _no_output:
        print("Calculating evaluation metrics...")



    if str(cutoff_value).upper() == "J":
        cutoff_value = roc_curve.get_youdens_j_point(inferred_probabilities, truth_labels)


    if not _no_output:
        print(f"Base evaluation statistics:\n"
              f"Accuracy: "
              f"{round(accuracy.calculate_accuracy(inferred_probabilities, truth_labels, cutoff_value), round_values)}\n"
              f"Balanced Accuracy: "
              f"{round(accuracy.calculate_balanced_accuracy(inferred_probabilities, truth_labels, cutoff_value), round_values)}\n"
              f"MCC: "
              f"{round(mcc.calculate_matthews_correlation_coefficient(inferred_probabilities, truth_labels, cutoff_value), round_values)}\n"
              f"ROC-AUC: "
              f"{round(roc_curve.calculate_roc_auc(inferred_probabilities, truth_labels), round_values)}")

     # Calculate Metrics

    metrics = [
        ["Accuracy:",
         round(accuracy.calculate_accuracy(inferred_probabilities, truth_labels, cutoff_value), round_values)],
        ["Balanced Accuracy:",
         round(accuracy.calculate_balanced_accuracy(inferred_probabilities, truth_labels, cutoff_value), round_values)],
        ["Mathews Correlation Coefficient:",
         round(mcc.calculate_matthews_correlation_coefficient(inferred_probabilities, truth_labels, cutoff_value), round_values)],
        ["ROC-AUC:",
         round(roc_curve.calculate_roc_auc(inferred_probabilities, truth_labels, ), round_values)],
        ["Sensitivity:",
         round(core_metrics.calculate_sensitivity(inferred_probabilities, truth_labels, cutoff_value), round_values)],
        ["Specificity:",
         round(core_metrics.calculate_specificity(inferred_probabilities, truth_labels, cutoff_value), round_values)],
        ["Precision:",
         round(core_metrics.calculate_precision(inferred_probabilities, truth_labels, cutoff_value), round_values)],
        ["PR-AUC:",
         round(pr_curve.calculate_pr_auc(inferred_probabilities, truth_labels), round_values)],
        ["Cutoff:",
         cutoff_value]
    ]

    if _no_output:
        return metrics


    roc_curve.draw_roc_curve(inferred_probabilities, truth_labels, output_directory)
    pr_curve.draw_pr_curve(inferred_probabilities, truth_labels, output_directory)

    # Path determination
    output_filepath = str(Path(str(output_directory)).resolve())

    verify_directory(utilities.string_manipulation.extract_base_directory_from_path(output_directory), 2)

    csv_filepath = output_filepath + "/evaluations.csv"


    with open(csv_filepath, "w", newline="") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerows(metrics)

    print(f"Finished writing evaluations to {output_filepath}")

    return True


def get_sample_indices(num_images, num_samples):
    """
    Samples images with replacement
    """
    selected_indices = []

    for i in range(num_samples):
        random_idx = random.randint(0, num_images-1)
        selected_indices.append(random_idx)

    return selected_indices


def infer_labels_cuda(model, data):
    model.eval()

    with torch.inference_mode():
        img = data[0]
        inferred_logits = model(img.unsqueeze(0)).squeeze()
        inferred_prediction_values = torch.sigmoid(inferred_logits)

        return [inferred_prediction_values.detach().cpu().tolist(), data[1].detach().cpu().tolist()]


def run_bootstrap_evaluation(repetitions: int, model: nn.Module, ds: DataStructure, output_directory, cutoff,device: str = "cpu"):
    print(f"Running bootstrap evaluation on model using {device}...")

    # Get Data
    num_partitions = ds.get_num_partitions()
    ds.reset_dataloader_index()
    image_packages = []
    for i in range(num_partitions):
        partition = ds.get_next_dataloader()
        for images, labels, filepaths in partition:
            for j in range(len(images)):
                image_packages.append([images[j], labels[j], filepaths[j]])

    # Data and model to device
    for i in range(len(image_packages)):
        image_packages[i][0] = image_packages[i][0].to(device)

    model = model.to(device)
    inferred_data = []

    for i in range(repetitions):
        print(f"Bootstrapping fold {i+1} of {repetitions}")
        inferred_data.append([])
        # Get indices
        selected_indices = get_sample_indices(len(image_packages), len(image_packages))

        # Get inferences
        for idx in selected_indices:
            inferred_data[i].append(infer_labels_cuda(model, image_packages[idx]))


    metrics = []
    # Get metrics
    for chunk in inferred_data:
        metrics.append(
            write_evaluations(inferred_probabilities=[package[0] for package in chunk],
                              truth_labels=[package[1] for package in chunk],
                              cutoff_value=cutoff,
                              output_directory="",
                              round_values=3,
                              _no_output=True))


    accuracies = []
    balanced_accuracies = []
    mcc = []
    roc_auc = []
    sensitivities = []
    specificities = []
    precisions = []
    pr_auc = []

    for metric in metrics:
        accuracies.append(metric[0][1])
        balanced_accuracies.append(metric[1][1])
        mcc.append(metric[2][1])
        roc_auc.append(metric[3][1])
        sensitivities.append(metric[4][1])
        specificities.append(metric[5][1])
        precisions.append(metric[6][1])
        pr_auc.append(metric[7][1])


    bootstrapped_metrics = []
    bootstrapped_metrics.append(["Accuracy:", f"{round(np.mean(accuracies), 3)} ± {round(np.std(accuracies), 4)}"])
    bootstrapped_metrics.append(["Balanced Accuracy:", f"{round(np.mean(balanced_accuracies), 3)} ± {round(np.std(balanced_accuracies), 4)}"])
    bootstrapped_metrics.append(["Mathews Correlation Coefficient:", f"{round(np.mean(mcc), 3)} ± {round(np.std(mcc), 4)}"])
    bootstrapped_metrics.append(["ROC-AUC:", f"{round(np.mean(roc_auc), 3)} ± {round(np.std(roc_auc), 4)}"])
    bootstrapped_metrics.append(["Sensitivity:", f"{round(np.mean(sensitivities), 3)} ± {round(np.std(sensitivities), 4)}"])
    bootstrapped_metrics.append(["Specificity:", f"{round(np.mean(specificities), 3)} ± {round(np.std(specificities), 4)}"])
    bootstrapped_metrics.append(["Precision:", f"{round(np.mean(precisions), 3)} ± {round(np.std(precisions), 4)}"])
    bootstrapped_metrics.append(["PR-AUC:", f"{round(np.mean(pr_auc), 3)} ± {round(np.std(pr_auc), 4)}"])
    bootstrapped_metrics.append(["Cutoff:", f"{cutoff}"])


    with open(str(output_directory) + "/bootstrapped_evaluations.csv", "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(bootstrapped_metrics)

    print(f"Finished bootstrap evaluation.\n"
          f"Metrics saved at {str(output_directory) + "/bootstrapped_evaluations.csv"}")


def run_evaluation(model: nn.Module, ds: DataStructure, cutoff_value, output_directory, device: str = "cpu",
                    target_sensitivity=None, metrics_decimal_places=4,_no_output=False):
    print(f"Running evaluation on model using {device}")

    if target_sensitivity and cutoff_value:
        print("WARNING in run_evaluation.\n"
              "cutoff_value and target_sensitivity are both set. cutoff_value will be used.\n"
              "If target_sensitivity should be used to calculate the cutoff, set cutoff value to None")
        target_sensitivity = None
    elif cutoff_value:
        print(f"Using cutoff value of {cutoff_value}")
    elif target_sensitivity:
        print(f"Using target sensitivity of {target_sensitivity} to calculate threshold")
    else:
        print("ERROR in run_evaluation.\n"
              "one of cutoff_value or target_sensitivity needs to be set.")
        quit()

    model.to(device)
    inferred_probs = []
    truth_labels = []

    num_partitions = ds.get_num_partitions()
    ds.reset_dataloader_index()

    with torch.inference_mode():
        for i in range(num_partitions):
            partition = ds.get_next_dataloader()

            for images, labels, filepaths in partition:
                images, labels = images.to(device), labels.to(device)

                # images = images.unsqueeze(1)

                inferred_logits = model(images).squeeze()
                inferred_prediction_values = torch.sigmoid(inferred_logits)  # Round these to get labels

                # Save inferred probabilities and truth labels
                inferred_probs.append(inferred_prediction_values.detach().cpu().tolist())
                truth_labels.append(labels.detach().cpu().tolist())

    if not _no_output:
        verify_directory(output_directory, 2)
        write_evaluations(flatten_list(inferred_probs),
                          flatten_list(truth_labels),
                          cutoff_value,
                          output_directory,
                          metrics_decimal_places)
        return True
    else:
        print("NOTE: no_output set to True in run_evaluation.\n"
              "Outputs will not be written to file, but will instead be returned to the calling script.")
        return flatten_list(inferred_probs), flatten_list(truth_labels)

if __name__ == "__main__":
    model = CNN()
    model.load("E:\Project\Savefiles\Final_Segmented_Models\SHENTBX_TBvsHealthy\shen_tbx_model_HvTB.pkl")
    datas = DataStructure()

    datas.set_native_datapath("E:\Project\data\Split\TrainingDataloaders\TBX_SHEN_HvTB_EVAL")
    run_bootstrap_evaluation(500, model, datas, "E:\Project\Savefiles\Final_Segmented_Models\SHENTBX_TBvsHealthy/", 0.411,"cuda")
    # run_evaluation()