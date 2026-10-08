import random
import time
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
import pympler.asizeof


def write_evaluations(inferred_probabilities, truth_labels, cutoff_value, output_directory, round_values=4, _no_output=False):

    print("Calculating evaluation metrics...")

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
        inferred_logits = model(img.unsqueeze(0).unsqueeze(0)).squeeze()
        inferred_prediction_values = torch.sigmoid(inferred_logits)

        return [inferred_prediction_values.detach().cpu().tolist(), data[1].detach().cpu().tolist()]


def run_bootstrap_evaluation(repetitions: int, model: nn.Module, ds: DataStructure, output_directory, device: str = "cpu"):
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
    time_1 = time.time()
    for i in range(repetitions):
        inferred_data.append([])
        # Get indices
        selected_indices = get_sample_indices(len(image_packages), len(image_packages))

        # Get inferences
        for idx in selected_indices:
            inferred_data[i].append(infer_labels_cuda(model, image_packages[idx]))


    print(time.time() - time_1)


    time_1 = time.time()
    metrics = []
    # Get metrics
    for chunk in inferred_data:
        metrics.append(write_evaluations([package[0] for package in chunk], [package[0] for package in chunk], "", 3, True))

    print(metrics)
    print(time.time() - time_1)


def run_evaluation(model: nn.Module, ds: DataStructure, cutoff_value,output_directory, device: str = "cpu",
                   metrics_decimal_places=4,_no_output=False):

    print(f"Running evaluation on model using {device} with cutoff value {cutoff_value}...")

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

                images = images.unsqueeze(1)

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
    model.load("E:\Project\Savefiles\Models\Retrained_model\model_SickVsTB_retrained.pkl")
    datas = DataStructure()

    datas.set_native_datapath("E:\Project\TESTING\Data\Dataloaders\Test")
    # run_bootstrap_evaluation(1, model, datas, "", "cuda")
    # run_evaluation()