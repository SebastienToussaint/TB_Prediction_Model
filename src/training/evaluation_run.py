import math
import os
import random
import time

import torch
from torch import nn
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
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def write_evaluations(inferred_probabilities, truth_labels, output_directory, round_values=4):

    print("Calculating evaluation metrics...")

    print(f"Base evaluation statistics:\n"
          f"Accuracy: "
          f"{round(accuracy.calculate_accuracy(inferred_probabilities, truth_labels), round_values)}\n"
          f"Balanced Accuracy: "
          f"{round(accuracy.calculate_balanced_accuracy(inferred_probabilities, truth_labels), round_values)}\n"
          f"MCC: "
          f"{round(mcc.calculate_matthews_correlation_coefficient(inferred_probabilities, truth_labels), round_values)}\n"
          f"ROC-AUC: "
          f"{round(roc_curve.calculate_roc_auc(inferred_probabilities, truth_labels), round_values)}")

     # Calculate Metrics

    metrics = [
        ["Accuracy:",
         round(accuracy.calculate_accuracy(inferred_probabilities, truth_labels), round_values)],
        ["Balanced Accuracy:",
         round(accuracy.calculate_balanced_accuracy(inferred_probabilities, truth_labels), round_values)],
        ["Mathews Correlation Coefficient:",
         round(mcc.calculate_matthews_correlation_coefficient(inferred_probabilities, truth_labels), round_values)],
        ["ROC-AUC:",
         round(roc_curve.calculate_roc_auc(inferred_probabilities, truth_labels), round_values)],
        ["Recall:",
         round(core_metrics.calculate_recall(inferred_probabilities, truth_labels), round_values)],
        ["Specificity:",
         round(core_metrics.calculate_specificity(inferred_probabilities, truth_labels), round_values)],
        ["Precision:",
         round(core_metrics.calculate_precision(inferred_probabilities, truth_labels), round_values)],
        ["PR-AUC:",
         round(pr_curve.calculate_pr_auc(inferred_probabilities, truth_labels), round_values)],
        ["Diagnostic Odds Ratio:",
         round(diagnostic_odds_ratio.calculate_diagnostic_odds_ratio(inferred_probabilities, truth_labels), round_values)],
        ["F1-Score:",
         round(f1_score.calculate_f1_score(inferred_probabilities, truth_labels), round_values)],
        ["Fowlkes Mallows Index:",
         round(fowlkes_mallows_index.calculate_fowlkes_mallows_index(inferred_probabilities, truth_labels), round_values)],
        ["Informedness:",
         round(informedness.calculate_informedness(inferred_probabilities, truth_labels), round_values)],
        ["Jaccard Index:",
         round(jaccard_index.calculate_jaccard_index(inferred_probabilities, truth_labels), round_values)],
        ["Markedness:",
         round(markedness.calculate_markedness(inferred_probabilities, truth_labels), round_values)],
        ["Prevalence Threshold",
         round(prevalence_threshold.calculate_prevalence_threshold(inferred_probabilities, truth_labels), round_values)]
    ]

    roc_curve.draw_roc_curve(inferred_probabilities, truth_labels, output_directory)
    pr_curve.draw_pr_curve(inferred_probabilities, truth_labels, output_directory)

    # Path determination
    output_filepath = str(Path(str(output_directory)).resolve())
    csv_filepath = output_filepath + "/evaluations.csv"


    with open(csv_filepath, "w", newline="") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerows(metrics)

    print(f"Finished writing evaluations to {output_filepath}")


def get_sample_indices(num_images, num_samples):
    """
    Samples images with replacement
    """
    selected_indices = []

    for i in range(num_samples):
        random_idx = random.randint(0, num_images-1)
        selected_indices.append(random_idx)

    return selected_indices


def infer_label_chunk(img_indices, model: nn.Module, device, image_packages):

    data = []
    print("PID:", os.getpid())
    print("torch threads:", torch.get_num_threads())
    print("interop threads:", torch.get_num_interop_threads())


    with torch.inference_mode():
        for idx in img_indices:
            img = image_packages[idx][0].to(device)
            inferred_logits = model(img.unsqueeze(0).unsqueeze(0)).squeeze()
            inferred_prediction_values = torch.sigmoid(inferred_logits)  # Round these to get labels

            data.append([inferred_prediction_values.detach().cpu().tolist(), image_packages[idx][1].detach().cpu().tolist()])

    return data


def run_evaluation(model: nn.Module, ds: DataStructure, output_directory, device: str = "cpu",
                   metrics_decimal_places=4,_no_output=False):

    print(f"Running evaluation on model using {device}...")

    num_partitions = ds.get_num_partitions()
    ds.reset_dataloader_index()

    # Load all images.
    image_packages = []
    for i in range(num_partitions):
        partition = ds.get_next_dataloader()
        for images, labels, filepaths in partition:
            for j in range(len(images)):
                image_packages.append([images[j], labels[j], filepaths[j]])

    selected_indices = get_sample_indices(len(image_packages), len(image_packages))

    model.to(device)
    model.eval()

    # Try send all of image_packages to device in the beginning? Look at space

    inferred_data = []
    # truth_labels =[]



    # print(selected_indices)


    time_1 = time.time()
    # inferred_probs.append(infer_label_chunk(selected_indices, model, device, image_packages))

    worker_count = 1
    print(f"Using {worker_count} workers.")

    infer_label_chunk_partial = partial(
        infer_label_chunk,
        model=model,
        device=device,
        image_packages=image_packages
    )

    num_indices_per_process = len(selected_indices)/worker_count

    print("PID:", os.getpid())
    print("torch threads:", torch.get_num_threads())
    print("interop threads:", torch.get_num_interop_threads())

    index_chunks = []

    for i in range(worker_count):
        index_chunks.append(selected_indices[0:int(math.ceil(min(num_indices_per_process, len(selected_indices))))])

        for j in range(int(math.ceil(min(num_indices_per_process, len(selected_indices))))):
            selected_indices.pop(0)


    print(index_chunks)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
            inferred_data = list(executor.map(infer_label_chunk_partial, index_chunks))


    print(inferred_data)

    print(f"{time.time() - time_1}")

    quit()
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


    print(len(flatten_list(inferred_probs)))

    # if not _no_output:
    #     verify_directory(output_directory, 2)
    #     write_evaluations(flatten_list(inferred_probs),
    #                       flatten_list(truth_labels),
    #                       output_directory,
    #                       metrics_decimal_places)
    #     return True
    # else:
    #     print("NOTE: no_output set to True in run_evaluation.\n"
    #           "Outputs will not be written to file, but will instead be returned to the calling script.")
    #     return flatten_list(inferred_probs), flatten_list(truth_labels)

if __name__ == "__main__":
    model = CNN()
    # model.create_standard(1, 64, 1)
    model.load("/home/sebastien/Project/SaveFiles/Models/SickVsTB/model_SickVsTB.pkl")
    ds = DataStructure()

    ds.set_native_datapath("/home/sebastien/Project/PIPELINETESTING/Dataloaders/TBX11K/SickVsTB_BS30/")
    run_evaluation(model, ds, "")