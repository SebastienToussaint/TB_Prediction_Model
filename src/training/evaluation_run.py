import torch
from torch import nn
from  shared_objects.data_handling import CustomDataset, DataStructure
from evaluation_metrics import accuracy, core_metrics, diagnostic_odds_ratio, f1_score, fowlkes_mallows_index
from evaluation_metrics import informedness, jaccard_index, markedness, mcc, prevalence_threshold
from evaluation_metrics import roc_curve, pr_curve
from utilities.directory_handling import verify_directory
from pathlib import Path
import csv
from utilities.array_handling import flatten_list


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



def run_evaluation(model: nn.Module, ds: DataStructure, output_directory, device: str = "cpu",
                   metrics_decimal_places=4,_no_output=False):

    print(f"Running evaluation on model using {device}...")

    num_partitions = ds.get_num_partitions()

    model.to(device)
    model.eval()

    inferred_probs = []
    truth_labels =[]

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
                          output_directory,
                          metrics_decimal_places)
        return True
    else:
        print("NOTE: no_output set to True in run_evaluation.\n"
              "Outputs will not be written to file, but will instead be returned to the calling script.")
        return flatten_list(inferred_probs), flatten_list(truth_labels)
