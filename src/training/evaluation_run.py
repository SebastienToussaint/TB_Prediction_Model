import torch
from torch import nn
from  shared_objects import data_handling
from evaluation_metrics import accuracy, core_metrics, diagnostic_odds_ratio, f1_score, fowlkes_mallows_index
from evaluation_metrics import informedness, jaccard_index, markedness, mcc, prevalence_threshold
from evaluation_metrics import roc_curve
from utilities.directory_handling import verify_directory
from pathlib import Path
import csv
from utilities.array_handling import flatten_list



def write_evaluations(inferred_probabilities, truth_labels, output_directory):

    print("Calculating evaluation metrics...")

    round_values = 4
    print(f"Base evaluation statistics:\n"
          f"Accuracy: {round(accuracy.calculate_accuracy(inferred_probabilities, truth_labels), round_values)}\n"
          f"Balanced Accuracy: {round(accuracy.calculate_balanced_accuracy(inferred_probabilities, truth_labels), round_values)}\n"
          f"MCC: {round(mcc.calculate_matthews_correlation_coefficient(inferred_probabilities, truth_labels), round_values)}")

     # Calculate Metrics

    metrics = [
        ["Accuracy:", accuracy.calculate_accuracy(inferred_probabilities, truth_labels)],
        ["Balanced Accuracy:", accuracy.calculate_balanced_accuracy(inferred_probabilities, truth_labels)],
        ["Mathews Correlation Coefficient:", mcc.calculate_matthews_correlation_coefficient(inferred_probabilities, truth_labels)],
        ["ROC_AUC:", roc_curve.calculate_roc_auc(inferred_probabilities, truth_labels)],
        ["Recall:", core_metrics.calculate_recall(inferred_probabilities, truth_labels)],
        ["Specificity:", core_metrics.calculate_specificity(inferred_probabilities, truth_labels)],
        ["Precision:", core_metrics.calculate_precision(inferred_probabilities, truth_labels)],
        ["Diagnostic Odds Ratio:", diagnostic_odds_ratio.calculate_diagnostic_odds_ratio(inferred_probabilities, truth_labels)],
        ["F1-Score:", f1_score.calculate_f1_score(inferred_probabilities, truth_labels)],
        ["Fowlkes Mallows Index:", fowlkes_mallows_index.calculate_fowlkes_mallows_index(inferred_probabilities, truth_labels)],
        ["Informedness:", informedness.calculate_informedness(inferred_probabilities, truth_labels)],
        ["Jaccard Index:", jaccard_index.calculate_jaccard_index(inferred_probabilities, truth_labels)],
        ["Markedness:", markedness.calculate_markedness(inferred_probabilities, truth_labels)],
        ["Prevalence Threshold", prevalence_threshold.calculate_prevalence_threshold(inferred_probabilities, truth_labels)]
    ]

    roc_curve.draw_roc_curve(inferred_probabilities, truth_labels, output_directory)

    # Path determination
    output_filepath = str(Path(str(output_directory)).resolve())
    csv_filepath = output_filepath + "/evaluations.csv"


    with open(csv_filepath, "w") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerows(metrics)

    print(f"Finished writing evaluations to {output_filepath}")



def run_evaluation(model: nn.Module, ds: data_handling.DataStructure, output_directory, device: str = "cpu"):

    verify_directory(output_directory, 2)

    print(f"Running evaluation on model using {device}...")

    num_partitions = ds.get_num_partitions()

    model.to(device)
    model.eval()

    inferred_probs = []
    truth_labels =[]

    with torch.inference_mode():
        for i in range(num_partitions):
            partition = ds.get_next_dataloader()

            for images, labels in partition:
                images, labels = images.to(device), labels.to(device)

                images = images.unsqueeze(1)

                inferred_logits = model(images).squeeze()
                inferred_prediction_values = torch.sigmoid(inferred_logits)  # Round these to get labels

                # Save inferred probabilities and truth labels
                inferred_probs.append(inferred_prediction_values.detach().cpu().tolist())
                truth_labels.append(labels.detach().cpu().tolist())


    write_evaluations(flatten_list(inferred_probs), flatten_list(truth_labels), output_directory)
