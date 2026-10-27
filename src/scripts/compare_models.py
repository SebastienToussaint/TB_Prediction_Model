import sys
from pathlib import Path

import torch.cuda

from utilities.string_manipulation import extract_filename_from_path
from training.evaluation_run import run_evaluation
from shared_objects.models import CNN
from shared_objects.data_handling import DataStructure, CustomDataset
from evaluation_metrics import accuracy
from evaluation_metrics import mcc
from evaluation_metrics import roc_curve
from evaluation_metrics import core_metrics
from evaluation_metrics import pr_curve
from utilities.directory_handling import verify_directory
import csv

if __name__ == "__main__":
    # Argument management
    arguments = []
    for i in range(1,len(sys.argv)):
        arguments.append(sys.argv[i])

    if arguments.count("-o") != 1 or arguments.count("-d") != 1:
        print("Error in compare_models. Incorrect arguments provided.\n"
              "Missing output or dataloader tag.\n"
              "Usage: python -m *.compare_models {<model_path>['->'<model_name> (OPTIONAL)] (REPEATABLE)} <-d> <path_to_dataloader_directory> <-o> <output_directory>")
        quit(1)

    try:
        dataloader_directory = Path(arguments.pop(arguments.index("-d") + 1)).resolve()
        arguments.pop(arguments.index("-d"))

        output_directory = Path(arguments.pop(arguments.index("-o") + 1)).resolve()
        arguments.pop(arguments.index("-o"))
    except IndexError:
        print("Error in compare_models. Incorrect arguments provided.\n"
              "Missing parameter after a tag.\n"
              "Usage: python -m *.compare_models {<model_path>['->'<model_name> (OPTIONAL)] (REPEATABLE)} <-d> <path_to_dataloader_directory> <-o> <output_directory>")
        quit(1)

    # Create datastructure
    ds = DataStructure()
    ds.set_native_datapath(dataloader_directory)

    verify_directory(output_directory, 2)

    # Initialize models array
    models = []
    for i, arg in enumerate(arguments):
        models.append({})
        if "->" in arg:
            models[i]["path"] = Path(arg[0:arg.index("->")]).resolve()
            models[i]["name"] = arg[arg.index("->")+2:]
        else:
            models[i]["path"] = Path(arg).resolve()
            models[i]["name"] = extract_filename_from_path(arg, 1)

    for m in models:

        print("") # For cleaner printing
        model_instance = CNN()
        model_instance.load(m["path"])
        m["probabilities"], m["truth_labels"] = run_evaluation(model_instance, ds, "j","", "cuda" if torch.cuda.is_available() else "cpu", _no_output=True)

        m["accuracy"] = round(accuracy.calculate_accuracy(m["probabilities"], m["truth_labels"]), 5)
        m["balanced_accuracy"] = round(accuracy.calculate_balanced_accuracy(m["probabilities"], m["truth_labels"]), 5)
        m["mcc"] = round(mcc.calculate_matthews_correlation_coefficient(m["probabilities"], m["truth_labels"]), 5)
        m["roc-auc"] = round(roc_curve.calculate_roc_auc(m["probabilities"], m["truth_labels"]), 5)
        m["sensitivity"] = round(core_metrics.calculate_sensitivity(m["probabilities"], m["truth_labels"]), 5)
        m["specificity"] = round(core_metrics.calculate_specificity(m["probabilities"], m["truth_labels"]), 5)
        m["precision"] = round(core_metrics.calculate_precision(m["probabilities"], m["truth_labels"]), 5)
        m["pr-auc"] = round(pr_curve.calculate_pr_auc(m["probabilities"], m["truth_labels"]), 5)

    output_data = [["Model name", "Accuracy", "Balanced accuracy", "Mcc", "Sensitivity", "Specificity", "Precision", "ROC-AUC", "PR-AUC"]]


    for m in models:
        output_data.append([m["name"], m["accuracy"], m["balanced_accuracy"], m["mcc"], m["sensitivity"], m["specificity"],
                            m["precision"], m["roc-auc"], m["pr-auc"]])

    with open(str(output_directory) + "/comparative_evaluations.csv", "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(output_data)

    combined_probabilities = []
    truth_labels = []
    combined_model_names = []
    for m in models:
        combined_probabilities.append(m["probabilities"])
        truth_labels.append(m["truth_labels"])
        combined_model_names.append(m["name"])

    roc_curve.draw_multiple_model_roc_curve(combined_probabilities, truth_labels, combined_model_names, output_directory)

    print(f"\nModel comparison saved to {output_directory}")