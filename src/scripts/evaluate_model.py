import sys
import torch
from shared_objects.models import CNN
from training.evaluation_run import run_evaluation
from shared_objects.data_handling import DataStructure, CustomDataset
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error in evaluate_model.py. Incorrect arguments provided.\n"
              "Usage: python -m *.evaluate_model <path_to_model> <path_to_dataloaders> <output_directory>")
        quit()

    model_path = sys.argv[1]
    dataloaders_path = sys.argv[2]
    save_directory = Path(sys.argv[3]).resolve()

    model = CNN()
    model.load(model_path)

    ds = DataStructure()
    ds.set_native_datapath(dataloaders_path)

    run_evaluation(model, ds, save_directory, "cuda" if torch.cuda.is_available() else "cpu")
