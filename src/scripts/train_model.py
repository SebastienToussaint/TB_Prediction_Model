# Script to train a model
from training.trainer import Trainer
import sys
from utilities.directory_handling import verify_directory
from shared_objects import models
from shared_objects.data_handling import DataStructure, CustomDataset
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error in train_model.py. Incorrect arguments provided.\n"
              "Usage: python -m *.train_model <path_to_config_file>")
        quit()


    t = Trainer()
    config = t.decode_config_file(sys.argv[1])

    # Ensure save directory is good before starting training
    model_save_directory = Path(config["model_save_directory"]).resolve()
    verify_directory(model_save_directory, 2)

    model = models.CNN()

    if config["model_path"] in ["", " "]:
        model.create_standard(1, 64, 1)
    else:
        model.load(config["model_path"])

    ds = DataStructure()
    ds.set_native_datapath(config["dataloaders_path"])

    t.train(config["num_epochs"], model, ds, config["loss_function"], config["optimizer"], config["learning_rate"],
            config["training_device"], config["evaluation_partitions"], config["save_metrics"], config["metrics_to_log"])

    model.save(model_save_directory, config["model_save_name"])
