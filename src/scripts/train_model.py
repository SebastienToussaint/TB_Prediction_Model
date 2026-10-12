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


    interval_save_path = str(model_save_directory)+"/intervals/"
    if config["model_save_interval"] > 0:
        verify_directory(interval_save_path, 2)

    model = models.CNN()

    if config["model_path"] in ["", " "]:
        model.create_standard(1, 64, 1)
        # print("SIMPLE MODEL")
        # model.create_simple()
    else:
        model.load(config["model_path"])

    ds = DataStructure()
    ds.set_native_datapath(config["dataloaders_path"])

    t.train(number_of_epochs=config["num_epochs"], model=model,data_structure=ds,
            augments_per_image=config["augments_per_image"],
            augmentation_ranges=config["augmentation_ranges"], loss_function=config["loss_function"],
            optimizer=config["optimizer"], learning_rate=config["learning_rate"], target_device=config["training_device"],
            number_of_evaluation_partitions=config["evaluation_partitions"],
            model_save_interval=config["model_save_interval"], interval_save_path=interval_save_path,
            save_metrics=config["save_metrics"],
            metrics_to_log=config["metrics_to_log"])

    model.save(model_save_directory, config["model_save_name"])
