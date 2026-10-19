# Script to manually observe model behaviour
from training.trainer import Trainer
import sys
from shared_objects import models
from shared_objects.data_handling import DataStructure, CustomDataset

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error in manual_observation.py. Incorrect arguments provided.\n"
              "Usage: python -m *.trainer <path_to_model> <path_to_dataloaders>")
        quit()


    t = Trainer()

    model_0 = models.CNN()
    model_0.create_standard(1, 64, 1)
    # model_0.load(sys.argv[1])

    ds = DataStructure()
    ds.set_native_datapath(sys.argv[2])

    t.manual_observation(model_0, ds)
