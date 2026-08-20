import pickle
import torch
from torch import nn
from utilities.directory_handling import verify_directory
from pathlib import Path

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )
        self.evaluations = {}

    def load(self, path_to_model):
        if not str(path_to_model).endswith(".pkl"):
            print(f"Error in CNN.load(). Provided path needs to be a .pkl file. (Provided: {path_to_model})")
            quit()

        verify_directory(path_to_model, 0)

        with open(path_to_model, "rb") as file:
            loaded_data = pickle.load(file)
            self.layers = loaded_data[0]
            self.evaluations = loaded_data[1]


    def save(self, save_directory: str | Path, savefile_name: str):
        verify_directory(save_directory, 2)
        if not savefile_name.endswith(".pkl"):
            if "." in savefile_name:
                savefile_name = savefile_name[0:savefile_name.rindex(".")]
                print("Overwriting extension of save filename with .pkl")

            savefile_name += ".pkl"


        save_data = [self.layers, self.evaluations]

        with open(str(save_directory) + "/" + savefile_name, "wb") as savefile:
            pickle.dump(save_data, savefile)

        print(f"Compressed datafile saved at {str(save_directory) + "/" + savefile_name}")

        return str(save_directory) + "/" + savefile_name


    def print_layer_stack(self):
        for layer in self.layers:
            print(layer)

    def create_standard(self, input_shape, hidden_units, output_shape):
        """
        Creates a standard CNN structure
        :return:
        """
        self.layers = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units,3,1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units,3,1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(hidden_units*7*7, output_shape)
        )


    def forward(self, x):
        return self.layers(x)

