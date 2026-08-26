import pickle
import torch
from torch import nn
from utilities.directory_handling import verify_directory
from pathlib import Path

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = "cnn"
        self.layers = nn.Sequential(
        )
        self.evaluations = {}
        self.prediction_probabilities = [] #TODO: Look into how not to clear these everytime model is initialized
        self.truth_labels = []

        # Whenever adding new parameters, need to add save file

    def store_probabilities(self, prediction_probabilities, truth_labels):
        #TODO: see if there's a way to do this directly onto disk without loading it into memory /
        # ,otherwise can always unload stuff from memory -> load this from disk -> add to it -> save to disk -> /
        # reload data that was in mem (like dataloaders).
        self.prediction_probabilities.append(prediction_probabilities)
        self.truth_labels.append(truth_labels)



    def load(self, path_to_model):
        if not str(path_to_model).endswith(".pkl"):
            print(f"Error in CNN.load(). Provided path needs to be a .pkl file. (Provided: {path_to_model})")
            quit()

        verify_directory(path_to_model, 0)


        with open(path_to_model, "rb") as file:
            self.__dict__.update(pickle.load(file).__dict__)
        #     loaded_data = pickle.load(file)
        #     self.model_type = loaded_data[0].copy()
        #     self.layers = loaded_data[1].copy()
        #     self.evaluations = loaded_data[2].copy()
        #     self.prediction_probabilities = loaded_data[3].copy()
        #     self.truth_labels = loaded_data[4].copy()
        #
        # del loaded_data

        print(f"Loaded model from {path_to_model}")

        return self


    def save(self, save_directory: str | Path, savefile_name: str):
        verify_directory(save_directory, 2)
        if not savefile_name.endswith(".pkl"):
            if "." in savefile_name:
                savefile_name = savefile_name[0:savefile_name.rindex(".")]
                print("Overwriting extension of save filename with .pkl")

            savefile_name += ".pkl"


        # save_data = [self.model_type, self.layers, self.evaluations, self.prediction_probabilities, self.truth_labels]

        with open(str(save_directory) + "/" + savefile_name, "wb") as savefile:
            pickle.dump(self, savefile)

        print(f"Compressed datafile saved at {str(save_directory) + "/" + savefile_name}")

        return str(save_directory) + "/" + savefile_name


    def print_layer_stack(self):
        for layer in self.layers:
            print(layer)

    def create_standard(self, colour_channels, hidden_units, output_shape):
        """
        Creates a standard CNN structure
        """
        self.layers = nn.Sequential(
            nn.Conv2d(colour_channels, hidden_units,3,1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units,3,1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_units, output_shape)
        )

    def forward(self, x):
        return self.layers(x)

