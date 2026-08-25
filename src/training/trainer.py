import torch
import matplotlib.pyplot as plt
import pickle
from utilities.directory_handling import verify_directory
from training import data_handling

class Trainer:
    def __init__(self):
        self.data_handler = None
        self.model = None

    def train(self):
        pass

    def load_data(self):
        pass

    def load_pickled_model(self, filepath):
        if not str(filepath).endswith(".pkl"):
            print(f"Error when loading model. Function requires filepath ending in .pkl")
        with open(filepath, "rb") as file:
            self.model = pickle.load(file)

    def set_model(self, model):
        self.model = model

    def save(self, savefile_directory, savefile_name: str):
        verify_directory(savefile_directory, 1)

        if not savefile_name.endswith(".pkl"):
            savefile_name += ".pkl"

        with open(str(savefile_directory) + "/" + savefile_name, "wb") as file:
                pickle.dump(self, file)

        print(f"Trainer was successfully saved to {str(savefile_directory)}/{savefile_name}")

    @staticmethod
    def manual_observation(model, data_structure: data_handling.DataStructure):
        """
        Take given model and infer on a sample, showing the image, the correct label and the inferred label for
        manual checking
        :return:
        """

        if not data_structure.dataloaders:
            data_structure.load_dataset(True)

        for dataloader in data_structure.dataloaders:
            for batch in dataloader:
                batch_images = batch[0]
                batch_labels = batch[1]

                batch_images = batch_images.unsqueeze(1)
                predicted_labels = torch.round(torch.sigmoid(model(batch_images)))


                for i in range(len(batch_images)):

                    img = batch_images[i].squeeze()
                    label = batch_labels[i]
                    plt.imshow(img, cmap="gray")
                    plt.title(f"Predicted: {predicted_labels[i].item()} Real: {label}")
                    plt.show()

