import torch
import matplotlib.pyplot as plt
import pickle
from utilities.directory_handling import verify_directory
from data_handling import DataStructure, CustomDataset # Import CustomDataset to prevent error with pickle
from training_loop import run_training_loop
import models


class Trainer:
    def __init__(self, model: torch.nn.Module|None=None, data_structure:DataStructure|None=None):
        self.data_structure = data_structure
        self.model = model

    def train(self, number_of_epochs, model: torch.nn.Module|None=None, data_structure:DataStructure|None=None,
              loss_function:str="BCE", optimizer:str="SGD", learning_rate:float|int=0.001, target_device:str="Any"):

        # Parameter Management
        if not self.model and not model:
            print("Error in trainer.train(): No Model loaded into trainer. Either set one via "
                  "trainer.set_model() or provide one.")
            quit()
        if not self.data_structure and not data_structure:
            print(
                "Error in trainer.train(): No Data Structure loaded into trainer. Either set one via "
                "trainer.set_data_structure() or provide one.")
            quit()
        if model and self.model:
            print("Training will be done on provided model over native trainer model.")
        elif self.model:
            print("Training will be done on native trainer model.")
            model = self.model
        if data_structure and self.data_structure:
            print("Training will be done with provided data handler over native trainer data structure.")
        elif self.data_structure:
            print("Training will be done on native trainer data structure.")
            data_structure = self.data_structure

        # Prevents warnings
        if model is None or data_structure is None:
            print("Error in trainer.train(), model or data_structure not set.")
            quit()

        loss_fn = self._get_loss_fn(loss_function)
        optim = self._get_optimizer(optimizer, model, learning_rate)
        device = self._get_target_device(target_device)

        run_training_loop(model, data_structure, number_of_epochs, loss_fn, optim, device)


    @staticmethod
    def _get_optimizer(name, model, lr):
        match name.upper():
            case "SGD":
                return torch.optim.SGD(params=model.parameters(), lr=lr)
            case "SGD-MOMENTUM":
                return torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
            case "ADAM":
                return torch.optim.Adam(params=model.parameters(), lr=lr)
            case "ADAMW":
                return torch.optim.AdamW(params=model.parameters(), lr=lr)
            case "ADAFACTOR":
                return torch.optim.Adafactor(params=model.parameters(), lr=lr)
            case "RMSPROP":
                return torch.optim.RMSprop(params=model.parameters(), lr=lr)
            case _:
                print(f"Error in trainer.train(), {name} is not a valid optimizer.\n"
                      f"Options: SGD, SGD-Momentum, Adam, AdamW, Adafactor, RMSprop.")
                quit()


    @staticmethod
    def _get_loss_fn(name):
        match name.upper():
            case "BCE":
                return torch.nn.BCEWithLogitsLoss()
            case "BCE-SMOOTHED":
                print("BCE-Smoothed not implemented yet") #TODO: Implement smoothed loss for transformer
                quit()
            case _:
                print(f"Error in trainer.train(), {name} is not a valid loss function.\n"
                      f"Options: BCE, BCE-SMOOTHED.")
                quit()

    @staticmethod
    def _get_target_device(name):
        match name.upper():
            case "ANY":
                return "cuda" if torch.cuda.is_available() else "cpu"
            case "CPU":
                return "cpu"
            case "cuda":
                return "cuda"
            case "gpu":
                return "gpu"
            case _:
                print(f"Error in trainer.train() {name} is not a valid target device.\n"
                      f"Options: Any (will use cuda/gpu if available, else cpu), cpu, cuda, gpu (same as cuda).")
                quit()

    def set_data_structure(self, data_structure: DataStructure):
        self.data_structure = data_structure

    def set_model(self, model):
        self.model = model

    def save(self, savefile_directory, savefile_name: str):
        verify_directory(savefile_directory, 1)

        if not savefile_name.endswith(".pkl"):
            savefile_name += ".pkl"

        with open(str(savefile_directory) + "/" + savefile_name, "wb") as file:
                pickle.dump(self, file)

        print(f"Trainer was successfully saved to {str(savefile_directory)}/{savefile_name}")

    def save_model(self):
        if self.model is None:
            print("No native model to save")
            return

        self.model.save()

    @staticmethod
    def manual_observation(model, data_structure: DataStructure):
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


model_0 = models.CNN()
model_0.create_standard(1, 128, 1)
ds = DataStructure()
ds.set_native_datapath("test_data/pickled_data/fashion_original_small.pkl")
ds.load_dataset(True)


t = Trainer()
t.train(3, model_0, ds, "BCE", "SGD", 0.001, "ANY")
