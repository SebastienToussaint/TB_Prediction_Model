import sys
import torch
import matplotlib.pyplot as plt
from shared_objects.data_handling import DataStructure, CustomDataset  # Import CustomDataset to prevent error with pickle
from training.training_loop import run_training_loop
from shared_objects import models
from training.evaluation_run import run_evaluation
from pathlib import Path
from utilities.directory_handling import verify_directory

class Trainer:
    def __init__(self):
        pass

    def train(self, number_of_epochs, model: torch.nn.Module, data_structure:DataStructure,
              augments_per_image: int=0, augmentation_ranges=((-0.1, 0.1),(-0.1, 0.1), (-15, 15)),
              loss_function:str="BCE", optimizer:str="SGD", learning_rate:float|int=0.001, target_device:str="Any",
              number_of_evaluation_partitions:int =0, save_metrics=True, metrics_to_log=None):

        # Parameter Management
        if not model:
            print("Error in trainer.train(): model is None")
            quit()
        if not data_structure:
            print(
                "Error in trainer.train(): data_structure is None")
            quit()

        loss_fn = self._get_loss_fn(loss_function)
        optim = self._get_optimizer(optimizer, model, learning_rate)
        device = self._get_target_device(target_device)

        run_training_loop(model=model, data_structure=data_structure, augments_per_image=augments_per_image,
                          augment_ranges=augmentation_ranges, number_of_epochs=number_of_epochs, loss_fn=loss_fn,
                          optimizer=optim, target_device=device, num_evaluation_partitions=number_of_evaluation_partitions,
                          save_probabilities_to_model=save_metrics, metrics_to_log=metrics_to_log)

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
                return "cuda"
            case _:
                print(f"Error in trainer.train() {name} is not a valid target device.\n"
                      f"Options: Any (will use cuda/gpu if available, else cpu), cpu, cuda, gpu (same as cuda).")
                quit()

    def set_data_structure(self, data_structure: DataStructure):
        self.data_structure = data_structure

    def set_model(self, model):
        self.model = model

    @staticmethod
    def manual_observation(model, data_structure: DataStructure):
        """
        Take given model and infer on a sample, showing the image, the correct label and the inferred label for
        manual checking
        :return:
        """

        model.to("cpu")

        num_partitions = data_structure.get_num_partitions()

        model.eval()

        with torch.inference_mode():
            dataloader = None
            for i in range(num_partitions):
                dataloader = data_structure.get_next_dataloader()

                for batch in dataloader:
                    batch_images = batch[0]
                    batch_labels = batch[1]
                    batch_filepaths = batch[2]

                    batch_images = batch_images.to("cpu")
                    batch_labels = batch_labels.to("cpu")

                    batch_images = batch_images.unsqueeze(1)

                    predicted_labels = torch.round(torch.sigmoid(model(batch_images)))


                    for j in range(len(batch_images)):

                        img = batch_images[j].squeeze()

                        print(img.dtype)
                        label = batch_labels[j]
                        plt.imshow(img, cmap="gray")
                        plt.title(f"Predicted: {predicted_labels[j].item()} Real: {label}\n"
                                  f"Filepath: {batch_filepaths[j]}")
                        plt.show()
                        plt.close()


    @staticmethod
    def evaluate_model(model, data_structure: DataStructure, save_directory):
        run_evaluation(model, data_structure, save_directory, "cpu")

    @staticmethod
    def decode_config_file(path_to_template):
        """
        :return: Dictionary containing decoded config info
        """

        decoded_dict = {}

        with open(path_to_template, "r") as file:
            for line in file:
                # Remove comments
                if "#" in line:
                    line = line[0:line.index("#")]

                # Remove leading and trailing spaces, skip empty line
                line = line.strip()
                if line in [""]:
                    continue

                # Decode each line
                match line[0:line.index("=")].upper():
                    case "MODEL_PATH":
                        model_path = line[line.index("=")+1:]
                        model_path = model_path.replace("'", "").replace('"', "")
                        decoded_dict["model_path"] = model_path
                    case "MODEL_SAVE_DIRECTORY":
                        model_save_directory = line[line.index("=")+1:]
                        model_save_directory = model_save_directory.replace("'", "").replace('"', "")
                        decoded_dict["model_save_directory"] = model_save_directory
                    case "MODEL_SAVE_INTERVAL":
                        model_save_interval = line[line.index("=")+1:]
                        try:
                            model_save_interval = int(model_save_interval)
                            if model_save_interval<0:
                                raise ValueError
                        except ValueError:
                            print("Error while decoding config file. model_save_interval needs to be a positive integer or 0")
                            quit()

                        decoded_dict["model_save_interval"] = model_save_interval
                    case "MAKE_CHARTS":
                        make_charts = line[line.index("=")+1:]
                        if make_charts.upper() == "TRUE":
                            make_charts = True
                        elif make_charts.upper() == "False":
                            make_charts = False
                        else:
                            print("Error while decoding config file. make_charts needs to be True or False")
                            quit()
                        decoded_dict["make_charts"] = make_charts
                    case "DATALOADERS_PATH":
                        dataloaders_path = line[line.index("=")+1:]
                        dataloaders_path = dataloaders_path.replace("'", "").replace('"', "")
                        decoded_dict["dataloaders_path"] = dataloaders_path

                    case "NUM_EPOCHS":
                        num_epochs = line[line.index("=")+1:]
                        try:
                            num_epochs = int(num_epochs)
                            if num_epochs<1:
                                raise ValueError
                        except ValueError:
                            print("Error while decoding config file. num_epochs needs to be a positive integer")
                            quit()

                        decoded_dict["num_epochs"] = num_epochs
                    case "LOSS_FUNCTION":
                        loss_function = line[line.index("=")+1:]
                        decoded_dict["loss_function"] = loss_function

                    case "OPTIMIZER":
                        optimizer = line[line.index("=")+1:]
                        decoded_dict["optimizer"] = optimizer
                    case "LEARNING_RATE":
                        learning_rate = line[line.index("=")+1:]
                        try:
                            learning_rate = float(learning_rate)
                            if learning_rate<0:
                                raise ValueError
                        except ValueError:
                            print("Error while decoding config file. learning_rate needs to be a positive real number")
                            quit()
                        decoded_dict["learning_rate"] = learning_rate
                    case "TRAINING_DEVICE":
                        training_device = line[line.index("=")+1:]
                        decoded_dict["training_device"] = training_device
                    case "EVALUATION_PARTITIONS":
                        evaluation_partitions = line[line.index("=")+1:]
                        try:
                            evaluation_partitions = int(evaluation_partitions)
                            if evaluation_partitions<0:
                                raise ValueError
                        except ValueError:
                            print("Error while decoding config file. evaluation_partitions needs to be a positive integer or 0")
                            quit()
                        decoded_dict["evaluation_partitions"] = evaluation_partitions
                    case "SAVE_METRICS":
                        save_metrics = line[line.index("=")+1:]
                        if save_metrics.upper() == "TRUE":
                            save_metrics = True
                        elif save_metrics.upper() == "False":
                            save_metrics = False
                        else:
                            print("Error while decoding config file. save_metrics needs to be True or False")
                            quit()

                        decoded_dict["save_metrics"] = save_metrics

                    case "METRICS_TO_LOG":
                        metrics_to_log = line[line.index("=")+1:]
                        decoded_dict["metrics_to_log"] = metrics_to_log.split(",")

                    case "MODEL_SAVE_NAME":
                        model_save_name = line[line.index("=")+1:]
                        decoded_dict["model_save_name"] = model_save_name

                    case "AUGMENTS_PER_IMAGE":
                        augments_per_image = line[line.index("=")+1:]
                        try:
                            augments_per_image = int(augments_per_image)
                            if augments_per_image < 0:
                                raise ValueError
                        except ValueError:
                            print(
                                "Error while decoding config file. augments_per_image needs to be a positive integer or 0")
                            quit()

                        decoded_dict["augments_per_image"] = augments_per_image

                    case "AUGMENTATION_PERCENTAGE_RANGES":
                        line = line[line.index("=")+1:]
                        line = line.replace("[", "")
                        line = line.replace("]", "")
                        line = line.split(",")

                        if not len(line) == 6:
                            print("ERROR WITH ARRAY")
                            quit()

                        try:
                            aug_ranges = [[float(line[0]), float(line[1])],
                                          [float(line[2]), float(line[3])],
                                          [float(line[4]), float(line[5])]]
                        except ValueError:
                            print(
                                "Error while decoding config file. augmentation_ranges "
                                "needs to be in form [[x1,x2],[y1,y2],[r1,r2]].\n"
                                "Where x[0->1], y[0->1] and r are real values")
                            quit()

                        decoded_dict["augmentation_ranges"] = aug_ranges


                    case _:
                        print(f"Error while decoding config file. {line} is not valid.")
                        quit()

        return decoded_dict

if __name__ == "__main__":
    pass














