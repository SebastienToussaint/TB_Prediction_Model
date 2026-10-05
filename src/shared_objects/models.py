import pickle
import csv
from shared_objects.data_handling import CustomDataset
import torch.cuda
from pympler.asizeof import asizeof
from torch import nn
from utilities.directory_handling import verify_directory
from evaluation_metrics import accuracy, core_metrics, diagnostic_odds_ratio, f1_score, fowlkes_mallows_index
from evaluation_metrics import informedness, jaccard_index, markedness, mcc, prevalence_threshold
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = "cnn"
        self.layers = nn.Sequential(
        )

        self.train_prediction_probabilities = []
        self.train_truth_labels = []
        self.test_prediction_probabilities = []
        self.test_truth_labels = []
        self.save_directory = ""

        self.image_scores_training = {}
        self.image_scores_testing = {}

    def forward(self, x):
        return self.layers(x)

    def update_image_scores_training(self, correct_images, incorrect_images):
        for image in correct_images:
            if self.image_scores_training.get(image):
                self.image_scores_training[image] += 1
            else:
                self.image_scores_training[image] = 1

        for image in incorrect_images:
            if self.image_scores_training.get(image):
                self.image_scores_training[image] -= 1
            else:
                self.image_scores_training[image] = -1

    def update_image_scores_testing(self, correct_images, incorrect_images):
        for image in correct_images:
            if self.image_scores_testing.get(image):
                self.image_scores_testing[image] += 1
            else:
                self.image_scores_testing[image] = 1

        for image in incorrect_images:
            if self.image_scores_testing.get(image):
                self.image_scores_testing[image] -= 1
            else:
                self.image_scores_testing[image] = -1


    def store_probabilities(self, train_prediction_probabilities, train_truth_labels,
                            test_prediction_probabilities=None, test_truth_labels=None):
        #TODO: see if there's a way to do this directly onto disk without loading it into memory /
        # ,otherwise can always unload stuff from memory -> load this from disk -> add to it -> save to disk -> /
        # reload data that was in mem (like dataloaders).
        self.train_prediction_probabilities.append(train_prediction_probabilities)
        self.train_truth_labels.append(train_truth_labels)
        if test_prediction_probabilities:
            self.test_prediction_probabilities.append(test_prediction_probabilities)
            self.test_truth_labels.append(test_truth_labels)

    def load(self, path_to_model):
        if not str(path_to_model).endswith(".pkl"): #TODO: Uncomment but check which extension torch.save uses
            print(f"Error in CNN.load(). Provided path needs to be a .pkl file. (Provided: {path_to_model})")
            quit()

        verify_directory(path_to_model, 0)

        self.__dict__.update(
        torch.load(path_to_model, map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
                   weights_only=False).__dict__)

        print(f"Loaded model from {path_to_model}")
        self.save_directory = path_to_model

        return self

    def get_memory_space(self):
        return asizeof(self)/1000000


    def save(self, save_directory: str | Path = "", savefile_name: str = "model"):

        if not savefile_name.endswith(".pkl"):
            if "." in savefile_name:
                savefile_name = savefile_name[0:savefile_name.rindex(".")]
                print("Overwriting extension of save filename with .pkl")

            savefile_name += ".pkl"

        save_directory = str(Path(str(save_directory)).resolve())

        verify_directory(save_directory, 2)

        save_directory = str(Path(str(save_directory)).resolve())

        self.save_directory = save_directory + "/" + savefile_name

        torch.save(self, save_directory + "/" + savefile_name)

        print(f"Model saved at {save_directory + "/" + savefile_name}")

        return save_directory


    def print_layer_stack(self):
        for layer in self.layers:
            print(layer)

    def create_standard(self, colour_channels, hidden_units, output_shape):
        """
        Creates a standard CNN structure
        """
        self.layers = nn.Sequential(
            nn.Conv2d(colour_channels, hidden_units,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_units, hidden_units,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_units*2, hidden_units*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_units*2, output_shape)
        )
        print("Succesfully created standard model.")

    def calculate_batch_metrics(self, group_batches: int= 1):


        metrics = ["Accuracy", "Balanced Accuracy", "Recall", "Specificity", "Precision", "DOR", "F1", "Fowlkes Mallows Index",
                   "Informedness", "Jaccard Index", "Markedness", "MCC", "Prevalence Threshold"]

        training_evaluations = {}
        testing_evaluations = {}

        for metric in metrics:
            if not training_evaluations.get(metric):
                training_evaluations[metric] = []
            if not testing_evaluations.get(metric):
                testing_evaluations[metric] = []

        for epoch, epoch_probabilities in enumerate(self.train_prediction_probabilities):
            for batch_num, batch_probabilities in enumerate(epoch_probabilities):
                training_evaluations["Accuracy"].append(accuracy.calculate_accuracy(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Balanced Accuracy"].append(
                    accuracy.calculate_balanced_accuracy(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Recall"].append(
                    core_metrics.calculate_recall(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Specificity"].append(
                    core_metrics.calculate_specificity(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Precision"].append(
                    core_metrics.calculate_precision(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["DOR"].append(
                    diagnostic_odds_ratio.calculate_diagnostic_odds_ratio(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["F1"].append(
                    f1_score.calculate_f1_score(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Fowlkes Mallows Index"].append(
                    fowlkes_mallows_index.calculate_fowlkes_mallows_index(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Informedness"].append(
                    informedness.calculate_informedness(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Jaccard Index"].append(
                    jaccard_index.calculate_jaccard_index(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Markedness"].append(
                    markedness.calculate_markedness(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["MCC"].append(
                    mcc.calculate_matthews_correlation_coefficient(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                training_evaluations["Prevalence Threshold"].append(
                    prevalence_threshold.calculate_prevalence_threshold(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

        for epoch, epoch_probabilities in enumerate(self.test_prediction_probabilities):
            for batch_num, batch_probabilities in enumerate(epoch_probabilities):
                testing_evaluations["Accuracy"].append(accuracy.calculate_accuracy(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Balanced Accuracy"].append(
                    accuracy.calculate_balanced_accuracy(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Recall"].append(
                    core_metrics.calculate_recall(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Specificity"].append(
                    core_metrics.calculate_specificity(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Precision"].append(
                    core_metrics.calculate_precision(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["DOR"].append(
                    diagnostic_odds_ratio.calculate_diagnostic_odds_ratio(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["F1"].append(
                    f1_score.calculate_f1_score(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Fowlkes Mallows Index"].append(
                    fowlkes_mallows_index.calculate_fowlkes_mallows_index(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Informedness"].append(
                    informedness.calculate_informedness(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Jaccard Index"].append(
                    jaccard_index.calculate_jaccard_index(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Markedness"].append(
                    markedness.calculate_markedness(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["MCC"].append(
                    mcc.calculate_matthews_correlation_coefficient(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

                testing_evaluations["Prevalence Threshold"].append(
                    prevalence_threshold.calculate_prevalence_threshold(batch_probabilities, self.test_truth_labels[epoch][batch_num]))

        return training_evaluations, testing_evaluations

    def draw_eval_graphs(self, save_directory:str|Path, window_size: int = None,
                         group_batches: int = 1):



        metrics = ["Accuracy", "Balanced Accuracy", "Recall", "Specificity", "Precision", "DOR", "F1",
                   "Fowlkes Mallows Index",
                   "Informedness", "Jaccard Index", "Markedness", "MCC", "Prevalence Threshold"]



        training_evaluations, testing_evaluations = self.calculate_batch_metrics(group_batches)

        if not len(training_evaluations["Accuracy"]) > 0:
            print("Error when drawing eval graphs. No training data found in model. Set save_metrics=True in training template.")
            return


        verify_directory(save_directory, 2)

        if not window_size:
            window = max(1, int(len(training_evaluations["Accuracy"])/20))
        else:
            window = window_size

        for metric in metrics:
            plt.plot(training_evaluations[metric], alpha=0.2, label = "Raw")
            plt.plot(range(window - 1, len(training_evaluations[metric])),
                     np.convolve(training_evaluations[metric], np.ones(window) / window, mode="valid"),
                     linewidth=2,
                     label="Smoothed")

            # plt.title(f"{metric} change over batches")
            plt.xlabel("Batch Number")
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(str(Path(str(save_directory)).resolve()) + "/training_" + metric.replace(" ", "_") + ".svg", format="svg")
            plt.close()



        if len(testing_evaluations["Accuracy"]) > 0:
            if not window_size:
                window = max(1, int(len(testing_evaluations["Accuracy"]) / 20))
            else:
                window = window_size

            for metric in metrics:
                plt.plot(testing_evaluations[metric], alpha=0.2, label="Raw")
                plt.plot(range(window - 1, len(testing_evaluations[metric])),
                         np.convolve(testing_evaluations[metric], np.ones(window) / window, mode="valid"),
                         linewidth=2,
                         label="Smoothed")

                # plt.title(f"{metric} change over batches")
                plt.xlabel("Batch number")
                plt.ylabel(metric)
                plt.legend()
                plt.savefig(str(Path(str(save_directory)).resolve()) + "/testing_" + metric.replace(" ", "_") + ".svg", format="svg")
                plt.close()


        print(f"Eval graphs successfully created and saved at {str(Path(str(save_directory)).resolve())}")

    def output_image_scores(self, output_directory:str|Path):

        verify_directory(output_directory, 2)

        data = [["Training Data Path", "Score"]]

        for key in self.image_scores_training:
            data.append([key, self.image_scores_training[key]])

        data.append(["", ""])
        data.append(["Testing Data Path", "Score"])

        for key in self.image_scores_testing:
            data.append([key, self.image_scores_testing[key]])

        with open(output_directory + "/image_scores.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

        print(f"Image scores successfully saved to {str(Path(output_directory).resolve())}")





if __name__ == "__main__":
    m = CNN()
    m.load("E:\Project\Savefiles\Models\ShenTBX_HealthyVsTB\model_BS75.pkl")
    m.draw_eval_graphs("E:\Project\Savefiles\Models\ShenTBX_HealthyVsTB/Metrics")
    # m.output_image_scores("/home/sebastien/Project/SaveFiles/Models/SickVsTB/")
