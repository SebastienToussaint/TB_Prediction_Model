import pickle
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
        self.training_batch_evaluations = {}
        self.testing_batch_evaluations = {}
        self.train_prediction_probabilities = []
        self.train_truth_labels = []
        self.test_prediction_probabilities = []
        self.test_truth_labels = []
        self.save_directory = ""

        # Whenever adding new parameters, need to add save file

    def forward(self, x):
        return self.layers(x)

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

        # print("Saved probabilities")
        # print(f"Size of probs in memory: {(asizeof(self.train_prediction_probabilities, self.train_truth_labels,
        #                                            test_prediction_probabilities, test_truth_labels))/1000000} MB")



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
        self.save_directory = path_to_model

        return self


    def save(self, save_directory: str | Path, savefile_name: str):

        if not savefile_name.endswith(".pkl"):
            if "." in savefile_name:
                savefile_name = savefile_name[0:savefile_name.rindex(".")]
                print("Overwriting extension of save filename with .pkl")

            savefile_name += ".pkl"

        save_directory = str(Path(str(save_directory)).resolve())

        verify_directory(save_directory, 2)

        save_directory = str(Path(str(save_directory)).resolve())


        # save_data = [self.model_type, self.layers, self.evaluations, self.prediction_probabilities, self.truth_labels]

        with open(save_directory + "/" + savefile_name, "wb") as savefile:
            pickle.dump(self, savefile)

        print(f"Compressed datafile saved at {save_directory + "/" + savefile_name}")

        return save_directory


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

            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden_units*2, hidden_units*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_units*2, output_shape)
        )

    def calculate_batch_metrics(self, group_batches: int= 1):


        # num_epochs = len(self.prediction_probabilities)
        # num_batches_per_epoch = len(self.prediction_probabilities[0])
        # batch_length = len(self.prediction_probabilities[0][0])

        # print(f"{num_epochs} epochs with {num_batches_per_epoch} batches, each containing {batch_length} images, "
        #       f"for a total of {num_epochs*num_batches_per_epoch} batches and {num_epochs*num_batches_per_epoch*batch_length} images")

        metrics = ["accuracy", "balanced_accuracy", "recall", "specificity", "precision", "dor", "f1", "fowlkes_mallows_index",
                   "informedness", "jaccard_index", "markedness", "mcc", "prevalence_threshold"]

        for metric in metrics:
            if not self.training_batch_evaluations.get(metric):
                self.training_batch_evaluations[metric] = []

        for epoch, epoch_probabilities in enumerate(self.train_prediction_probabilities):
            for batch_num, batch_probabilities in enumerate(epoch_probabilities):
                self.training_batch_evaluations["accuracy"].append(accuracy.calculate_accuracy(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["balanced_accuracy"].append(
                    accuracy.calculate_balanced_accuracy(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["recall"].append(
                    core_metrics.calculate_recall(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["specificity"].append(
                    core_metrics.calculate_specificity(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["precision"].append(
                    core_metrics.calculate_precision(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["dor"].append(
                    diagnostic_odds_ratio.calculate_diagnostic_odds_ratio(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["f1"].append(
                    f1_score.calculate_f1_score(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["fowlkes_mallows_index"].append(
                    fowlkes_mallows_index.calculate_fowlkes_mallows_index(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["informedness"].append(
                    informedness.calculate_informedness(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["jaccard_index"].append(
                    jaccard_index.calculate_jaccard_index(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["markedness"].append(
                    markedness.calculate_markedness(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["mcc"].append(
                    mcc.calculate_matthews_correlation_coefficient(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

                self.training_batch_evaluations["prevalence_threshold"].append(
                    prevalence_threshold.calculate_prevalence_threshold(batch_probabilities, self.train_truth_labels[epoch][batch_num]))

        return self.training_batch_evaluations

    def draw_eval_graphs(self, save_directory:str|Path = "", window_size: int = None):
        metrics = ["accuracy", "balanced_accuracy", "recall", "specificity", "precision", "dor", "f1",
                   "fowlkes_mallows_index",
                   "informedness", "jaccard_index", "markedness", "mcc", "prevalence_threshold"]

        verify_directory(save_directory, 2)

        if not window_size:
            window = max(3, int(len(self.training_batch_evaluations["accuracy"])/20))
        else:
            window = window_size

        for metric in metrics:
            plt.plot(self.training_batch_evaluations[metric], alpha=0.2, label = "Raw")
            plt.plot(range(window - 1, len(self.training_batch_evaluations[metric])),
                     np.convolve(self.training_batch_evaluations[metric], np.ones(window) / window, mode="valid"),
                     linewidth=2,
                     label="Smoothed")

            plt.title(f"{metric} change over batches")
            plt.xlabel("Batch")
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(str(Path(str(save_directory)).resolve()) + "/" + metric + ".svg", format="svg")
            plt.close()

        print(f"Eval graphs successfully created and saved at {str(Path(str(save_directory)).resolve())}")











if __name__ == "__main__":
    m = CNN()
    m.load("E:/Project\data\PIPELINETEST\Models\Model0\model0.pkl")
    m.calculate_batch_metrics()
    m.draw_eval_graphs("E:/Project/data/PIPELINETEST/Models/Model0/Metrics")
