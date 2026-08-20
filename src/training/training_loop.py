import torch
from torch import nn

from training import data_handling
from training.data_handling import CustomDataset
import trainer
from evaluation_metrics import *
from training.evaluation_metrics import accuracy
from model import CNN


def run_training_loop(model: nn.Module, data_structure: data_handling.DataStructure, number_of_epochs: int, loss_fn, optimizer, target_device: str = "cpu",
                      target_metric = None, target_metric_value: float|int = 1):

    model.to(target_device)


    train_dataloader = data_structure.dataloaders[0]
    test_dataloader = data_structure.dataloaders[1]


    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch}/{number_of_epochs}")
        total_train_loss = 0

        # Training run
        model.train()
        compiled_training_inferred_probabilities = []
        compiled_training_inferred_labels = []
        compiled_training_labels = []
        for batch, (images, labels) in enumerate(train_dataloader):

            inferred_logits = model(images).squeeze()
            inferred_prediction_values = torch.sigmoid(inferred_logits)
            inferred_labels = torch.round(inferred_prediction_values)

            compiled_training_inferred_probabilities += list(inferred_prediction_values.float())
            compiled_training_inferred_labels += list(inferred_labels.float())
            compiled_training_labels += list(labels.float())

            batch_loss = loss_fn(inferred_logits, labels.float())
            total_train_loss += batch_loss

            optimizer.zero_grad()

            batch_loss.backward()

            optimizer.step()


        train_accuracy = accuracy.calculate_accuracy(compiled_training_inferred_labels, compiled_training_labels)
        train_balanced_accuracy = accuracy.calculate_balanced_accuracy(compiled_training_inferred_labels, compiled_training_labels)


        average_train_loss = total_train_loss/len(train_dataloader)


        # Test run
        total_test_loss = 0
        model.eval()

        compiled_testing_inferred_probabilities = []
        compiled_testing_inferred_labels = []
        compiled_testing_labels = []

        with torch.inference_mode():
            for images, labels in test_dataloader:
                inferred_logits = model(images).squeeze()
                inferred_prediction_values = torch.sigmoid(inferred_logits)
                inferred_labels = torch.round(inferred_prediction_values)

                compiled_testing_inferred_probabilities += list(inferred_prediction_values.float())
                compiled_testing_inferred_labels += list(inferred_labels.float())
                compiled_testing_labels += list(labels.float())

                total_test_loss += loss_fn(inferred_logits, labels.float())


        # print(calculate_accuracy(compiled_testing_inferred_labels, compiled_testing_labels))

        average_test_loss = total_test_loss/len(test_dataloader)

        print(f"Epoch {epoch} | Training loss: {average_train_loss} | Test Loss: {average_test_loss} | Accuracy: {train_accuracy} | Balanced Accuracy: {train_balanced_accuracy}")



class TestModel(nn.Module):
    def __init__(self, input_shape, hidden_nodes, output_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, output_shape)
        )

    def forward(self, x):
        return self.layers(x)

# model_0 = TestModel(10000,2000, 1)
model_0 = CNN()
model_0.create_standard(784, 64, 1)
ds = data_handling.DataStructure()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)
loss_function = nn.BCEWithLogitsLoss()

ds.set_native_datapath("test_data/pickled_data/fashion.pkl")
ds.load_dataset(True)
run_training_loop(model_0, ds, 10, loss_function, optimizer)

ds.shuffle_data()

trainer.Trainer.manual_observation(model_0, ds)