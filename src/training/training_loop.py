import torch
from torch import nn

from training import data_handling
import trainer
from evaluation_metrics import *
from training.evaluation_metrics import accuracy
from model import CNN


def run_training_loop(model: nn.Module, data_structure: data_handling.DataStructure, number_of_epochs: int, loss_fn, optimizer, target_device: str = "cpu",
                      target_metric = None, target_metric_value: float|int = 1):

    print(target_device)
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

            images, labels = images.to(target_device), labels.to(target_device)

            # print(f"Processing batch {batch}...")
            images = images.unsqueeze(1) # TODO: Need function to check if dims are correct (for model type) and update them on the fly
            inferred_logits = model(images).squeeze()
            inferred_prediction_values = torch.sigmoid(inferred_logits)
            inferred_labels = torch.round(inferred_prediction_values)

            compiled_training_inferred_probabilities += inferred_prediction_values.detach().cpu().tolist()
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
                images, labels = images.to(target_device), labels.to(target_device)
                images = images.unsqueeze(1) # TODO: Need function to check if dims are correct (for model type) and update them on the fly
                inferred_logits = model(images).squeeze()
                inferred_prediction_values = torch.sigmoid(inferred_logits)
                inferred_labels = torch.round(inferred_prediction_values)

                compiled_testing_inferred_probabilities += inferred_prediction_values.detach().cpu().tolist()
                compiled_testing_inferred_labels += list(inferred_labels.float()) #TODO detach and put on cpu to save on mem.
                compiled_testing_labels += list(labels.float())

                total_test_loss += loss_fn(inferred_logits, labels.float())


        # print(calculate_accuracy(compiled_testing_inferred_labels, compiled_testing_labels))

        average_test_loss = total_test_loss/len(test_dataloader)

        print(f"Epoch {epoch} | Training loss: {average_train_loss} | Test Loss: {average_test_loss} | Accuracy: {train_accuracy} | Balanced Accuracy: {train_balanced_accuracy}")



class TestModel(nn.Module):
    def __init__(self, input_shape, hidden_nodes, output_shape):
        super().__init__()
        self.model_type = "linear"
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

model_benchmark = TestModel(784,2000, 1)
model_0 = CNN()
model_0.create_standard(1, 256, 1)
ds = data_handling.DataStructure()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)
loss_function = nn.BCEWithLogitsLoss()

ds.set_native_datapath("test_data/pickled_data/fashion_light_split_100.pkl")
ds.load_dataset(True)
# print("Running benchmark...")
# run_training_loop(model_benchmark, ds, 50, loss_function, optimizer)
print("Running training...")
train_mode = ("cuda" if torch.cuda.is_available() else "cpu")
run_training_loop(model_0, ds, 10, loss_function, optimizer, target_device=train_mode)

ds.shuffle_data()

trainer.Trainer.manual_observation(model_0, ds)