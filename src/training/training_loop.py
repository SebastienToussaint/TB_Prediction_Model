from pympler.asizeof import asizeof, asized
import torch
from torch import nn
from training import data_handling
from training.evaluation_metrics import accuracy


def run_training_loop(model: nn.Module, data_structure: data_handling.DataStructure, number_of_epochs: int, loss_fn, optimizer, target_device: str = "cpu",
                      target_metric = None, target_metric_value: float|int = 1):

    print("Starting training loop")
    print(f"Training on {target_device}")
    model.to(target_device)
    cross_validation_test_index = 0

    for epoch in range(number_of_epochs):

        # Update which split testing will be done on
        cross_validation_test_index += 1
        if cross_validation_test_index == len(data_structure.dataloaders):
            cross_validation_test_index = 0

        print(f"Epoch {epoch+1}/{number_of_epochs}")
        total_train_loss = 0
        total_batches_processed = 0

        # Training run
        model.train()
        compiled_training_inferred_probabilities = [] # Stores all the probabilities from the training run in this epoch
        compiled_training_labels = [] # Truth labels for training run

        for idx, dataloader in enumerate(data_structure.dataloaders):
            if idx == cross_validation_test_index:
                print(f"Skipping dataloader {cross_validation_test_index} which is being used for testing this generation")
                continue

            print(f"Processing dataloader {idx}...")
            for batch, (images, labels) in enumerate(dataloader):
                images, labels = images.to(target_device), labels.to(target_device)

                # Infer labels
                images = images.unsqueeze(1) # TODO: Need function to check if dims are correct (for model type) and update them on the fly
                inferred_logits = model(images).squeeze()
                inferred_prediction_values = torch.sigmoid(inferred_logits)

                # Save inferred information
                compiled_training_inferred_probabilities += inferred_prediction_values.detach().cpu().tolist()
                compiled_training_labels += labels.detach().cpu().tolist()

                # Perform standard training loop operations
                batch_loss = loss_fn(inferred_logits, labels.float())
                total_train_loss += batch_loss
                total_batches_processed += 1

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        # Calculate training accuracies and avg loss
        train_accuracy = accuracy.calculate_accuracy(compiled_training_inferred_probabilities, compiled_training_labels)
        train_balanced_accuracy = accuracy.calculate_balanced_accuracy(compiled_training_inferred_probabilities, compiled_training_labels)


        # Test run
        total_test_loss = 0
        model.eval()

        compiled_testing_inferred_probabilities = []
        compiled_testing_labels = []

        with torch.inference_mode():
            print(f"Testing on dataloader {cross_validation_test_index}")
            for images, labels in data_structure.dataloaders[cross_validation_test_index]:
                images, labels = images.to(target_device), labels.to(target_device)
                images = images.unsqueeze(1) # TODO: Need function to check if dims are correct (for model type) and update them on the fly
                inferred_logits = model(images).squeeze()
                inferred_prediction_values = torch.sigmoid(inferred_logits) # Round these to get labels

                compiled_testing_inferred_probabilities += inferred_prediction_values.detach().cpu().tolist()
                compiled_testing_labels += labels.detach().cpu().tolist()

                total_test_loss += loss_fn(inferred_logits, labels.float())

        test_balanced_accuracy = accuracy.calculate_balanced_accuracy(compiled_testing_inferred_probabilities, compiled_testing_labels)


        # Keep inferred probabilities from testing in model
        model.store_probabilities(compiled_testing_inferred_probabilities, compiled_testing_labels)

        # Get average loss values
        average_train_loss = total_train_loss / total_batches_processed
        average_test_loss = total_test_loss / len(data_structure.dataloaders[cross_validation_test_index])


        print(f"Epoch {epoch} | Training loss: {average_train_loss} | Test Loss: {average_test_loss} | Accuracy: {train_accuracy} | Balanced Accuracy: {train_balanced_accuracy} "
              f"| Test Balanced Acc: {test_balanced_accuracy}")
