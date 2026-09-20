import time

from pympler.asizeof import asizeof, asized
import torch
from torch import nn
from shared_objects import data_handling
from evaluation_metrics import accuracy, mcc, f1_score, core_metrics
from utilities.array_handling import flatten_list
from evaluation_metrics import metric_counts

def log_metrics(epoch: int, metrics_to_log, train_loss, train_probs, train_labels,
                test_loss = None, test_probs = None, test_labels = None):

    round_values = 4
    log_string = f"Epoch {epoch + 1} "
    if test_loss:
        log_string += (f"| Loss: {train_loss} "
                       f"(Test: {test_loss}) ")
    else:
        log_string += f"| Loss: {train_loss} "

    if metrics_to_log:
        # Build log string based on metrics to log
        for metric in metrics_to_log:
            match metric.upper():
                case "ACC":
                    log_string += f"| Accuracy: {round(accuracy.calculate_accuracy(train_probs, train_labels), round_values)} "
                    if test_labels:
                        log_string += f"(Test: {round(accuracy.calculate_accuracy(test_probs, test_labels), round_values)}) "
                case "BACC":
                    log_string += f"| Balanced Accuracy: {round(accuracy.calculate_balanced_accuracy(train_probs, train_labels), round_values)} "
                    if test_labels:
                        log_string += f"(Test: {round(accuracy.calculate_balanced_accuracy(test_probs, test_labels), round_values)}) "
                case "MCC":
                    log_string += f"| MCC: {round(mcc.calculate_matthews_correlation_coefficient(train_probs, train_labels), round_values)} "
                    if test_labels:
                        log_string += f"(Test: {round(mcc.calculate_matthews_correlation_coefficient(test_probs, test_labels), round_values)}) "
                case "F1":
                    log_string += f"| F1-Score: {round(f1_score.calculate_f1_score(train_probs, train_labels), round_values)} "
                    if test_labels:
                        log_string += f"(Test: {round(f1_score.calculate_f1_score(test_probs, test_labels), round_values)}) "
                case "RECALL":
                    log_string += f"| Recall: {round(core_metrics.calculate_recall(train_probs, train_labels), round_values)} "
                    if test_labels:
                        log_string += f"(Test: {round(core_metrics.calculate_recall(test_probs, test_labels), round_values)}) "
                case "SPECIFICITY":
                    log_string += f"| Specificity: {round(core_metrics.calculate_specificity(train_probs, train_labels), round_values)} "
                    if test_labels:
                        log_string += f"(Test: {round(core_metrics.calculate_specificity(test_probs, test_labels), round_values)}) "
                case "Precision":
                    log_string += f"| Precision: {round(core_metrics.calculate_precision(train_probs, train_labels), round_values)} "
                    if test_labels:
                        log_string += f"(Test: {round(core_metrics.calculate_precision(test_probs, test_labels), round_values)}) "
                case _:
                    if epoch == 0:
                        print(f"Could not find {metric} as metric to log.")
                        print("Options are: ACC, BACC, MCC, F1, Recall, Specificity, Precision.")

    print(log_string)


def update_model_image_scores(model, inferred_probs, truth_labels, image_paths, training: bool):
    correct_indices = []
    correct_indices.extend(metric_counts.get_true_positives(inferred_probs, truth_labels)[1])
    correct_indices.extend(metric_counts.get_true_negatives(inferred_probs, truth_labels)[1])

    correct_paths = []
    incorrect_paths = []
    for i, path in enumerate(image_paths):
        if i in correct_indices:
            correct_paths.append(path)
        else:
            incorrect_paths.append(path)

    if training:
        model.update_image_scores_training(correct_paths, incorrect_paths)
    else:
        model.update_image_scores_testing(correct_paths, incorrect_paths)


def run_training_loop(model: nn.Module, data_structure: data_handling.DataStructure, number_of_epochs: int, loss_fn,
                      optimizer, target_device: str = "cpu", num_evaluation_partitions = 0,
                      save_probabilities_to_model = True, metrics_to_log:list|None = None):

    # Log training info
    print("Starting training loop...\n")
    print(f"Training with following settings:\n"
          f" -Epochs: {number_of_epochs}\n"
          f" -Device: {target_device}\n"
          f" -Loss function: {loss_fn}\n"
          f" -Optimizer: {optimizer.__class__.__name__}\n"
          f" -Number of Partitions: {data_structure.get_num_partitions()}\n"
          f" -Number of partitions reserved for evaluation: {num_evaluation_partitions}\n"
          f" -Save inferred probabilities to model: {save_probabilities_to_model}\n")


    model.to(target_device)

    # Get and verify number of partitions
    num_training_partitions = data_structure.get_num_partitions()
    if num_evaluation_partitions > 0:
        num_training_partitions -= num_evaluation_partitions
    if num_training_partitions <= 0:
        print(f"Error while training, not enough partitions ({num_training_partitions}) to reserve one for evaluation.\n"
              f"Please create more partitions or set reserve_evaluation_partition to False")
        quit()

    # Training loop
    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch+1}/{number_of_epochs}")

        # TRAINING RUN ============================================================================

        model.train()

        total_train_loss = 0
        train_batches_processed = 0
        training_probs = [] # Stores all the probabilities from the training run in this epoch
        training_labels = [] # Truth labels for training run
        training_filepaths = []

        for i in range(num_training_partitions):
            time_1 = time.time()

            # Get the next chunk of data
            train_dataloader = data_structure.get_next_dataloader()
            train_dataloader = data_structure.augment_data(train_dataloader, [-50, 50], [-50, 50], [-15, 15])
            print(f"Took {time.time()-time_1} for data loading and augmentation.")

            for batch, (images, labels, filepaths) in enumerate(train_dataloader):
                print(f"Processing {len(images)} images in batch")
                images, labels = images.to(target_device), labels.to(target_device)

                # Infer labels
                images = images.unsqueeze(1) # TODO: Need function to check if dims are correct (for model type) and update them on the fly
                inferred_logits = model(images).squeeze()
                inferred_prediction_values = torch.sigmoid(inferred_logits)

                # Save inferred probabilities and truth labels
                training_probs.append(inferred_prediction_values.detach().cpu().tolist())
                training_labels.append(labels.detach().cpu().tolist())
                training_filepaths.append(filepaths)

                # Perform standard training loop operations
                batch_loss = loss_fn(inferred_logits, labels.float())
                total_train_loss += batch_loss.item()
                train_batches_processed += 1

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        # =========================================================================================

        # TESTING RUN =============================================================================

        total_test_loss = None
        average_test_loss = None
        test_batches_processed = 0
        testing_probs = []
        testing_labels = []
        testing_filepaths = []

        if num_evaluation_partitions > 0:
            total_test_loss = 0
            model.eval()

            for i in range(num_evaluation_partitions):
                test_dataloader = data_structure.get_next_dataloader()

                # Run evaluation on model
                with torch.inference_mode():
                    for images, labels, filepaths in test_dataloader:
                        images, labels = images.to(target_device), labels.to(target_device)

                        # Infer labels
                        images = images.unsqueeze(1) # TODO: Need function to check if dims are correct (for model type) and update them on the fly
                        inferred_logits = model(images).squeeze()
                        inferred_prediction_values = torch.sigmoid(inferred_logits) # Round these to get labels

                        # Save inferred probabilities and truth labels
                        testing_probs.append(inferred_prediction_values.detach().cpu().tolist())
                        testing_labels.append(labels.detach().cpu().tolist())
                        testing_filepaths.append(filepaths)

                        total_test_loss += loss_fn(inferred_logits, labels.float()).item()
                        test_batches_processed += 1

        # =========================================================================================

        # Get average loss values
        average_train_loss = total_train_loss / train_batches_processed
        if total_test_loss:
            average_test_loss = total_test_loss/test_batches_processed

        # Save inferred probabilities and labels to model
        if save_probabilities_to_model:
            if num_evaluation_partitions > 0:
                update_model_image_scores(model, flatten_list(training_probs), flatten_list(training_labels),
                                          flatten_list(training_filepaths), True)
                update_model_image_scores(model, flatten_list(testing_probs), flatten_list(testing_labels),
                                          flatten_list(testing_filepaths), False)

                model.store_probabilities(training_probs, training_labels, testing_probs, testing_labels)
            else:
                update_model_image_scores(model, flatten_list(training_probs), flatten_list(training_labels),
                                          flatten_list(training_filepaths), True)

                model.store_probabilities(training_probs, training_labels)


        # Log all requested metrics as well as training and testing loss
        log_metrics(epoch, metrics_to_log, average_train_loss, flatten_list(training_probs), flatten_list(training_labels),
                    average_test_loss, flatten_list(testing_probs), flatten_list(testing_labels))
