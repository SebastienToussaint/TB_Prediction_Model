from pympler.asizeof import asizeof, asized
import torch
from torch import nn
from training import data_handling
from training.evaluation_metrics import accuracy, mcc, f1_score, core_metrics
from utilities.array_handling import flatten_list

def log_metrics(epoch, metrics_to_log, train_loss, train_probs, train_labels,
                test_loss = None, test_probs = None, test_labels = None):

    round_values = 4

    log_string = f"Epoch {epoch} "

    if test_loss:
        log_string += (f"| Loss: {train_loss} "
                       f"(Test: {test_loss}) ")
    else:
        log_string += f"| Loss: {train_loss} "



    if metrics_to_log:

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



def run_training_loop(model: nn.Module, data_structure: data_handling.DataStructure, number_of_epochs: int, loss_fn,
                      optimizer, target_device: str = "cpu", reserve_evaluation_partition = False,
                      save_metrics = True, metrics_to_log:list = None):

    print("Starting training loop")
    print(f"Training on {target_device}")
    model.to(target_device)


    num_training_partitions = data_structure.get_num_partitions()

    if reserve_evaluation_partition:
        num_training_partitions -= 1

    if num_training_partitions <= 0:
        print(f"Error while training, not enough partitions ({num_training_partitions}) to reserve one for evaluation.\n"
              f"Please create more partitions or set reserve_evaluation_partition to False")
        quit()


    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch+1}/{number_of_epochs}")

        # Training run
        model.train()

        total_train_loss = 0
        train_batches_processed = 0
        training_probs = [] # Stores all the probabilities from the training run in this epoch
        training_labels = [] # Truth labels for training run

        for i in range(num_training_partitions):

            train_dataloader = data_structure.get_next_dataloader()
            # print(f"Running training loop on partition {i+1}/{num_training_partitions}")

            for batch, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(target_device), labels.to(target_device)

                # Infer labels
                images = images.unsqueeze(1) # TODO: Need function to check if dims are correct (for model type) and update them on the fly
                inferred_logits = model(images).squeeze()
                inferred_prediction_values = torch.sigmoid(inferred_logits)

                # Save inferred information
                training_probs.append(inferred_prediction_values.detach().cpu().tolist())
                training_labels.append(labels.detach().cpu().tolist())

                # Perform standard training loop operations
                batch_loss = loss_fn(inferred_logits, labels.float())
                total_train_loss += batch_loss
                train_batches_processed += 1

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        # Calculate training accuracies and avg loss
        # print(training_probs)
        train_accuracy = accuracy.calculate_accuracy(flatten_list(training_probs), flatten_list(training_labels))
        train_balanced_accuracy = accuracy.calculate_balanced_accuracy(flatten_list(training_probs), flatten_list(training_labels))


        total_test_loss = None
        average_test_loss = None
        test_batches_processed = 0
        testing_probs = []
        testing_labels = []

        if reserve_evaluation_partition:
            # print(f"Running evaluation on final partition ({num_training_partitions+1})...")
            # Test run
            total_test_loss = 0
            model.eval()

            test_dataloader = data_structure.get_next_dataloader()

            with torch.inference_mode():

                for images, labels in test_dataloader:
                    images, labels = images.to(target_device), labels.to(target_device)
                    images = images.unsqueeze(1) # TODO: Need function to check if dims are correct (for model type) and update them on the fly
                    inferred_logits = model(images).squeeze()
                    inferred_prediction_values = torch.sigmoid(inferred_logits) # Round these to get labels

                    testing_probs.append(inferred_prediction_values.detach().cpu().tolist())
                    testing_labels.append(labels.detach().cpu().tolist())

                    total_test_loss += loss_fn(inferred_logits, labels.float())
                    test_batches_processed += 1


        # Get average loss values
        average_train_loss = total_train_loss / train_batches_processed
        if total_test_loss:
            average_test_loss = total_test_loss/test_batches_processed

        if save_metrics:
            if reserve_evaluation_partition:
                model.store_probabilities(training_probs, training_labels, testing_probs, testing_labels)
            else:
                model.store_probabilities(training_probs, training_labels)



        log_metrics(epoch, metrics_to_log, average_train_loss, flatten_list(training_probs), flatten_list(training_labels),
                    average_test_loss, flatten_list(testing_probs), flatten_list(testing_labels))
