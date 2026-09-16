from objects.models import CNN
from objects.data_handling import DataStructure
import torch

def run_cross_validation(model_class, data_structure, number_of_epochs: int, loss_fn, optimizer, target_device: str = "cpu"):

    # Log info
    print(f"Running cross validation with following settings:\n"
          f" -Model Class: {model_class}\n"
          f" -Epochs: {number_of_epochs}\n"
          f" -Device: {target_device}\n"
          f" -Loss function: {loss_fn}\n"
          f" -Optimizer: {optimizer.__class__.__name__}\n"
          f" -Number of Folds: {data_structure.get_num_partitions()}\n")

    folds = data_structure.get_num_partitions()

    models = []
    optimizers = []
    evaluations = []

    for i in range(folds):
        models.append(model_class())

    for model in models:
        model.create_standard(1 , 64, 1)
        optimizers.append(optimizer(model.parameters(), lr=0.001))


    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch+1}...")
        for model_idx, model in enumerate(models):
            print(f"Model {model_idx+1}...")

            model.to(target_device)

            optimizer = optimizers[model_idx]

            total_train_loss = 0
            train_batches_processed = 0
            training_probs = []
            training_labels = []

            total_test_loss = 0
            test_batches_processed = 0
            testing_probs = []
            testing_labels = []

            ds.reset_dataloader_index()

            # Training ============================================================================
            for i in range(folds):
                dataloader = ds.get_next_dataloader()

                if not i == model_idx:
                    model.train()
                    for images, labels, filenames in dataloader:
                        # Update image state
                        images, labels = images.to(target_device), labels.to(target_device)
                        images = images.unsqueeze(1)

                        # Infer labels for each model
                        inferred_logits = model(images).squeeze()
                        inferred_prediction_values = torch.sigmoid(inferred_logits)

                        # Save inferred probabilities and truth labels
                        training_probs.append(inferred_prediction_values.detach().cpu().tolist())
                        training_labels.append(labels.detach().cpu().tolist())

                        # Perform standard training loop operations
                        batch_loss = loss_fn(inferred_logits, labels.float())
                        total_train_loss += batch_loss.item()
                        train_batches_processed += 1

                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()

            # Testing ============================================================================

            dataloader = None # Prevents referencing error

            for i in range(model_idx + 1):  #TODO: Check this is correct
                dataloader = ds.get_next_dataloader()

            model.eval()
            with torch.inference_mode():
                for images, labels, filenames in dataloader:
                    # Update image state
                    images, labels = images.to(target_device), labels.to(target_device)
                    images = images.unsqueeze(1)

                    # Infer labels for each model
                    inferred_logits = model(images).squeeze()
                    inferred_prediction_values = torch.sigmoid(inferred_logits)

                    # Save inferred probabilities and truth labels
                    testing_probs.append(inferred_prediction_values.detach().cpu().tolist())
                    testing_labels.append(labels.detach().cpu().tolist())

                    total_test_loss += loss_fn(inferred_logits, labels.float()).item()
                    test_batches_processed += 1



            model.store_probabilities(training_probs, training_labels, testing_probs, testing_labels)

            # Move device back to cpu to save on vram
            if target_device == "cuda":
                model.to("cpu")


    return models




ds = DataStructure()
ds.reset_dataloader_index()
ds.set_native_datapath("E:/Project/TESTING/Data/Dataloaders/fashion_dataloaders")

validated_models = run_cross_validation(CNN, ds, 2, torch.nn.BCEWithLogitsLoss(), torch.optim.Adam, "cuda")

for idx, model in enumerate(validated_models):
    model.draw_eval_graphs(f"E:\Project\TESTING\Models\CV_models/model_{idx}")




