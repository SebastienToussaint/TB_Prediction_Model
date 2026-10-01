import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("Error while creating templates: Incorrect arguments provided.\n"
          "Usage: python make_template.py <template_type (train, data)> <template_save_path>")
    quit()

template_type = sys.argv[1]
save_path = sys.argv[2]

if template_type.upper() not in ["TRAIN", "TRAINER", "TRAINING", "DATA"]:
    print(f"Template type {template_type} not found.\n"
          f"Options: TRAIN, DATA")
    quit()

if not save_path.endswith(".txt"):
    print(f"Save path {save_path} needs to be a .txt file")
    quit()

save_path = Path(save_path).resolve()

if template_type.upper() in ["TRAIN", "TRAINER", "TRAINING"]:
    lines = [
        "model_path=''                   # If left empty, will create model from scratch\n",
        "model_save_directory=''\n",
        "model_save_name=model.pkl\n",
        "dataloaders_path=''\n",
        "num_epochs=10\n",
        "evaluation_partitions=0          # How many dataloaders should be reserved for evaluation every epoch\n",
        "loss_function=BCE\n",
        "optimizer=Adam                # SGD, SGD-Momentum, ADAM, ADAMW, ADAFACTOR, RMSPROP\n",
        "learning_rate=0.001\n",
        "training_device=Any             # Any, CUDA, CPU\n",
        "model_save_interval=5\n",
        "save_metrics=True               # True, False\n",
        "metrics_to_log=Acc,Bacc,MCC     # Acc, Bacc, MCC, F1, Recall, Specificity, Precision\n",
        "make_charts=True                # Automatically makes the metric charts in model_save_path/metrics/"
    ]
    with open(save_path, "w") as file:
        file.writelines(lines)
