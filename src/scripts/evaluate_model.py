import sys
import torch
from shared_objects.models import CNN
from training.evaluation_run import run_evaluation
from shared_objects.data_handling import DataStructure, CustomDataset
from pathlib import Path
from utilities.directory_handling import verify_directory


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Error in evaluate_model.py. Incorrect arguments provided.\n"
              "Usage: python -m *.evaluate_model <path_to_model> <path_to_dataloaders> <output_directory> (OPTIONAL) -co <cutoff_value> -ts <target_sensitivity>")
        quit()

    model_path = sys.argv[1]
    dataloaders_path = sys.argv[2]
    save_directory = Path(sys.argv[3]).resolve()
    verify_directory(save_directory, 2)

    cutoff = 0.5
    target_sensitivity = None
    if "-co" in sys.argv:
        if "-ts" in sys.argv:
            print("WARNING while running evaluation: Incorrect arguments provided.\n"
                  "Target sensitivity and Cutoff value provided. Given Cutoff value will be used over target sensitivity.")
        try:
            cutoff = float(sys.argv[sys.argv.index("-co") + 1])

            if not 0 <= cutoff <= 1:
                raise ValueError
        except IndexError:
            print("ERROR while running evaluation: Incorrect arguments provided.\n"
                  "Cutoff value not provided after -co tag")
            quit()
        except ValueError:
            if sys.argv[sys.argv.index("-co") + 1].upper() != "J":
                print("ERROR while running evaluation: Incorrect arguments provided.\n"
                      f"Incorrect cutoff value {sys.argv[sys.argv.index("-co") + 1]} given after -co tag.\n"
                      f"Value needs to be between 0 and 1 or 'j' for the cutoff to be set at youden's j point.")
                quit()
            else:
                cutoff = "j"

    elif "-ts" in sys.argv:
        try:
            target_sensitivity = float(sys.argv[sys.argv.index("-ts") + 1])
            if not 0 <= target_sensitivity <= 1:
                raise ValueError
        except IndexError:
            print("ERROR while running evaluation: Incorrect arguments provided.\n"
                  "Target sensitivity value not provided after -ts tag")
            quit()
        except ValueError:
            print("ERROR while running evaluation: Incorrect arguments provided.\n"
                  f"Incorrect target sensitivity value {sys.argv[sys.argv.index("-ts") + 1]} given after -ts tag.\n"
                  f"Value needs to be between 0 and 1")
            quit()

    elif len(sys.argv) > 4:
        print("ERROR while running evaluation: Incorrect arguments provided.\n"
              f"Unknown argument(s) given {sys.argv[4:]}.\n"
              "Usage: python -m *.evaluate_model <path_to_model> <path_to_dataloaders> <output_directory> (OPTIONAL) -co <cutoff_value> -ts <target_sensitivity>")
        quit()

    model = CNN()
    model.load(model_path)

    ds = DataStructure()
    ds.set_native_datapath(dataloaders_path)

    if target_sensitivity is not None:
        run_evaluation(model, ds, None, save_directory, "cuda" if torch.cuda.is_available() else "cpu",
                       target_sensitivity)
    else:
        run_evaluation(model, ds, cutoff, save_directory, "cuda" if torch.cuda.is_available() else "cpu")
