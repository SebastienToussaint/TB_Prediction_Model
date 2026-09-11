import sys
from pathlib import Path
from utilities.directory_handling import verify_directory
from data_handling import DataStructure

def _process_config(path_to_file):
    path_to_file = str(Path(str(path_to_file)).resolve())

    image_directories = []
    save_directory = ""
    num_partitions = 0
    batch_size = 0
    num_negative_images = "ALL"
    num_positive_images = "ALL"
    negative_split_percentage = None

    with open(path_to_file, "r") as file:

        for line in file:
            if line[0] == "#" or line == "":
                continue

            # Clean up line
            if "#" in line:
                line = line[0: line.index("#")]

            line = line.strip()

            if "=" not in line:
                print(f"Unknown line {line} found in config file at {path_to_file}")

            # Process line
            if line.startswith("ImageDirectory"):
                path = line[line.index("=")+1:]
                path = str(Path(str(path)).resolve())
                verify_directory(path, 0)
                image_directories.append(path)
            elif line.startswith("NumPartitions"):
                try:
                    num_partitions = int(line[line.index("=")+1:])
                except ValueError:
                    print(f"Error while processing config file {path_to_file}\n"
                          f"{line[line.index("=")+1:]} is not a valid option for NumPartitions. This needs to be a positive integer.")
                    quit()
            elif line.startswith("BatchSize"):
                try:
                    batch_size = int(line[line.index("=")+1:])
                except ValueError:
                    print(f"Error while processing config file {path_to_file}\n"
                          f"{line[line.index("=")+1:]} is not a valid option for BatchSize. This needs to be a positive integer.")
                    quit()
            elif line.startswith("SaveDirectory"):
                path = line[line.index("=") + 1:]
                path = str(Path(str(path)).resolve())
                save_directory = path
            elif line.startswith("NumberNegativeImages"):
                if not line[line.index("=") + 1:].upper() == "ALL":
                    try:
                        num_negative_images = int(line[line.index("=") + 1:])
                        if num_negative_images < 0:
                            raise ValueError
                    except ValueError:
                        print(f"Error while processing config file {path_to_file}\n"
                              f"{line[line.index("=") + 1:]} is not a valid option for NumberNegativeImages. This needs to be a positive integer.")
                        quit()
            elif line.startswith("NumberPositiveImages"):

                if not line[line.index("=") + 1:].upper() == "ALL":
                    try:
                        num_positive_images = int(line[line.index("=") + 1:])
                        if num_positive_images < 0:
                            raise ValueError
                    except ValueError:
                        print(f"Error while processing config file {path_to_file}\n"
                              f"{line[line.index("=") + 1:]} is not a valid option for NumberPositiveImages. This needs to be a positive integer.")
                        quit()

            elif line.startswith("NegativeSplitPercentage"):
                if line[line.index("=") + 1:].upper() == "NONE":
                    negative_split_percentage = None
                else:
                    try:
                        negative_split_percentage = float(line[line.index("=") + 1:])
                        if 0 <= negative_split_percentage <= 1:
                            raise ValueError
                    except ValueError:
                        print(f"Error while processing config file {path_to_file}\n"
                              f"{line[line.index("=") + 1:]} is not a valid option for NegativeSplitPercentage. This needs to be a float from 0-1.")
                        quit()
            else:
                print(f"Unknown line {line} found in config file at {path_to_file}")

    if not image_directories:
        print(f"Error while processing config file {path_to_file}\n"
              f"ImageDirectory parameter not found. Use make generate_datasat_config to create template")
        quit()
    elif not save_directory:
        print(f"Error while processing config file {path_to_file}\n"
              f"SaveDirectory parameter not found. Use make generate_datasat_config to create template")
        quit()
    elif not num_partitions:
        print(f"Error while processing config file {path_to_file}\n"
              f"NumPartitions parameter not found. Use make generate_datasat_config to create template")
        quit()
    elif not batch_size:
        print(f"Error while processing config file {path_to_file}\n"
              f"BatchSize parameter not found. Use make generate_datasat_config to create template")
        quit()

    return image_directories, save_directory, num_partitions, batch_size, num_negative_images, num_positive_images, negative_split_percentage

def _process_args():
    image_directories = [sys.argv[1]]
    num_partitions = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    save_directory = sys.argv[4]
    num_negative_images = "ALL"
    num_positive_images = "ALL"
    negative_split_percentage = None

    try:
        if len(sys.argv) > 5:
            if sys.argv[5].upper() == "ALL":
                num_negative_images = "ALL"
            else:
                num_negative_images = int(sys.argv[5])
        if len(sys.argv) > 6:
            if sys.argv[6].upper() == "ALL":
                num_positive_images = "ALL"
            else:
                num_positive_images = int(sys.argv[6])
        if len(sys.argv) > 7:
            negative_split_percentage = float(sys.argv[7])
    except ValueError:
        print("Error while creating dataset. Incorrect parameter usage.\n"
              "Usage: make create_dataset ARGS='<images_directory> <num_partitions> <batch_size> <save_directory>"
              " (OPTIONAL) <num_negative_images> <num_positive_images> <negative_split_percentage>'\n"
              "OR\n"
              "Usage: make create_dataset ARGS='<configfile.txt>'\n"
              "(Use make generate_datasat_config to create template)")

    return image_directories, save_directory, num_partitions, batch_size, num_negative_images, num_positive_images, negative_split_percentage

def create_dataset():
    generate_from_txt = False
    if len(sys.argv) < 2:
        print("Error while creating dataset. Incorrect parameter usage.\n"
              "Usage: make create_dataset ARGS='<images_directory> <num_partitions> <batch_size> <save_directory>"
              " (OPTIONAL) <num_negative_images> <num_positive_images> <negative_split_percentage>'\n"
              "OR\n"
              "Usage: make create_dataset ARGS='<configfile.txt>'\n"
              "(Use make generate_datasat_config to create template)")

    elif sys.argv[1].endswith(".txt"):
        generate_from_txt = True

    elif len(sys.argv) < 5:
        print("Error while creating dataset. Incorrect parameter usage.\n"
              "Usage: make create_dataset ARGS='<images_directory> <num_partitions> <batch_size> <save_directory>"
              " (OPTIONAL) <num_negative_images='ALL'> <num_positive_images='ALL'> <negative_split_percentage>'")
        quit()

    if generate_from_txt:
        img_dirs, save_dir, num_parts, batch_sz, num_neg, num_pos, nsp = _process_config(sys.argv[1])
    else:
        img_dirs, save_dir, num_parts, batch_sz, num_neg, num_pos, nsp = _process_args()

    ds = DataStructure()

    print(f"Creating dataset with following parameters:\n"
          f"Image Directories: {img_dirs}\n"
          f"Save Directory: {save_dir}\n"
          f"Number of partitions: {num_parts}\n"
          f"Batch Size: {batch_sz}\n"
          f"Number of negative images: {num_neg}\n"
          f"Number of positive images: {num_pos}\n"
          f"Negative Split Percentage: {nsp}")

    # ds.create_dataset(img_dirs, num_parts, batch_sz, save_dir, num_neg, num_pos, nsp)

create_dataset()