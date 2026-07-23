import sys
from pathlib import Path

from utilities.directory_handling import verify_directory
import shutil

# Argument structure:
# Target directory - Prefix to use - Keep labels (0, 1) - Label tag (if keep label 1) - Add label (0, 1) (if keep label 0) -
# Label to add (if keep label 0 and add label 1)

directory_path = Path(sys.argv[1])
verify_directory(directory_path, 0)
prefix = sys.argv[2]
keep_labels = int(sys.argv[3])
label_tag = "" # Prevent reference error
if keep_labels:
    label_tag = sys.argv[4]
add_labels = 0 # Prevent reference error
label_to_add = "" # Prevent reference error
if not keep_labels and len(sys.argv) >= 5:
    add_labels = int(sys.argv[4])
    if add_labels:
        label_to_add = sys.argv[5]

number_of_files = len(list(directory_path.iterdir()))

file_index = 1
# Go through directory and apply name changes
for file in sorted(directory_path.iterdir()):

    # print(file.name)
    if "." in file.name:
        file_extension = file.name[file.name.rindex("."):]
        new_filename = file.name[:file.name.rindex(".")]
    else:
        print(f"No extension found for {file}. Renaming without extension")
        file_extension = ""
        new_filename = file.name

    zeros = max(0, len(str(number_of_files)) - len(str(file_index)) + 1)

    if keep_labels:
        label = new_filename[new_filename.rindex(label_tag)+1:]
        new_filename = f"{prefix}_{zeros*'0'}{file_index}{label_tag}{label}{file_extension}"
    elif add_labels:
        new_filename = f"{prefix}_{zeros*'0'}{file_index}{label_tag}{label_to_add}{file_extension}"
    else:
        new_filename = f"{prefix}_{zeros*'0'}{file_index}{file_extension}"

    file_index += 1

    print(f"Renaming file {file} to {new_filename}")
    shutil.move(str(directory_path.absolute()) + "/" + file.name, str(directory_path.absolute()) + "/" + new_filename)
