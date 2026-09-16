import sys
from pathlib import Path
from utilities.directory_handling import verify_directory
import shutil
from utilities.string_manipulation import extract_filename_from_path
import csv

def rename_file_no_csv(file, file_index, number_of_files):
    """
    Renames files without adjusting an associated metadata file
    """
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

def rename_file_csv(file, csv_data, file_index, number_of_files):
    """
    Renames files while adjusting the given metadata structure (renames old names in metadata to new names)
    """
    current_file_name = extract_filename_from_path(file.name, 1)
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

    # Update csv data
    for row in csv_data:
        if row[0].endswith(current_file_name):
            row[0] = extract_filename_from_path(new_filename, 1)
            return

    print(f"No metadata found for {file.name}")


# Argument structure:
# Target directory - Metadata directory- Prefix to use - Keep labels (0, 1) - Label tag (if keep label 1) - Add label (0, 1) (if keep label 0) -
# Label to add (if keep label 0 and add label 1)

# Argument Management
try:
    directory_path = Path(sys.argv[1])
    verify_directory(directory_path, 0)

    metadata_filepath = sys.argv[2]

    prefix = sys.argv[3]
    keep_labels = int(sys.argv[4])
    label_tag = "" # Prevent reference error
    if keep_labels:
        label_tag = sys.argv[5]
    add_labels = 0 # Prevent reference error
    label_to_add = "" # Prevent reference error
    if not keep_labels and len(sys.argv) >= 5:
        add_labels = int(sys.argv[5])
        if add_labels:
            label_to_add = sys.argv[6]
except:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make standardize_filenames ARGS='<directory_to_search> <metadata_directory (None)> <new_prefix>"
          " <keep_labels [0,1]> <label_identifier (if keep_labels = 1)> "
          "(optional) <add_label [0, 1] (if keep_labels = 0)> <label_to_add>'")
    quit()


index = 1
total_num_files = len(list(directory_path.iterdir()))

# Go through directory and apply name changes
if str(metadata_filepath).endswith(".csv"):
    print("Renaming and updating metadata file")

    data = []
    with open(metadata_filepath, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            data.append(line)

    for filepath in sorted(directory_path.iterdir()):
        rename_file_csv(filepath, data, index, total_num_files)
        index += 1

        with open(metadata_filepath, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(data)

else:
    print("Renaming without metadata tracking")
    for filepath in sorted(directory_path.iterdir()):
        rename_file_no_csv(filepath, index, total_num_files)
        index += 1

print("Done.")