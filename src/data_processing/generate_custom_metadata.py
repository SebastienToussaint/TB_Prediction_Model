# Generates custom metadata from scratch. For use when study does not have pre-existing metadata to draw from.
# Simply records the names of the images withing the directory and saves the provided label and study of origin.
# Will append to the given csv file, so can consecutively run to apply different labels

import os.path
import sys
from pathlib import Path
import csv
from utilities.directory_handling import verify_directory
from utilities.string_manipulation import extract_filename_from_path

def write_custom_csv(csv_file,  filepath, label, study_name):
    """
    Writes the given parameters into the provided csv_file in the standard metadata format of this project
    """
    data_writer = csv.writer(csv_file)

    corresponding_image_name = extract_filename_from_path(filepath, 1)
    patient_sex = None
    patient_age = None
    tb_status = label
    additional_information = None

    data_line = [corresponding_image_name, patient_sex, patient_age, tb_status,
                 additional_information, corresponding_image_name, study_name]

    data_writer.writerow(data_line)

# Argument Management
if len(sys.argv) != 5:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make metadata_to_csv ARGS='<metadata_directory> <csv_savefile_path> <label (TB, Healthy, Other)> "
          "<study_name>")
    quit()

target_directory = Path(sys.argv[1])
save_directory = Path(sys.argv[2])
patient_label = sys.argv[3]
study_of_origin = sys.argv[4]

verify_directory(target_directory, 0)

# Verify save directory exists and create parent directories if needed
if str(save_directory).endswith(".csv"):
    csv_path = save_directory.resolve()
    base_directory = str(save_directory)[0:str(save_directory).index(extract_filename_from_path(str(save_directory), 1))]
    verify_directory(base_directory, 1)
else:
    verify_directory(save_directory, 1)
    csv_path = f"{save_directory.resolve()}/metadata.csv"

# Need to know for insertion of headers or not
metadata_file_exists = os.path.exists(csv_path) and (os.path.getsize(csv_path) > 0)

# Write standardised metadata to csv file
with open(csv_path, "a", newline="", encoding="utf-8") as csv_savefile:
    if not metadata_file_exists:
        # Write header
        csv_writer = csv.DictWriter(csv_savefile, ["Associated CXR", "Sex", "Age", "Status",
                                                   "Notes", "Original CXR Name", "Study of Origin"])
        csv_writer.writeheader()

    for file in target_directory.iterdir():
        print(f"Writing {file.resolve()}...")
        write_custom_csv(csv_savefile, file.resolve(), patient_label, study_of_origin)

print("Done.")
