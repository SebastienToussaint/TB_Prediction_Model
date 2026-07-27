import sys
from pathlib import Path
import csv
from utilities.directory_handling import verify_directory
from utilities.string_manipulation import extract_num_from_string, extract_filename_from_path




def write_csv_montgomery(csv_file,  metadata_filepath):
    """
    Extract metadata from a given metadata file with the montgomery study format and write to given csv file.
    """
    data_writer = csv.writer(csv_file)

    with open(metadata_filepath, "r") as metadata_file:
        corresponding_image_name = metadata_file.name
        corresponding_image_name = extract_filename_from_path(corresponding_image_name, 1)

        patient_sex_line = metadata_file.readline()
        patient_sex = patient_sex_line[patient_sex_line.rindex(":")+1:].strip()

        patient_age_line = metadata_file.readline()
        patient_age = str(int(extract_num_from_string(patient_age_line))) #str(int(...)) to remove leading 0's

        tb_status_line = metadata_file.read()
        if tb_status_line.strip() == "normal":
            tb_status = "Healthy"
            additional_information = ""
        else:
            tb_status = "TB"
            additional_information = tb_status_line

    data_line = [corresponding_image_name, patient_sex, patient_age, tb_status, additional_information]

    data_writer.writerow(data_line)


def write_csv_shenzhen(csv_file,  metadata_filepath):
    data_writer = csv.writer(csv_file)

    with open(metadata_filepath, "r") as metadata_file:
        corresponding_image_name = metadata_file.name
        corresponding_image_name = extract_filename_from_path(corresponding_image_name)

        patient_sex_age_line = metadata_file.readline()
        patient_sex = patient_sex_age_line[0:patient_sex_age_line.index(" ")]

        if patient_sex.upper() in ["FEMALE,", "FEMALE", "FEMAL"]: # Done to account for inconsistencies in metadata files
            patient_sex = "F"
        elif patient_sex.upper() in ["MALE,", "MALE"]:
            patient_sex = "M"


        patient_age = extract_num_from_string(patient_sex_age_line)

        tb_status_line = metadata_file.read()
        if tb_status_line.strip() == "normal":
            tb_status = "Healthy"
            additional_information = ""
        else:
            tb_status = "TB"
            additional_information = tb_status_line

    data_line = [corresponding_image_name, patient_sex, patient_age, tb_status, additional_information]

    data_writer.writerow(data_line)


target_directory = Path(sys.argv[1])
save_directory = Path(sys.argv[2])
template = sys.argv[3]

verify_directory(target_directory, 0)

if str(save_directory).endswith(".csv"):
    csv_path = save_directory.resolve()
    base_directory = str(save_directory)[0:str(save_directory).index(extract_filename_from_path(str(save_directory), 1))]
    verify_directory(base_directory, 1)
else:
    verify_directory(save_directory, 1)
    csv_path = f"{save_directory.resolve()}/metadata.csv"

with open(csv_path, "a", newline="", encoding="utf-8") as csv_savefile:
    for file in target_directory.iterdir():
        print(f"Writing {file.resolve()}...")
        if template.upper() in ["MONTGOMERY", "MONT"]:
            write_csv_montgomery(csv_savefile, file.resolve())
        elif template.upper() in ["SHENZHEN", "SHEN"]:
            write_csv_shenzhen(csv_savefile, file.resolve())


print("Done.")
