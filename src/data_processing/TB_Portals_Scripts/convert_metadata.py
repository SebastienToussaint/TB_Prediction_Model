import sys
from pathlib import Path
import csv
from portal_utils import verify_directory

if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Error while converting metadata. Incorrect usage.\n"
              "Usage: make convert_metadata ARGS='<target_metadata_file> <savefile_path>'")
        quit()
    target_file = Path(sys.argv[1]).resolve()
    savefile_path = Path(sys.argv[2]).resolve()

    if not str(target_file).endswith(".csv"):
        print("Error while converting metadata. Target file needs to be a .csv file.")
        quit()
    if not str(savefile_path).endswith(".csv"):
        print("Error while converting metadata. Savefile path needs to be a .csv file.")
        quit()

    verify_directory(target_file, 0)
    verify_directory(target_file, 1)

    raw_data = []

    with open(target_file, "r") as file:
        csv_reader = csv.reader(file)

        for line in csv_reader:
            raw_data.append(line)

    filtered_data = [["Associated CXR", "Sex", "Age", "BMI", "Country", "Status", "Notes", "Original CXR Name", "Study of Origin"]]
    processed_images = []


    for line in raw_data[1:]:


        associated_cxr = line[0]

        if associated_cxr in processed_images:
            continue

        sex = line[3][0].upper()
        age = line[2]
        bmi = line[13]
        status = "TB"
        country = line[12]
        notes = line[11]
        study_of_origin = "TB Portals"

        if bmi == "NULL":
            bmi = "unkown"

        notes = "Comorbidity: " + notes

        filtered_data.append([associated_cxr, sex, age, bmi, country, status, notes, associated_cxr, study_of_origin])
        processed_images.append(associated_cxr)

    with open(savefile_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(filtered_data)
