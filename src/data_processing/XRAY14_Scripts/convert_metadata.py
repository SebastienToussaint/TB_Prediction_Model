import csv
import sys
from pathlib import Path


if len(sys.argv) != 3:
    print("ERROR while converting metadata. Incorrect arguments provided.\n"
          "Usage: python -m *.data_processing.XRAY14_Scripts.converty metadata <metadata filepath> <output path>")
    quit()

metadata_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()


if not str(metadata_path).endswith(".csv"):
    print("ERROR while converting metadata. Incorrect arguments provided.\n"
          f"Metadata path needs to be a .csv file. Provided: {metadata_path}")
    quit()

if not str(output_path).endswith(".csv"):
    print("ERROR while converting metadata. Incorrect arguments provided.\n"
          f"Output path needs to be a .csv file. Provided: {output_path}")
    quit()

metadata = [["Associated CXR", "Sex", "Age", "BMI", "Country", "Status",
            "Notes", "Original CXR Name", "Study of Origin"]]

with open(metadata_path, "r") as file:
    csv_reader = csv.reader(file)

    for i, line in enumerate(csv_reader):
        if i == 0:
            continue

        metadata.append([line[0], line[5], line[4], "unknown", "unknown", "Sick Non TB", "Finding: " + line[1], line[0], "XRAY14"])


with open(output_path, "w", newline="") as file:
    csv_writer = csv.writer(file)

    csv_writer.writerows(metadata)



