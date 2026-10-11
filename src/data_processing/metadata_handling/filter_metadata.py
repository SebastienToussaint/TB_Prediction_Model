import csv
import sys
from pathlib import Path
from utilities.string_manipulation import extract_filename_from_path

if len(sys.argv) != 3:
    print("ERROR while converting metadata. Incorrect arguments provided.\n"
          "Usage: python -m *.data_processing.metadata_handling.filter_metadata <metadata filepath> <images path>")
    quit()

metadata_path = Path(sys.argv[1]).resolve()
images_path = Path(sys.argv[2]).resolve()


if not str(metadata_path).endswith(".csv"):
    print("ERROR while converting metadata. Incorrect arguments provided.\n"
          f"Metadata path needs to be a .csv file. Provided: {metadata_path}")
    quit()

if not images_path.is_dir():
    print("ERROR while converting metadata. Incorrect arguments provided.\n"
          "Images path needs to be a directory.")
    quit()



metadata = []

with open(metadata_path, "r") as file:
    csv_reader = csv.reader(file)

    for line in csv_reader:
        metadata.append(line)

image_names = []
for path in images_path.iterdir():
    image_names.append(extract_filename_from_path(path, 0))


i = 0
while i < len(metadata):
    if metadata[i][0] not in image_names and i != 0:
        metadata.pop(i)
    else:
        i += 1

with open(metadata_path, "w", newline="") as file:
    csv_writer = csv.writer(file)

    csv_writer.writerows(metadata)

