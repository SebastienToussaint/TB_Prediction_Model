import csv
import shutil
import sys
from pathlib import Path
from utilities.string_manipulation import extract_filename_from_path

metadata = []
selected_ids = []

target_dir = Path(sys.argv[1])
dest = sys.argv[2]
metadata_path = sys.argv[3]

with open(metadata_path, "r") as file:
    csv_reader = csv.reader(file)

    for line in csv_reader:
        if line[3] not in selected_ids and line[1].upper() != "NO FINDING":
            selected_ids.append(line[3])
            metadata.append(line)



image_paths = [line[0] for line in metadata]

def traverse_directory(directory: Path, destination):
    for d in directory.iterdir():
        if d.is_dir():
            traverse_directory(d, destination)
        elif str(d).endswith(".png"):
            image_name = extract_filename_from_path(str(d), 0)
            if image_name in image_paths:
                shutil.copy(d, str(destination) + f"/Sick_non_TB/{image_name}")






# Print csv
with open(f"{dest}/xray14_metadata.csv", "w", newline="") as file:
    csv_writer = csv.writer(file)

    csv_writer.writerows(metadata)

traverse_directory(target_dir, dest)



