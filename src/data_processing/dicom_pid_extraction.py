# Extract the PID from all .dcm files in a directory and write them to an output file
import os
import sys
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pathlib import Path
from utilities.directory_handling import verify_directory
from utilities.string_manipulation import extract_filename_from_path

# Argument Management
if len(sys.argv) != 3:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make dicom_pid_extraction ARGS='<directory_to_search> <savefile_path>'")
    quit()

directory = sys.argv[1]
verify_directory(directory, 0)
save_directory = Path(sys.argv[2])

# Verify save directory exists and create parent directories if needed
if str(save_directory).endswith(".txt"):
    save_path = save_directory.resolve()
    base_directory = str(save_directory)[0:str(save_directory).index(extract_filename_from_path(str(save_directory), 1))]
    verify_directory(base_directory, 1)
else:
    verify_directory(save_directory, 1)
    save_path = f"{save_directory.resolve()}/pids.txt"

written_ids = []

# Open save file
with open(save_path, "a") as savefile:
    # Go through each file in the target directory.
    # If target is .dcm, attempt to extract PatientID and add to save file if it hasn't been added yet
    for file in os.listdir(directory):
        if file.endswith(".dcm"):
            try:
                if dcmread(directory + "/" + file).dir() in written_ids:
                    print("ID already written")
                else:
                    pid = dcmread(directory + "/" + file).PatientID
                    written_ids.append(pid)
                    print(pid)
                    savefile.write(pid+ "\n")
            except InvalidDicomError:
                print(f"Could not get PID from file {file}. Continuing")

print("Done.")