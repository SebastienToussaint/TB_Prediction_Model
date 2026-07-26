# Extract the PID from all .dcm files in a directory and write them to an output file

import os
import sys
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from utilities.directory_handling import verify_directory

if len(sys.argv) != 3:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make dicom_pid_extraction ARGS='<directory_to_search> <savefile>'")
    quit()

# Get directory from arguments
directory = sys.argv[1]

verify_directory(directory, 0)

written_ids = []

# Open save file
with open(sys.argv[2], "a") as savefile:
    # Go through each file in the target directory.
    # If target is .dcm, attempt to extract PatientID and add to save file if it hasn't been added yet
    for file in os.listdir(directory):
        if file.endswith(".dcm"):
            try:
                if dcmread(directory + "/" + file).PatientID in written_ids:
                    print("ID already written")
                else:
                    written_ids.append(dcmread(directory + "/" + file).PatientID)
                    print(dcmread(directory + "/" + file).PatientID)
                    savefile.write(dcmread(directory + "/" + file).PatientID + "\n")
            except InvalidDicomError:
                print(f"Could not get PID from file {file}. Continuing")

print("Done.")