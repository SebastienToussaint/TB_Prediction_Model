# Extract the PID from all .dcm files in a directory and write them to an output file
import csv
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

directory = Path(sys.argv[1]).resolve()
verify_directory(directory, 0)
save_directory = Path(sys.argv[2])


# Verify save directory exists and create parent directories if needed
if str(save_directory).endswith(".csv"):
    save_path = save_directory.resolve()
    base_directory = str(save_directory)[0:str(save_directory).index(extract_filename_from_path(str(save_directory), 1))]
    verify_directory(base_directory, 1)
else:
    verify_directory(save_directory, 1)
    save_path = f"{save_directory.resolve()}/pids.csv"

written_ids = []

data = [["PatientID", "PatientName"]]

# Get data from files.
for file in directory.iterdir():
    if str(file).endswith(".dcm"):
        try:

            pid = dcmread(file).PatientID
            pname = dcmread(file).PatientName

            if pid not in written_ids:
                written_ids.append(pid)
                pname = dcmread(file).PatientName
                data.append([pid, pname])

        except:
            print(f"Could not get ID or name from file {file}")

# Write data to csv
with open(save_path, "w", newline="") as file:
    csv_writer = csv.writer(file)

    csv_writer.writerows(data)

print(f"Finished extracting PIDs. Outputs saved in {save_directory}")

