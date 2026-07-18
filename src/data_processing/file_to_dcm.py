# Recursively search through a directory and rename all files with no extension found within a DICOMDIR tagged folder
# to *.dcm

from pathlib import Path
import os
import sys
import shutil
from utilities.directory_handling import verify_directory

if len(sys.argv) != 2:
    print("ERROR: Incorrect arguments provided\nUsage: python file_to_dcm.py <directory_to_search>")
    quit()

# Get Path information from command line
directory_path = sys.argv[1]

verify_directory(directory_path, 0)


# Get number of files in directory
num_files_renamed = 0

def find_dicom_dirs(directory):
    """
    Recursively find all subdirectories within the given directory that have the DICOMDIR tag
    """
    for subfile in os.listdir(directory):
        new_directory = directory + "/" + subfile

        if not os.path.isdir(new_directory) and new_directory.endswith("DICOMDIR"):
            print("Found DICOMDIR")
            traverse_directory_and_rename(directory)
            return

        if os.path.isdir(new_directory):
            find_dicom_dirs(new_directory)


def traverse_directory_and_rename(directory):
    """
    Go down each branch of a directory and add .dcm to any files without a file extension.
    :param directory: Directory to be searched for files.
    """
    global num_files_renamed
    for subfile in os.listdir(directory):
        new_directory = directory + "/" + subfile

        new_directory_as_path = Path(new_directory)
        if new_directory_as_path.suffix:
            continue

        if not os.path.isdir(new_directory):
            shutil.move(new_directory, new_directory + ".dcm")


            print(f"Renamed file {new_directory} to {new_directory}.dcm")

            num_files_renamed += 1

            continue

        traverse_directory_and_rename(new_directory)


find_dicom_dirs(directory_path)

print(f"Renamed {num_files_renamed} files.\nDone.")
