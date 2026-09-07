import os.path
import sys
from pathlib import Path
from portal_utils import extract_filename_from_path, extract_base_directory_from_path
import shutil


directory_path = Path(sys.argv[1]).resolve()

if not directory_path.exists():
    print(f"Could not find directory {directory_path}")
    quit()

def recursive_search_and_rename(parent_directory: Path):
    for directory in parent_directory.iterdir():
        if directory.is_dir():
            recursive_search_and_rename(directory)
        elif str(directory).endswith(".dcm"):
            print(f"Renaming {directory}")
            shutil.move(directory, str(parent_directory) + "/" + extract_filename_from_path(parent_directory, 1) + ".dcm")

        else:
            print(f"Non dcm file found {extract_filename_from_path(directory, 2)}")
            print(f"Size: {os.path.getsize(directory)}")
            if input("Convert above file to dcm?: ").upper() == "Y":
                shutil.move(directory, str(directory) + ".dcm")


recursive_search_and_rename(directory_path)