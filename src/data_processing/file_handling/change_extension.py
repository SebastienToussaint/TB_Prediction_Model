# Script was created primarily to change some files with a uppercase .DCM extension to .dcm
# Should not be used to change image formats such as .jpg -> .png

import shutil
import sys
from pathlib import Path
from utilities.string_manipulation import extract_filename_from_path
from utilities.directory_handling import verify_directory

# Argument management
if len(sys.argv) != 4:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make change_extension ARGS='<directory_to_search> <old_extension> <new_extension>'")
    quit()

target_directory = Path(sys.argv[1])
verify_directory(target_directory, 0)
old_extension = sys.argv[2]
new_extension = sys.argv[3]

# Change old extensions into new extensions
for file in target_directory.iterdir():
    if file.name.endswith(old_extension):
        shutil.move(f"{target_directory.resolve()}/{file.name}",
                    f"{target_directory.resolve()}/{extract_filename_from_path(file.name, 1)}{new_extension}")


