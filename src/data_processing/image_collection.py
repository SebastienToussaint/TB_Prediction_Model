# Recursively search through given directory and copy all .dcm, .img, .jpeg and .png files found to an output file

import os
import sys
import shutil
from utilities.directory_handling import verify_directory

if len(sys.argv) != 3:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make image_collection ARGS='<directory_to_search> <save_directory>'")
    quit()

# Get Path information from command line
directory_path = sys.argv[1]
save_directory_path = sys.argv[2]

verify_directory(directory_path, 0)
verify_directory(save_directory_path, 1)

# Get number of files in directory
index = len(os.listdir(save_directory_path))
num_files_copied = 0


def traverse_directory_and_copy(directory, save_directory):
    """
    Go down each branch of a directory and copy any .dcm, .img, .jpeg and .png files found to save folder.
    Renames copied file by adding an incrementing index to the beginning of the original file name (Avoids overwriting
    labels at the end of filenames).
    :param directory: Directory to be searched for images.
    :param save_directory: Directory where found images should be copied to.
    :return:
    """
    global index, num_files_copied
    for subfile in os.listdir(directory):
        new_directory = directory + "/" + subfile
        if not os.path.isdir(new_directory):
            if new_directory.endswith(".dcm") or new_directory.endswith(".jpeg") or new_directory.endswith(".png") \
                or new_directory.endswith(".img"):
                if "\\" in new_directory:
                    final_slash_index = max(new_directory.rindex("/"), new_directory.rindex("\\"))
                else:
                    final_slash_index = new_directory.rindex("/")

                new_name = str(index) + "_" + new_directory[final_slash_index + 1:]

                shutil.copy(new_directory, save_directory + "/" + new_name)
                print(f"Copied file {new_directory} to {save_directory}/{new_name}")
                index += 1
                num_files_copied += 1

            continue

        traverse_directory_and_copy(new_directory, save_directory)


traverse_directory_and_copy(directory_path, save_directory_path)

# Keep track of directories copied from
with open(save_directory_path + "/Compilation_history.txt", "a") as file:
    file.write(f"Added {num_files_copied} files from {directory_path}\n")


print(f"Copied {num_files_copied} files.\nDone.")