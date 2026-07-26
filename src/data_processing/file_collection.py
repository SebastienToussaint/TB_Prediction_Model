# Recursively search through given directory and copy all .dcm, .img, .jpeg and .png files found to an output file
# Can provide a file suffix, in which case only files with that suffix will be collected.

import os
import sys
import shutil
from utilities.directory_handling import verify_directory

def count_num_target_files(directory, file_suffixes, counted_files):
    """
    Recursively count number of target files within the target path
    :param directory: Directory to be searched for images.
    :return:
    """
    global number_of_files
    for subfile in os.listdir(directory):
        new_directory = directory + "/" + subfile
        if not os.path.isdir(new_directory):
            if len(file_suffixes) == 0 or new_directory.endswith(tuple(file_suffixes)):
                counted_files += 1
            continue
        count_num_target_files(new_directory, file_suffixes, counted_files)

    return counted_files

def traverse_directory_and_copy(directory, save_directory, file_suffixes):
    """
    Go down each branch of a directory and copy any .dcm, .img, .jpeg and .png files found to save folder.
    Renames copied file by adding an incrementing index to the beginning of the original file name (Avoids overwriting
    labels at the end of filenames).
    :param directory: Directory to be searched for images.
    :param save_directory: Directory where found images should be copied to.
    :return:
    """
    global index, num_files_copied, number_of_files
    for subfile in os.listdir(directory):
        new_directory = directory + "/" + subfile
        if not os.path.isdir(new_directory):
            if len(file_suffixes) == 0 or new_directory.endswith(tuple(file_suffixes)):
                if "\\" in new_directory:
                    final_slash_index = max(new_directory.rindex("/"), new_directory.rindex("\\"))
                else:
                    final_slash_index = new_directory.rindex("/")

                zeros = max(0, len(str(number_of_files)) - len(str(index)) + 1)

                new_name = f"{zeros*'0'}{index}" + "_" + new_directory[final_slash_index + 1:]

                shutil.copy(new_directory, save_directory + "/" + new_name)
                print(f"Copied file {new_directory} to {save_directory}/{new_name}")
                index += 1
                num_files_copied += 1

            continue

        traverse_directory_and_copy(new_directory, save_directory, file_suffixes)

if len(sys.argv) < 3:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make image_collection ARGS='<directory_to_search> <save_directory> (optional) {file suffixes}'")
    quit()

# Get Path information from command line
directory_path = sys.argv[1]
save_directory_path = sys.argv[2]

verify_directory(directory_path, 0)
verify_directory(save_directory_path, 1)

# Get acceptable suffixes if appropriate
collection_suffixes = []
if len(sys.argv) >= 4:
    for i in range(3, len(sys.argv)):
        collection_suffixes.append(sys.argv[i])

if len(collection_suffixes) > 0:
    if input(f"WARNING: Only files with the following suffixes will be copied to the save directory: {collection_suffixes}\n"
             f"Go ahead with file copying? (Y/n): ").upper() not in ["Y", "YES"]:
        print("Cancelling job...")
        quit()

# Get number of files in directory
index = len(os.listdir(save_directory_path))
number_of_files = count_num_target_files(directory_path, collection_suffixes, 0)
num_files_copied = 0

traverse_directory_and_copy(directory_path, save_directory_path, collection_suffixes)

# Keep track of directories copied from
with open(save_directory_path + "/Compilation_history.txt", "a") as file:
    file.write(f"Added {num_files_copied} files from {directory_path}\n")


print(f"Copied {num_files_copied} files.\nDone.")