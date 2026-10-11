# Recursively search through given directory and copy all .dcm, .img, .jpeg and .png files found to an output file
# Can provide a file suffix, in which case only files with that suffix will be collected.
import os
import sys
import shutil
from utilities.directory_handling import verify_directory
from utilities.string_manipulation import extract_filename_from_path
import csv
from pathlib import Path

def count_num_target_files(directory, file_suffixes, counted_files):
    """
    Recursively count number of target files within the target path
    """
    for subfile in os.listdir(directory):
        new_directory = directory + "/" + subfile
        if not os.path.isdir(new_directory):
            if len(file_suffixes) == 0 or new_directory.endswith(tuple(file_suffixes)):
                counted_files += 1
            continue
        count_num_target_files(new_directory, file_suffixes, counted_files)

    return counted_files

def traverse_directory_and_copy(directory, save_directory, file_suffixes, index=0,
                                metadata_tracking=False, metadata=None, number_of_files = 0):
    """
    Go down each branch of a directory and copy any files that have an extension found in the given list to the save folder.
    Renames copied file by adding an incrementing index to the beginning of the original file name (Avoids overwriting
    labels at the end of filenames).
    """
    if number_of_files == 0:
        number_of_files = count_num_target_files(directory, file_suffixes, 0)

    for subfile in os.listdir(directory):
        new_directory = directory + "/" + subfile

        if not os.path.isdir(new_directory):
            if len(file_suffixes) == 0 or new_directory.endswith(tuple(file_suffixes)):
                # Isolate name of file
                if "\\" in new_directory:
                    final_slash_index = max(new_directory.rindex("/"), new_directory.rindex("\\"))
                else:
                    final_slash_index = new_directory.rindex("/")

                # Leading zeros in filename
                zeros = max(0, len(str(number_of_files)) - len(str(index)) + 1)

                # Compile new name and save
                new_name = f"{zeros*'0'}{index}" + "_" + new_directory[final_slash_index + 1:]
                shutil.copy(new_directory, save_directory + "/" + new_name)
                print(f"Copied file {new_directory} to {save_directory}/{new_name}")

                index += 1

                # Update metadata
                if metadata_tracking:
                    for row in metadata:
                        if row[0] == extract_filename_from_path(new_directory, 1):
                            row[0] = extract_filename_from_path(new_name, 1)

            continue

        # Recursively search and copy
        index = traverse_directory_and_copy(new_directory, save_directory, file_suffixes, index, metadata_tracking, metadata, number_of_files)
    return index


def collect_files(target_directory, save_directory, metadata_filepath = None, file_suffixes = None):
    verify_directory(target_directory, 0)
    verify_directory(save_directory, 1)

    # Setup metadata tracking state
    metadata_tracking = False
    if metadata_filepath.upper() not in ["0", "NONE"]:
        verify_directory(metadata_filepath, 0)
        metadata_tracking = True

    if len(file_suffixes) > 0:
        if input(f"WARNING: Only files with the following suffixes will be copied to the save directory: {file_suffixes}\n"
                 f"Go ahead with file copying? (Y/n): ").upper() not in ["Y", "YES"]:
            print("Cancelling job...")
            quit()

    # Get number of files in directory
    index = len(os.listdir(save_directory))

    # With metadata tracking
    if metadata_tracking:
        # Open metadata file and copy all lines to variable
        metadata = []
        print("Moving files while tracking metadata...")
        # Read metadata
        with open(metadata_filepath, "r", newline="", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:
                metadata.append(line)

        # Collect files
        traverse_directory_and_copy(target_directory, save_directory, file_suffixes, index, metadata_tracking, metadata)

        # Write altered metadata back to metadata file
        with open(metadata_filepath, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(metadata)

    # Without metadata tracking
    else:
        print("Moving files without metadata tracking...")
        traverse_directory_and_copy(target_directory, save_directory, file_suffixes, index)

    print(f"Finished collecting files. Copied files saved to {Path(save_directory).resolve()}")


if __name__ == "__main__":
    # Argument Management
    if len(sys.argv) < 3:
        print("ERROR: Incorrect arguments provided\n"
              "Usage: make image_collection ARGS='<directory_to_search> <save_directory> <metadata_file_path [None]> (optional) {file suffixes}'")
        quit()

    directory_path = sys.argv[1]
    save_directory_path = sys.argv[2]

    metadata_file_path = sys.argv[3]

    # Get acceptable suffixes if appropriate
    collection_suffixes = []
    if len(sys.argv) >= 5:
        for i in range(4, len(sys.argv)):
            collection_suffixes.append(sys.argv[i])

    collect_files(directory_path, save_directory_path, metadata_file_path, collection_suffixes)
