# Checks if all files are of correct extension, looks for duplicate images, verifies all images are of same size (if applicable)
# (look into verifying colour channels/bit depth etc.)

from pathlib import Path
import sys
import hashlib
import cv2


def generate_image_hash(image_path):
    md5 = hashlib.md5()

    with open(image_path, "rb") as file:
        while chunk := file.read(8192):
            md5.update(chunk)

    return md5.hexdigest()


def add_image_hash(image_path, hash_dictionary):
    # Add image hash to dictionary
    hex_hash = generate_image_hash(image_path)
    if not hash_dictionary.get(hex_hash):
        hash_dictionary[hex_hash] = [image_path]
    else:
        hash_dictionary[hex_hash].append(image_path)


def verify_image_dimensions(image_path, expected_dimensions, bad_dimensions_list):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Could not load {image_path}")
        return False

    height, width = image.shape[:2]

    if width != expected_dimensions[0] or height != expected_dimensions[1]:
        print(width, height, expected_dimensions)
        bad_dimensions_list.append([image_path, f"{width}x{height}"])
        return False

    return True


def verify_file_extension(file, expected_extension, bad_extensions_list):
    if not file.name.endswith(expected_extension):
        bad_extensions_list.append(file)
        return False

    return True


def traverse_directory(directory_path, hash_dictionary, expected_extension, expected_image_dimensions,
                       bad_extensions_list, bad_dimensions_list):
    """
    Recursively traverses a directory, performing quality control checks on any non-directory files encountered
    Checks include:
    - Addition of image hash and filepath to hash_dictionary
    - Verification of file extension (.png)
    - Verification of image dimensions
    Will print out any abnormalities found
    (Does not check for image duplicates by itself)
    """
    for file in directory_path.iterdir():
        if file.is_dir():
            traverse_directory(file, hash_dictionary, expected_extension, expected_image_dimensions,
                               bad_extensions_list, bad_dimensions_list)
        else:
            print(f"Verifying {file.resolve()}...")
            if verify_file_extension(file, expected_extension, bad_extensions_list):
                add_image_hash(file, hash_dictionary)
                if expected_image_dimensions[0] > 0:
                    verify_image_dimensions(file, expected_image_dimensions, bad_dimensions_list)


def find_duplicates(dictionary):
    """
    Finds keys (in this case image hashes) that have multiple items and prints out the file path of duplicate images
    """
    duplicates_found = False
    print("\n\nFinding duplicates...\n")
    for key in dictionary:
        if len(dictionary[key]) > 1:
            duplicates_found = True
            print("  Duplicates found at:")
            for path in dictionary[key]:
                print("  " + str(path))

            print(f"  With hash {key}\n")

    if duplicates_found:
        print("Done.")
    else:
        print("No duplicates found!\n")

# Argument management
if len(sys.argv) == 3:
    target_directory = Path(sys.argv[1])
    expected_file_extension = sys.argv[2]
    expected_image_width = -1
    expected_image_height = -1
elif len(sys.argv) == 5:
    target_directory = Path(sys.argv[1])
    expected_file_extension = sys.argv[2]
    expected_image_width = int(sys.argv[3])
    expected_image_height = int(sys.argv[4])
else:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make quality_control ARGS='<directory_to_search> <expected_file_extension>"
          " (optional) <expected_image_width> <expected_image_height>'")
    quit()

image_hashes = {}
incorrect_extension_files = []
incorrect_dimension_files = []

# Verify target directory
traverse_directory(target_directory, image_hashes, expected_file_extension,
                   (expected_image_width, expected_image_height),
                   incorrect_extension_files, incorrect_dimension_files)
find_duplicates(image_hashes)

# Display non-standard files
if len(incorrect_extension_files) != 0:
    print(f"\nFiles with incorrect extension ({expected_file_extension} expected):")
    for filepath in incorrect_extension_files:
        print(f"  {filepath}")
else:
    print(f"All files have correct extension ({expected_file_extension})")

if expected_image_width > 0:
    if len(incorrect_dimension_files) != 0:
        print(f"\nImages with incorrect dimensions ({expected_image_width}x{expected_image_height} expected):")
        for img in incorrect_dimension_files:
            print(f"  {img[0].resolve()}: {img[1]}")
    else:
        print(f"All images have expected dimensions ({expected_image_width}x{expected_image_height})")
else:
    print("Did not check for image sizes")
