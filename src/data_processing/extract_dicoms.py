import sys
import cv2
import numpy as np
import pydicom
from pathlib import Path
from utilities.directory_handling import verify_directory
from utilities.string_manipulation import extract_filename_from_path


target_directory = Path(sys.argv[1]).resolve()
save_directory = Path(sys.argv[2]).resolve()
verify_directory(target_directory, 0)
verify_directory(save_directory, 1)

# Extract dicoms
for file in target_directory.iterdir():

    if not str(file).endswith(".dcm"):
        print(f"Non dcm file found at {file}")
        continue

    print(f"Processing {file}")
    dcm_file = pydicom.dcmread(file)


    image = dcm_file.pixel_array.astype(np.uint16)

    if dcm_file.get("BitsStored") and dcm_file.get("PhotometricInterpretation"):

        # Standardize photometric type and pixel range
        max_value = (1 << dcm_file.BitsStored) -1


        image = max_value - image


        image = image.astype(np.float32)

        image = image / max_value * 65535.0

    else:

        # Attempt to standardize pixel range by using max found pixel value
        print(f"Could not find BitsStored or PhotometricInterpretation parameter for dcm file {file}\n"
              f"Using default pixel array with scaling based on maximum brightness in pixel array")

        image = image.astype(np.float32)
        image = image / image.max() * 65535.0


    image = np.round(image).astype(np.uint16)

    # Save the image
    save_path = str(save_directory) + "/" + extract_filename_from_path(file, 1) + ".png"
    cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Quality control check
    saved_image = cv2.imread(save_path, cv2.IMREAD_UNCHANGED)

    assert saved_image.dtype == np.uint16
    assert np.array_equal(saved_image, image)


print(f"Extracted files saved at {str(save_directory.resolve())}")
