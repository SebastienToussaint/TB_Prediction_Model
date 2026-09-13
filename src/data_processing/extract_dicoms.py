import sys
import cv2
import numpy as np
import pydicom
from pathlib import Path
from utilities.directory_handling import verify_directory
from utilities.string_manipulation import extract_filename_from_path
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial


def extract_and_save_image(filepath, save_directory):
    dcm_file = pydicom.dcmread(filepath)

    image = dcm_file.pixel_array.astype(np.uint16)

    if dcm_file.get("BitsStored") and dcm_file.get("PhotometricInterpretation"):

        # Standardize photometric type and pixel range
        max_value = (1 << dcm_file.BitsStored) - 1

        # Invert pixel state if needed
        if dcm_file.PhotometricInterpretation.upper() == "MONOCHROME1":
            image = max_value - image

        image = image.astype(np.float32)

        image = image / max_value * 65535.0

    else:
        # Attempt to standardize pixel range by using max found pixel value
        print(f"Could not find BitsStored or PhotometricInterpretation parameter for dcm file {filepath}\n"
              f"Using default pixel array with scaling based on maximum brightness in pixel array")

        image = image.astype(np.float32)
        image = image / image.max() * 65535.0

    image = np.round(image).astype(np.uint16)

    # Save the image
    save_path = str(save_directory) + "/" + extract_filename_from_path(filepath, 1) + ".png"
    cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Quality control check
    saved_image = cv2.imread(save_path, cv2.IMREAD_UNCHANGED)

    assert saved_image.dtype == np.uint16
    assert np.array_equal(saved_image, image)

if __name__ == "__main__":
    target_directory = Path(sys.argv[1]).resolve()
    extraction_directory = Path(sys.argv[2]).resolve()
    verify_directory(target_directory, 0)
    verify_directory(extraction_directory, 1)

    # Get all paths
    dicom_paths = list(target_directory.iterdir())

    # Filter out non-dicoms
    i = 0
    while i < len(dicom_paths):
        if not str(dicom_paths[i]).endswith(".dcm"):
            dicom_paths.pop(i)
        else:
            i += 1

    extract_dicom_partial = partial(
        extract_and_save_image,
        save_directory=f"{extraction_directory}",
    )

    worker_count = os.cpu_count() or 1
    print(f"Using {worker_count} workers.")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        executor.map(extract_dicom_partial, dicom_paths)


    print(f"Extracted files saved at {str(extraction_directory)}")
