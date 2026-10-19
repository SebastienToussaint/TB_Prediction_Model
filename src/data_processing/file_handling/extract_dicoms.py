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


def downscale_image_no_padding(img, max_width, max_height):
    height, width = img.shape[:2]

    # Determine largest rescale factor that keeps aspect ratio
    rescale_factor = min(max_width / width, max_height / height)

    final_width = int(width * rescale_factor)
    final_height = int(height * rescale_factor)

    if rescale_factor >= 1:
        return img

    img = cv2.resize(img, [final_width, final_height], interpolation=cv2.INTER_AREA)

    return img


def extract_and_save_image(filepath, save_directory):
    """
    Extracts the image from the dicom file at the given path.
    Image gets standardized to uint16 and saved to the save directory,
    keeping its original name
    """
    dcm_file = pydicom.dcmread(filepath)

    image = dcm_file.pixel_array.astype(np.float32)

    low = float(image.min())
    high = float(image.max())


    # Invert pixel state if needed
    if str(dcm_file.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = (high + low) - image


    image = np.clip((image - low) / (high - low)*65535.0, 0, 65535)

    # Convert from float32 to uint16
    image = np.round(image).astype(np.uint16)

    image = downscale_image_no_padding(image, 1000, 1000)

    # Save the image
    save_path = str(save_directory) + "/" + extract_filename_from_path(filepath, 1) + ".png"
    cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Quality control check
    saved_image = cv2.imread(save_path, cv2.IMREAD_UNCHANGED)
    assert saved_image.dtype == np.uint16
    assert np.array_equal(saved_image, image)

if __name__ == "__main__":
    # Argument management
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

    # Setup multithreading partial function
    extract_dicom_partial = partial(
        extract_and_save_image,
        save_directory=f"{extraction_directory}",
    )


    # for path in dicom_paths:
    #     extract_and_save_image(path, "")
    worker_count = os.cpu_count() or 1
    print(f"Using {worker_count} workers.")

    # Call extraction function
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        executor.map(extract_dicom_partial, dicom_paths)


    print(f"Extracted files saved at {str(extraction_directory)}")
