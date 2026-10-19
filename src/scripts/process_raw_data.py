import sys
from pathlib import Path
from utilities.directory_handling import verify_directory
from data_processing.image_processing import standardize_images, adjust_size
from segmentation import generate_masks, normalize_masks, apply_masks
import cv2
import numpy as np

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("ERROR in process_raw_data. Incorrect arguments provided.\n"
              "Usage: python -m *.process_raw_data <images_directory> <output_directory> <final_height> <final_width")
        quit(1)

    # Argument management
    try:
        target_directory = Path(sys.argv[1]).resolve()
        verify_directory(target_directory, 0)
        save_directory = Path(sys.argv[2]).resolve()
        final_height = int(sys.argv[3])
        final_width = int(sys.argv[4])
    except Exception as e:
        print(f"ERROR in process_raw_data. Incorrect arguments provided (Exception: {e}).\n"
              "Usage: python -m *.process_raw_data <images_directory> <output_directory> <final_height> <final_width")
        quit(1)

    if save_directory.exists() and len(list(save_directory.iterdir())) > 0:
        print(f"WARNING in process_raw_data.\n"
              f"Save directory {save_directory} already exists and has contents.\n"
              f"Continuing risks overwriting some folders. It is strongly advised to create a new directory for this job.\n")
        answer = input(f"Continue with job in current save directory? (Y/n): ")
        if answer.upper() == "Y":
            answer = input(f"Please type yes to confirm: ")

        if answer.upper() not in ["YES"]:
            print("Exiting...")
            quit()


    # Setup working directories
    verify_directory(target_directory, 2)

    standard_imgs_dir = str(save_directory) + "/images"
    verify_directory(standard_imgs_dir, 2)

    default_mask_dir = str(save_directory) + "/masks/default/"
    verify_directory(default_mask_dir, 2)

    normalized_mask_dir = str(save_directory) + "/masks/normalized/"
    verify_directory(normalized_mask_dir, 2)

    # segmented_images_dir = str(save_directory) + "/segmented/"
    # verify_directory(segmented_images_dir, 2)


    # Standardize images to contrast corrected 512x512
    standardize_images.multiprocess_standardization_call(target_directory=target_directory,
                                                         save_directory=standard_imgs_dir,
                                                         resize_width=512,
                                                         resize_height=512)

    # QC here to check if all images correctly resized
    print("Performing QC on standardized_images")
    for file in Path(standard_imgs_dir).iterdir():
        if not str(file).endswith(".png"):
            print(f"Non png file found at {file}")
            continue

        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if not img.shape == (512, 512) or not img.dtype == np.uint16 or not img.ndim == 2:
            print(f"Reprocessing image {file}")
            standardize_images.standardize_image(file, standard_imgs_dir, 512, 512)



    # Generate masks
    print("Generating masks")
    generate_masks._external_call_generate_masks(standard_imgs_dir, default_mask_dir)

    # Normalize masks
    print("Normalizing masks")
    normalize_masks._external_call_normalize_mask(default_mask_dir, normalized_mask_dir)

    # Apply masks
    print("Applying masks")
    apply_masks._external_call_apply_masks(standard_imgs_dir, normalized_mask_dir, standard_imgs_dir)

    # Rescale segmented images
    print("Resizing images")
    adjust_size._external_call_resize_images(standard_imgs_dir, final_width, final_height)

    print("Finished processing images.")