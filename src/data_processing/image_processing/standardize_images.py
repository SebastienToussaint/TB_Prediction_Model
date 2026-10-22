import sys
from os import cpu_count
from pathlib import Path
import numpy as np
from utilities.directory_handling import verify_directory
import cv2
from data_processing.image_processing.adjust_size import resize_image
from data_processing.image_processing.adjust_contrast import normalize_range, transform_contrast_sigmoid
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial
from utilities.string_manipulation import extract_filename_from_path
import matplotlib.pyplot as plt


def standardize_image(image_path, save_path, target_width, target_height):

    # Read in image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Non image at {image_path}")
        quit()
    if len(image.shape) == 3:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    # Normalize pixel range to uint16
    normalized_image = normalize_range(image)

    # Apply sigmoid transformation to image
    contrasted_image = transform_contrast_sigmoid(normalized_image)

    # Resize image
    resized_image = resize_image(contrasted_image, target_width, target_height)

    # f, plot = plt.subplots(1, 3)
    # img1 = resize_image(image, resize_width, resize_height)
    # plot[0].imshow(img1, cmap="gray", vmin=0, vmax=img1.max())
    #
    # img2 = resize_image(normalized_image, resize_width, resize_height)
    # plot[1].imshow(img2, cmap="gray", vmin=0, vmax=65535)
    #
    # plot[2].imshow(resized_image, cmap="gray", vmin=0, vmax=65535)
    # plt.show()
    # plt.close()

    assert len(resized_image.shape) == 2
    assert resized_image.dtype == np.uint16

    cv2.imwrite(str(save_path) + "/" + extract_filename_from_path(image_path, 1) + ".png", resized_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def multiprocess_standardization_call(target_directory, save_directory, resize_width, resize_height, num_threads = None):
    print("Standardizing images...")
    # Get all paths
    image_paths = list(target_directory.iterdir())

    # Filter out non-pngs
    i = 0
    while i < len(image_paths):
        if not str(image_paths[i]).endswith(".png"):
            print(f"Non png filepath found at {image_paths[i]}")
            image_paths.pop(i)
        else:
            i += 1

    standardize_image_partial = partial(
        standardize_image,
        save_path=save_directory,
        target_width=resize_width,
        target_height=resize_height
    )


    if num_threads is None:
        worker_count = os.cpu_count() or 1
    else:
        worker_count = min(os.cpu_count(), num_threads)


    worker_count = min(round(len(image_paths)/10), worker_count)
    if worker_count == round(len(image_paths)/10):
        print(f"Workers set to {worker_count} to allow for at least 10 images per worker")

    if worker_count == 0:
        worker_count = 1

    print(f"Using {worker_count} workers.")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        executor.map(standardize_image_partial, image_paths)

    # for path in image_paths:
    #     standardize_image(path, save_directory, resize_width, resize_height)


    # Quality control because multithreading sometimes leaves one or two unstandardized
    print("Performing QC on standardized_images")
    for file in Path(save_directory).iterdir():
        if not str(file).endswith(".png"):
            print(f"Non png file found at {file}")
            continue

        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if not img.shape == (resize_height, resize_width) or not img.dtype == np.uint16 or not img.ndim == 2:
            print(f"Reprocessing image {file}")
            standardize_image(file, save_directory, resize_width, resize_height)

    print("Finished standardizing images.\n"
          f"Standardized images saved in {save_directory}")

if __name__ == "__main__":
    if len(sys.argv) not in [5, 7] :
        print("ERROR: Incorrect arguments provided\n"
              "Usage: make standardize_images ARGS='<images_directory> <save_directory> <target_width> <target_height> OPTIONAL -t <num_threads>")
        quit()

    target_dir = Path(sys.argv[1]).resolve()
    save_dir = Path(sys.argv[2]).resolve()
    resize_w = int(sys.argv[3])
    resize_h = int(sys.argv[4])

    verify_directory(target_dir, 0)
    verify_directory(target_dir, 2)

    number_of_threads = None
    if len(sys.argv) == 7:
        if not sys.argv[5].upper() == "-T":
            print("ERROR in process_raw_data. Incorrect arguments provided.\n"
                  "Usage: python -m *.process_raw_data <images_directory> <output_directory> <final_height> <final_width> OPTIONAL -t <num_threads>")
            quit(1)
        else:
            try:
                number_of_threads = int(sys.argv[6])
            except ValueError:
                print("ERROR in process_raw_data. Incorrect arguments provided.\n"
                      f"Num_threads value following -t flag needs to be an integer.")
                quit(1)

    multiprocess_standardization_call(target_dir, save_dir, resize_w, resize_h, number_of_threads)
