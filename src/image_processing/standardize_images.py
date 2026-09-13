import sys
from pathlib import Path
import numpy as np
from utilities.directory_handling import verify_directory
import cv2
from matplotlib import pyplot as plt
from adjust_size import resize_image
from adjust_contrast import normalize_range, transform_contrast_sigmoid
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial


def standardize_image(image_path, target_width, target_height):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Non image at {image_path}")
        quit()
    if len(image.shape) == 3:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    # f, plot = plt.subplots(1, 3)
    # image_resized = resize_image(image, resize_width, resize_height)
    # plot[0].imshow(image_resized, cmap="gray", vmin=0, vmax=65535)

    image = normalize_range(image)

    # image_resized = resize_image(image, resize_width, resize_height)
    # plot[1].imshow(image_resized, cmap="gray", vmin=0, vmax=65535)

    # image = transform_contrast_sigmoid(image)
    image = resize_image(image, target_width, target_height)

    # plot[2].imshow(image, cmap="gray", vmin=0, vmax=65535)
    # plt.show()
    # plt.close()


    assert len(image.shape) == 2
    assert image.dtype == np.uint16

    cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("ERROR: Incorrect arguments provided\n"
              "Usage: make standardize_images ARGS='<images_directory> <target_width> <target_height>'")
        quit()

    target_directory = Path(sys.argv[1]).resolve()
    resize_width = int(sys.argv[2])
    resize_height = int(sys.argv[3])

    verify_directory(target_directory, 0)

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
        target_width=resize_width,
        target_height=resize_height
    )

    worker_count = os.cpu_count() or 1
    print(f"Using {worker_count} workers.")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        executor.map(standardize_image_partial, image_paths)

    print("Finished standardizing images.")

