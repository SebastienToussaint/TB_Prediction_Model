import math

import cv2
import numpy as np
from pathlib import Path
from utilities.directory_handling import verify_directory
import sys

from matplotlib import pyplot as plt



def normalize_range(img, percentile_min=0.5, percentile_max=99.5):
        img = img.astype(np.uint16)

        assert img.dtype == np.uint16
        assert len(img.shape) == 2

        # Normalize contrast of image
        low, high = np.percentile(img, (percentile_min, percentile_max))
        img = np.clip(img, low, high)

        img_min, img_max = img.min(), img.max()

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 65535.0

        img = np.round(img).astype(np.uint16)

        return img


def transform_contrast_sigmoid(img):
    def get_optimal_parameters(pixel_array): #TODO: Need to play around a bit here to see what the optimal values are
        """
        Get the optimal cutoff and gain values for sigmoid transformation of an image
        """
        # Remove absolute blacks
        pixel_array = pixel_array[pixel_array > 0.02]
        cf = float((0.5+np.median(pixel_array))/2)  #TODO: Especially for here
        percentile_5, percentile_95 = np.percentile(pixel_array, [5, 90])
        iqr = percentile_95 - percentile_5
        if iqr < 0.0001:
            iqr = 1
        gn = float(4/iqr)   #TODO: And here
        return cf, gn


    # Standardize to 0-1 range
    img = (img - img.min()) / (img.max() - img.min())

    cutoff, gain = get_optimal_parameters(img)

    # Perform sigmoid transformation
    img = 1/(1+(math.e**(gain*(cutoff-img))))

    # Standardize back to uint16
    img = (img * 65535)

    img = np.round(img).astype(np.uint16)

    return img



if __name__ == "__main__":
    if len(sys.argv) not in [2,4]:
        print(f"Error while normalizing contrast. Incorrect parameter usage.\n"
              f"Usage: make normalize_contrast <images_directory> OPTIONAL <min_pixel_cutoff> <max_pixel_cutoff>")
        quit()

    target_directory = Path(sys.argv[1]).resolve()
    verify_directory(target_directory, 0)

    if len(sys.argv) == 4:
        try:
            cutoff_min = float(sys.argv[2])
            cutoff_max = float(sys.argv[3])
        except ValueError:
            print(f"Error while normalizing contrast. {sys.argv[2]} and {sys.argv[3]} are not valid parameters.\n"
                  f"Usage: make normalize_contrast <images_directory> OPTIONAL <min_pixel_cutoff> <max_pixel_cutoff>")
            quit()
    else:
        cutoff_min = 0.5
        cutoff_max = 99.5

    for file in target_directory.iterdir():
        if not str(file).endswith(".png"):
            print(f"Non png file found at {file}")
            continue

        print(f"Processing {file}")
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        if len(image.shape) == 3:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        image = normalize_range(image, cutoff_min, cutoff_max)
        image = transform_contrast_sigmoid(image)

        cv2.imwrite(file, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print("Done.")
