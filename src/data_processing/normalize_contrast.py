import cv2
import numpy as np
from pathlib import Path
from utilities.directory_handling import verify_directory
import sys


if len(sys.argv) not in [2,4]:
    print(f"Error while normalizing contrast. Incorrect parameter usage.\n"
          f"Usage: make normalize_contrast <images_directory> OPTIONAL <min_pixel_cutoff> <max_pixel_cutoff>")
    quit()

target_directory = Path(sys.argv[1]).resolve()
verify_directory(target_directory, 0)

if len(sys.argv) == 4:
    try:
        percentile_min = float(sys.argv[2])
        percentile_max = float(sys.argv[3])
    except ValueError:
        print(f"Error while normalizing contrast. {sys.argv[2]} and {sys.argv[3]} are not valid parameters.\n"
              f"Usage: make normalize_contrast <images_directory> OPTIONAL <min_pixel_cutoff> <max_pixel_cutoff>")
        quit()
else:
    percentile_min = 0.5
    percentile_max = 99.5


# Iterate through directory and normalize images
for file in target_directory.iterdir():

    if not str(file).endswith(".png"):
        print(f"Non png file found at {file}")
        continue

    print(f"Processing {file}")
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    if len(image.shape) == 3:
        print("Loading with grayscale")
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    image = image.astype(np.uint16)

    assert image.dtype == np.uint16
    assert len(image.shape) == 2

    # Normalize contrast of image
    low, high = np.percentile(image, (percentile_min, percentile_max))
    image = np.clip(image, low, high)

    img_min, img_max = image.min(), image.max()

    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min) * 65535.0

    image = np.round(image).astype(np.uint16)

    cv2.imwrite(file, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])



