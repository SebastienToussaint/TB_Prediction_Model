from pathlib import Path
from typing import final

import cv2
import sys
from utilities.directory_handling import verify_directory

def standardize_image_size(image_path, target_width, target_height):
    """
    Rescale given image while maintaining aspect ratio. Pads border with black pixels.
    Saves rescaled image to given path
    """
    print(f"Processing {image_path}...")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # TODO: Check if unchanged is correct

    if image is None: # Prevents unexpected type warnings
        print(f"Could not load image {image_path}")
        return

    height, width = image.shape[:2]

    # Determine largest rescale factor that keeps aspect ratio
    rescale_factor = min(target_width/width, target_height/height)

    final_width = int(width * rescale_factor)
    final_height = int(height * rescale_factor)


    if 1 > rescale_factor:
        interpolation_strategy = cv2.INTER_AREA # TODO: Check if these are correct
    else:
        interpolation_strategy = cv2.INTER_LANCZOS4


    image = cv2.resize(image, [final_width, final_height], interpolation=interpolation_strategy)

    # Padding
    x_padding = (target_width - final_width)//2
    y_padding = (target_height - final_height)//2

    x_padding_filler = target_width - final_width - 2*x_padding
    y_padding_filler = target_height - final_height - 2*y_padding

    image = cv2.copyMakeBorder(image,
                               y_padding + y_padding_filler, y_padding,
                               x_padding + x_padding_filler, x_padding,
                               cv2.BORDER_CONSTANT, value=0)

    cv2.imwrite(image_path, image, (cv2.IMWRITE_PNG_STRATEGY, 0)) # TODO: Check for alternate writing strategies

if len(sys.argv) != 4:
    print("ERROR: Incorrect arguments provided\n"
          "Usage: make standardize_images ARGS='<images_directory> <save_directory> <target_width> <target_height>'")
    quit()

target_directory = Path(sys.argv[1])
resize_width = int(sys.argv[2])
resize_height = int(sys.argv[3])

verify_directory(target_directory, 0)

for file in target_directory.iterdir():
    if file.name.endswith(".png"):
        standardize_image_size(f"{target_directory}/{file.name}",
                               resize_width, resize_height)

print("Done.")