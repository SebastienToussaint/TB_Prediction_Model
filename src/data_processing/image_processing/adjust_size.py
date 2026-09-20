from pathlib import Path
import cv2
import sys
from utilities.directory_handling import verify_directory
import numpy as np

def resize_image(img, target_width, target_height):
    """
    Rescale given image while maintaining aspect ratio. Pads border with black pixels.
    """

    height, width = img.shape[:2]

    # Determine largest rescale factor that keeps aspect ratio
    rescale_factor = min(target_width/width, target_height/height)

    final_width = int(width * rescale_factor)
    final_height = int(height * rescale_factor)

    if 1 > rescale_factor:
        interpolation_strategy = cv2.INTER_AREA # TODO: Check if these are correct
    else:
        interpolation_strategy = cv2.INTER_LANCZOS4


    img = cv2.resize(img, [final_width, final_height], interpolation=interpolation_strategy)

    # Padding
    x_padding = (target_width - final_width)//2
    y_padding = (target_height - final_height)//2

    x_padding_filler = target_width - final_width - 2*x_padding
    y_padding_filler = target_height - final_height - 2*y_padding

    img = cv2.copyMakeBorder(img,
                               y_padding + y_padding_filler, y_padding,
                               x_padding + x_padding_filler, x_padding,
                               cv2.BORDER_CONSTANT, value=0)

    return img

if __name__ == "__main__":
    # Argument Management
    if len(sys.argv) != 4:
        print("ERROR: Incorrect arguments provided\n"
              "Usage: make resize_images ARGS='<images_directory> <target_width> <target_height>'")
        quit()

    target_directory = Path(sys.argv[1])
    resize_width = int(sys.argv[2])
    resize_height = int(sys.argv[3])

    verify_directory(target_directory, 0)

    # Standardize images in target directory
    for file in target_directory.iterdir():
        if not file.name.endswith(".png"):
            print(f"Non png file found at {file}")
            continue

        print(f"Processing {file}...")

        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        image = resize_image(image, resize_width, resize_height)

        cv2.imwrite(file, image,(cv2.IMWRITE_PNG_COMPRESSION, 0))

    print("Done.")
