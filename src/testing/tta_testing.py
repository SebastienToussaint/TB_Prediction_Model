# Create new images by applying multiple transforms to existing images, including translation, rotation, mirroring etc.
# Note: Currently only applies transformation to .png images

from utilities.directory_handling import verify_directory
from pathlib import Path
import cv2
import numpy as np
import sys
from scipy.stats import truncnorm

# Global number of images transformed variable
num_images_transformed = 1

def get_random_value_normal(mean, lower_bound, upper_bound, standard_deviation):
    """
    Generate a random value on a normal distribution curve between given bounds
    """
    lower_bound /= standard_deviation
    upper_bound /= standard_deviation
    return int(truncnorm.rvs(lower_bound, upper_bound, mean, standard_deviation, 1)[0])


def translate_image(img: cv2.typing.MatLike, x_shift_range, y_shift_range):
    """
    Randomly shift image in x and y direction within given parameters
    :return: Translated Image
    """
    # Choose a random x and y amounts to shift the image
    x_shift = get_random_value_normal(mean=(x_shift_range[0] + x_shift_range[1]) / 2,
                                                   lower_bound=x_shift_range[0],
                                                   upper_bound=x_shift_range[1],
                                                   standard_deviation=max(x_shift_range) / 2)

    y_shift = get_random_value_normal(mean=(y_shift_range[0] + y_shift_range[1]) / 2,
                                                   lower_bound=y_shift_range[0],
                                                   upper_bound=y_shift_range[1],
                                                   standard_deviation=max(y_shift_range) / 2)

    # Set up shift matrix
    shift_matrix = np.float32([[1, 0, int(x_shift)],
                               [0, 1, int(y_shift)]])

    height, width = img.shape[:2]
    dimensions = (int(width), int(height))

    # Return shifted image
    return cv2.warpAffine(src=img, M=shift_matrix, dsize=dimensions)


def rotate_image(img: cv2.typing.MatLike, rotation_range):
    """
    Randomly rotate image within given parameters
    :return: Rotated Image
    """
    height, width = img.shape[:2]
    dimensions = (int(width), int(height))
    center = (width // 2, height // 2)

    rotation_amount = get_random_value_normal(mean=(rotation_range[0] + rotation_range[1]) / 2,
                                              lower_bound=rotation_range[0],
                                              upper_bound=rotation_range[1],
                                              standard_deviation=max(rotation_range) / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_amount, 1)

    return cv2.warpAffine(img, rotation_matrix, dimensions)


def transform_image_affine(filepath, savefile_path, num_files_total,
                    x_shift_range: list, y_shift_range: list,rotation_range: list):
    """
    Takes an image and applies to it affine transformations within the given parameters
    :param filepath: path to the directory of images to be transformed
    :param savefile_path: Folder to which transformed images should be saved to
    :param num_files_total: total number of files in directory
    :param x_shift_range: minimum and maximum values that the image may be shifted in the x direction
    :param y_shift_range: minimum and maximum values that the image may be shifted in the y direction
    :param rotation_range: minimum and maximum angle that image may be rotated
    :return:
    """

    global num_images_transformed

    # Open image and get information
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_name = filepath[filepath.rindex("/") + 1:]
    print(f"Transforming {img_name} ({num_images_transformed}/{num_files_total})")

    # Translation
    img = translate_image(img, x_shift_range, y_shift_range)

    # Rotation
    img = rotate_image(img, rotation_range)

    # Create finished filename and save
    transformed_img_name = img_name[:img_name.rindex(".")] + "_transformed" + img_name[img_name.rindex("."):]
    cv2.imwrite(savefile_path + "/" + transformed_img_name, img)

    num_images_transformed += 1


def mirror_image(image: cv2.typing.MatLike):
    """
    Mirror image along y-axis
    :return: Mirrored Image
    """
    return cv2.flip(image, 1)


# Look into usefulness/feasibility of this
def transform_image_aspect_ratio():
    pass


def transform_image_contrast():
    # Look at sigmoid based contrast adjustments. Better for x-rays to keep features intact
    pass


if len(sys.argv) != 3:
    print("Error: Incorrect usage\n"
          "Usage: make tta_testing ARGS='<target_directory> <save_directory>'")
    quit()

images_directory = Path(sys.argv[1])
save_directory = Path(sys.argv[2])

verify_directory(images_directory, 0)
verify_directory(save_directory, 1)

# Get number of files in directory
file_count = 0
for file in images_directory.iterdir():
    if file.is_file():
        file_count += 1

# Apply transforms to images
for file in images_directory.iterdir():
    if file.name.endswith(".png"):
        transform_image_affine(filepath=f"{images_directory}/{file.name}",
                               savefile_path=f"{save_directory}",
                               num_files_total=file_count,
                               x_shift_range=[-250, 250],
                               y_shift_range=[-100, 100],
                               rotation_range=[-10, 10])

print(f"\nSuccessfully transformed all images in {images_directory.absolute()}/\nTransformed images have been saved to {save_directory.absolute()}/")
