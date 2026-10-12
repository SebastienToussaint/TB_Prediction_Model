# Create new images by applying multiple transforms to existing images, including translation, rotation, mirroring etc.
# Note: Currently only applies transformation to .png images
import time
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial
import torch
from utilities.directory_handling import verify_directory
from pathlib import Path
import cv2
import numpy as np
import sys
from scipy.stats import truncnorm


def get_random_value_normal(mean, lower_bound, upper_bound, standard_deviation):
    """
    Generate a random value on a normal distribution curve between given bounds
    """
    while True:
        value = np.random.normal(mean, standard_deviation)
        if lower_bound <= value <= upper_bound:
            return value


def translate_image(img: cv2.typing.MatLike, x_shift_range, y_shift_range):
    """
    Randomly shift image in x and y direction within given parameters
    :return: Translated Image
    """
    # Choose a random x and y amounts to shift the image
    x_shift = get_random_value_normal(mean=(x_shift_range[0]+x_shift_range[1])/2,
                                      lower_bound=x_shift_range[0],
                                      upper_bound=x_shift_range[1],
                                      standard_deviation=(x_shift_range[1]-x_shift_range[0])/4)

    y_shift = get_random_value_normal(mean=(y_shift_range[0] + y_shift_range[1]) / 2,
                                      lower_bound=y_shift_range[0],
                                      upper_bound=y_shift_range[1],
                                      standard_deviation=(y_shift_range[1] - y_shift_range[0]) / 4)

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


def transform_image_affine(filepath, savefile_path,
                    x_shift_range: list, y_shift_range: list,rotation_range: list):
    """
    Takes an image and applies to it affine transformations within the given parameters
    :param filepath: path to the directory of images to be transformed
    :param savefile_path: Folder to which transformed images should be saved to
    :param x_shift_range: minimum and maximum values that the image may be shifted in the x direction
    :param y_shift_range: minimum and maximum values that the image may be shifted in the y direction
    :param rotation_range: minimum and maximum angle that image may be rotated
    :return:
    """

    # Open image and get information
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_name = filepath[filepath.rindex("/") + 1:]

    # Translation
    img = translate_image(img, x_shift_range, y_shift_range)

    # Rotation
    img = rotate_image(img, rotation_range)

    # Create finished filename and save
    transformed_img_name = img_name[:img_name.rindex(".")] + "_transformed" + img_name[img_name.rindex("."):]

    cv2.imwrite(savefile_path + "/" + transformed_img_name, img)

def mirror_image(image: cv2.typing.MatLike):
    """
    Mirror image along y-axis
    :return: Mirrored Image
    """
    return cv2.flip(image, 1)


# TODO: This function
def transform_image_contrast():
    # Look at sigmoid based contrast adjustments. Better for x-rays to keep features intact
    pass


def transform_pixel_array_affine(batch, x_shift_percentage_range: list, y_shift_percentage_range: list,rotation_range: list):


    if not -1<= x_shift_percentage_range[0] <= 1 or not -1<= x_shift_percentage_range[1] <= 1:
        print("ERROR in transform_pixel_array_affine(). Bad shift range given.\n"
              f"Given x_shift_percentage_range of {x_shift_percentage_range} has a value outside of [-1,1],\n"
              f"which will lead to some images getting shifted completely off screen.")

    if not -1<= y_shift_percentage_range[0] <= 1 or not -1<= y_shift_percentage_range[1] <= 1:
        print("ERROR in transform_pixel_array_affine(). Bad shift range given.\n" 
              f"Given x_shift_percentage_range of {y_shift_percentage_range} has a value outside of [-1,1],\n"
              f"which will lead to some images getting shifted completely off screen.")


    x_shift_range = [x_shift_percentage_range[0]*len(batch[0][0][0]), x_shift_percentage_range[1]*len(batch[0][0][0])]
    y_shift_range = [y_shift_percentage_range[0]*len(batch[0][0]), y_shift_percentage_range[1]*len(batch[0][0])]

    # cv2.imshow("1", batch[0].numpy().squeeze(0))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Translation
    img = translate_image(batch[0][0].numpy(), x_shift_range, y_shift_range)
    # Rotation
    img = rotate_image(img, rotation_range)

    # cv2.imshow("2", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return [torch.Tensor(img).unsqueeze(0), batch[1], "_-_TRANSFORMED_-_" + batch[2]]


def augment_data_from_dataloader(initial_data, num_augments, x_shift_percentage_range: list, y_shift_percentage_range: list,rotation_range: list):
    data = []

    # worker_count = 2
    # print(f"Using {worker_count} workers.")
    #
    # tpa_a_partial = partial(
    #     transform_pixel_array_affine,
    #     x_shift_range=x_shift_range,
    #     y_shift_range=y_shift_range,
    #     rotation_range=rotation_range
    # )
    #
    # with ProcessPoolExecutor(max_workers=worker_count) as executor:
    #         augmented_data = list(executor.map(tpa_a_partial, initial_data))

    for batch in initial_data:
        for k in range(num_augments):
            data.append(transform_pixel_array_affine(batch,
                        x_shift_percentage_range=x_shift_percentage_range,
                        y_shift_percentage_range=y_shift_percentage_range,
                        rotation_range=rotation_range))

    return data




if __name__ == "__main__":
    time1 = time.time()
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

    # Get all paths
    image_paths = list(images_directory.iterdir())

    # Filter out non-pngs
    i = 0
    while i < len(image_paths):
        if not str(image_paths[i]).endswith(".png"):
            print(f"Non png filepath found at {image_paths[i]}")
            image_paths.pop(i)
        else:
            i += 1

    transform_affine_partial = partial(
        transform_image_affine,
        savefile_path=f"{save_directory}",
        num_files_total=file_count,
        x_shift_range=[-250, 250],
        y_shift_range=[-100, 100],
        rotation_range=[-10, 10]
    )

    worker_count = os.cpu_count() or 1
    print(f"Using {worker_count} workers.")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
            executor.map(transform_affine_partial, image_paths)

    print(f"\nSuccessfully transformed all images {file_count} in {images_directory.absolute()}/\nTransformed images have been saved to {save_directory.absolute()}/")

    time2 = time.time()

    print(f"Took {time2-time1}")