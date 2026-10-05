import os
import sys
from pathlib import Path

import torch

from data_processing.image_processing.adjust_size import resize_image
from data_processing.image_processing.adjust_contrast import normalize_range, transform_contrast_sigmoid
import cv2
import numpy as np
from shared_objects import models

path_to_model = sys.argv[1]
path_to_images = Path(sys.argv[2]).resolve()
width = int(sys.argv[3])
height = int(sys.argv[4])


def _standardize_image_return(image_path, target_width, target_height):

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

    assert len(resized_image.shape) == 2
    assert resized_image.dtype == np.uint16

    return resized_image


good_images = []
bad_images = []

model = models.CNN()
model.load(path_to_model)

model = model.to("cuda")

for path in path_to_images.iterdir():
    print(f"Processing {path}")
    img = _standardize_image_return(path, width, height)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).unsqueeze(0)

    img = img.to("cuda")

    model.eval()
    with torch.inference_mode():
        inferred_logit = model(img).squeeze()
        inferred_prediction_value = torch.sigmoid(inferred_logit)
        prediction = torch.round(inferred_prediction_value)

    if prediction == 0:
        good_images.append(path)

    if prediction == 1:
        bad_images.append(path)

print(f"Number of bad images: {len(bad_images)}")

answer = input("Would you like to remove all images inferred as bad? Y/n: ")

if answer.upper() not in ["YES", "Y"]:
    print("Exiting")
    exit()


print("Removing images...")
for path in bad_images:
    print(f"Removing {path}")
    os.remove(path)