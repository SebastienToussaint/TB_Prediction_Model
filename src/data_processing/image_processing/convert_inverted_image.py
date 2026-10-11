import sys
import cv2

import os
import sys
from pathlib import Path

import torch

from data_processing.image_processing.adjust_size import resize_image
from data_processing.image_processing.adjust_contrast import normalize_range, transform_contrast_sigmoid
import cv2
import numpy as np
from shared_objects import models


def invert_image(img_to_invert):
    for i in range(len(img_to_invert)):
        for j in range(len(img_to_invert[i])):
            if not img_to_invert[i][j] == 0:
                img_to_invert[i][j] = 65535-img_to_invert[i][j]

    return img_to_invert



if __name__ == "__main__":
    path_to_model = sys.argv[1]
    path_to_images = Path(sys.argv[2]).resolve()

    good_images = []
    bad_images = []

    model = models.CNN()
    model.load(path_to_model)

    model = model.to("cuda")

    for path in path_to_images.iterdir():
        print(f"Processing {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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


    print("Inverting images...")
    for path in bad_images:
        cv2.imshow("C", invert_image(cv2.imread(path, cv2.IMREAD_UNCHANGED)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite(path, invert_image(cv2.imread(path, cv2.IMREAD_UNCHANGED)), [cv2.IMWRITE_PNG_COMPRESSION, 0])