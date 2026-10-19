import os
import sys
from pathlib import Path
import torch
from data_processing.image_processing.adjust_size import resize_image
import cv2
import numpy as np
from shared_objects import models



if __name__ == "__main__":

    path_to_model = sys.argv[1]
    path_to_images = Path(sys.argv[2]).resolve()

    bad_images = []

    model = models.CNN()
    model.load(path_to_model)

    model = model.to("cuda")

    for path in path_to_images.iterdir():
        print(f"Processing {path}")
        img = resize_image(cv2.imread(path, cv2.IMREAD_UNCHANGED), 80, 80)

        img = img.astype(np.float32)/65535

        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).unsqueeze(0)

        img = img.to("cuda")

        model.eval()
        with torch.inference_mode():
            inferred_logit = model(img).squeeze()
            inferred_prediction_value = torch.sigmoid(inferred_logit)
            prediction = torch.round(inferred_prediction_value)

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
