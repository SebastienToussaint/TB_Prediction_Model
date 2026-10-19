import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utilities.directory_handling import verify_directory
from utilities.string_manipulation import extract_filename_from_path


def smooth_mask(mask, close_fraction, dilate_fraction, use_hull=True):

    mask = (mask > 0).astype(np.uint8)

    og_mask = mask.copy()

    num_segments, labels, stats, overload = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_segments <= 1: # Blank mask
        return None


    keep = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:2] + 1

    mask = np.isin(labels, keep).astype(np.uint8)

    s = mask.shape[0]
    k = int(close_fraction*s) or 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    if use_hull:
        points = cv2.findNonZero(mask)
        hull = cv2.convexHull(points)
        mask = cv2.fillConvexPoly(np.zeros_like(mask), hull, 1)

    k = int(dilate_fraction * s) or 1

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    feathered_mask = cv2.GaussianBlur(mask.astype(np.float32), (0,0), 3)


    mask = mask.astype(np.uint8)*255

    feathered_mask = np.clip(np.round(feathered_mask * 255), 0, 255).astype(np.uint8)

    # print(num_segments)
    # print(labels)
    # print(stats)
    # print(overload)
    #
    # print(mask.shape)
    # print(mask.dtype)
    # print(mask.min())
    # print(mask.max())

    # # Compare original and mask
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axes[0].imshow(og_mask, cmap="gray")
    # axes[0].set_title("Original Mask")
    # axes[1].imshow(mask, cmap="gray")
    # axes[1].set_title("Normalized Mask")
    # axes[2].imshow(feathered_mask, cmap="gray")
    # axes[2].set_title("Feathered Mask")
    # plt.show()
    # plt.close(fig)

    return feathered_mask

def normalize_mask(image_path, out_path):
    loaded_mask = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    smoothed = smooth_mask(loaded_mask, 0.04, 0.06, True)

    if smoothed is None:
        print(f"Empty mask for {image_path}")
        quit()
    cv2.imwrite(str(out_path) + "/" + extract_filename_from_path(image_path, 0), smoothed,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])



def _external_call_normalize_mask(masks_directory, output_directory):
    for p in Path(masks_directory).iterdir():
        print(f"Processing {p}")
        normalize_mask(p, output_directory)


if __name__ == "__main__":
    mask_directory = Path(sys.argv[1]).resolve()
    verify_directory(mask_directory, 0)
    out_directory = Path(sys.argv[2]).resolve()
    verify_directory(out_directory, 2)

    for path in mask_directory.iterdir():
        print(f"Processing {path}")
        normalize_mask(path, out_directory)





