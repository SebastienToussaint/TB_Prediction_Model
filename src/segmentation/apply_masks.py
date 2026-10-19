import sys
from pathlib import Path
from utilities.directory_handling import verify_directory
import cv2
from utilities.string_manipulation import extract_filename_from_path
import numpy as np
import matplotlib.pyplot as plt

def apply_mask(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    assert image.dtype == np.uint16
    assert image.ndim == 2
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    assert mask.shape == image.shape

    image, mask = square_crop(image, mask)

    mask = (mask/255).astype(np.float32)

    img = (image*mask).astype(np.uint16)


    assert img.dtype == np.uint16
    assert img.ndim == 2
    assert img.max() > 1
    # #Compare original and mask
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axes[0].imshow(image, cmap="gray")
    # axes[0].set_title("Original Image")
    # axes[1].imshow(mask, cmap="gray")
    # axes[1].set_title("Mask")
    # axes[2].imshow(img, cmap="gray")
    # axes[2].set_title("Cropped Image")
    # plt.show()
    # plt.close(fig)

    return img



def square_crop(arr, m, margin_frac=0.10):
    """Square window centred on the mask bbox; out-of-bounds filled with 0."""
    ys, xs = np.where(m > 127)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * margin_frac)) | 1
    half = side // 2
    pad = half + 1
    a = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    mm = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    y0, x0 = cy + pad - half, cx + pad - half
    return a[y0:y0 + side, x0:x0 + side], mm[y0:y0 + side, x0:x0 + side]

def _external_call_apply_masks(images_directory, masks_directory, output_directory):
    image_paths = sorted(list(Path(images_directory).iterdir()))
    mask_paths = sorted(list(Path(masks_directory).iterdir()))

    for i, img_path in enumerate(image_paths):
        if img_path.name != mask_paths[i].name:
            print("NO MATCH")
            quit()

        print(f"Processing {img_path}")

        masked_image = apply_mask(img_path, mask_paths[i])

        cv2.imwrite(str(output_directory) + "/" + extract_filename_from_path(img_path, 0), masked_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == "__main__":
    images_dir = Path(sys.argv[1]).resolve()
    verify_directory(images_dir, 0)
    masks_dir = Path(sys.argv[2]).resolve()
    verify_directory(masks_dir, 0)
    out_dir = Path(sys.argv[3]).resolve()
    verify_directory(out_dir, 2)

    image_paths = sorted(list(images_dir.iterdir()))
    mask_paths = sorted(list(masks_dir.iterdir()))

    for i, img_path in enumerate(image_paths):
        if img_path.name != mask_paths[i].name:
            print("NO MATCH")
            quit()

        print(f"Processing {img_path}")

        masked_image = apply_mask(img_path, mask_paths[i])

        cv2.imwrite(str(out_dir) + "/" + extract_filename_from_path(img_path, 0), masked_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])