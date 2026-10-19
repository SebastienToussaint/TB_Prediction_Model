import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from utilities.directory_handling import verify_directory
import cv2
from data_processing.image_processing.adjust_size import resize_image
from data_processing.image_processing.adjust_contrast import normalize_range
from utilities.string_manipulation import extract_filename_from_path

def normalize_dtype(img):
    if img.dtype == np.uint16:
        return img
    elif img.dtype == np.uint8:
        return img.astype(np.uint16) * 257
    else:
        print(f"Unknown dtype for image {img.dtype}")
        quit()


def get_image_mask(model, img, target_indices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert img.ndim == 2
    assert img.dtype == np.uint16

    # Normalise to required range
    image = xrv.utils.normalize(
        img,
        maxval=65535,
        reshape=True
    ).astype(np.float32)

    # Convert image to 512x512 torch tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(image_tensor)

    # Get masks
    target_1 = output[0, target_indices[0]]
    combined_scores = target_1
    if len(target_indices) > 1:
        for idx in target_indices[1:]:
            combined_scores = torch.maximum(combined_scores, output[0, idx])

    # Create mask
    mask = (combined_scores > -0.5).float()

    # Resize the binary mask to match the original image.
    mask = F.interpolate(
        mask[None, None],
        size=img.shape,
        mode="nearest"
    )[0, 0].cpu().numpy().astype(np.uint8)

    return mask


def generate_masks(model, targets: list, image_path, output_directory):
    # SET TARGETS

    available_targets = list(model.targets)
    # Possible targets:
    # ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula', 'Left Lung', 'Right Lung', 'Left Hilus Pulmonis',
    # 'Right Hilus Pulmonis', 'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum', 'Weasand', 'Spine']
    if len(targets) == 0:
        print("Error while segmenting images.\n"
              "No targets selected\n"
              f"Available targets: {available_targets}")

    target_indices = []

    for target in targets:
        if target not in available_targets:
            print(f"Error in segment_images().\n"
                  f"{target} is not an available target.\n"
                  f"Available targets: {available_targets}")
            quit()

        target_indices.append(available_targets.index(target))

    if not str(image_path).endswith(".png"):
        return

    # LOAD AND NORMALIZE IMAGE
    print(f"Processing {image_path}...")
    raw_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if raw_image.ndim != 2:
        raw_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if raw_image is None:
        print(f"Could not load image {image_path}...")
        quit(1)

    # masking_image = normalize_range(raw_image)
    masking_image = resize_image(raw_image, 512, 512)

    assert masking_image.dtype == np.uint16
    assert masking_image.shape == (512, 512)
    assert 255 < masking_image.max()

    # GET MASK
    mask = get_image_mask(model, masking_image, target_indices)

    # APPLY MASK
    low, high = np.percentile(masking_image, [1, 99])

    masking_image = np.clip(
        (masking_image.astype(np.float32) - low)
        / (high - low + 1e-8),
        0, 1
    )

    masked_image = ((masking_image * mask) * 65535).astype(np.uint16)

    assert masked_image.ndim == 2
    assert masked_image.dtype == np.uint16

    # Convert mask for saving
    mask = (mask*255).astype(np.uint8)

    # Normalize standard image for saving
    standard_image = normalize_dtype(raw_image)
    standard_image = resize_image(standard_image, 512, 512)

    cv2.imwrite(str(output_directory) + "/" + extract_filename_from_path(image_path, 0),
                mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # cv2.imwrite(str(output_directory) + "/originals/" + extract_filename_from_path(image_path, 0),
    #             standard_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Compare original and mask
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axes[0].imshow(standard_image, cmap="gray")
    # axes[0].set_title("Original X-ray")
    # axes[1].imshow(mask, cmap="gray")
    # axes[1].set_title("Mask")
    # axes[2].imshow(masked_image, cmap="gray", vmin=0, vmax=65535)
    # axes[2].set_title("Mask applied to X-ray")
    #
    # plt.tight_layout()
    # plt.show()


def _external_call_generate_masks(images_directory, output_directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Segmenting with {device}")

    # Load model
    model = xrv.baseline_models.chestx_det.PSPNet().to(device)
    model.eval()

    for file in Path(images_directory).iterdir():
        generate_masks(model, ["Left Lung", "Right Lung", "Spine", "Facies Diaphragmatica", "Heart"], file, output_directory)

    print(f"Segments succesfully saved at {output_directory}")

if __name__ == "__main__":

    # Variable management
    search_directory = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    verify_directory(output_dir, 2)

    verify_directory(str(output_dir), 2)
    # verify_directory(str(output_dir) + "/originals/", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Segmenting with {device}")

    # Load model
    model = xrv.baseline_models.chestx_det.PSPNet().to(device)
    model.eval()


    for file in search_directory.iterdir():
        generate_masks(model, ["Left Lung", "Right Lung", "Spine", "Facies Diaphragmatica", "Heart"], file, output_dir)
