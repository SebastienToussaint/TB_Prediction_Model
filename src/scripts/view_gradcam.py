import sys

from utilities.directory_handling import verify_directory
from shared_objects.models import CNN
from shared_objects import models
from shared_objects.gradcam_visualization import visualize_gradcam
from shared_objects.data_handling import DataStructure, CustomDataset
import torch
import matplotlib.pyplot as plt
from pathlib import Path



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("ERROR in view_gradcam. Incorrect arguments provided.\n"
              "Usage: pythom -m *.view_gradcam <path_to_model> <path_to_dataloaders> <target_layer>")
        quit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.CNN()
    model.load(str(Path(sys.argv[1]).resolve()))
    model.eval()
    model.to(device)

    model.print_layer_stack()


    verify_directory(Path(sys.argv[2]).resolve(), 0)

    ds = DataStructure()
    ds.set_native_datapath(str(Path(sys.argv[2]).resolve()))
    dl = ds.get_next_dataloader()

    try:
        target_layer = model.layers[int(sys.argv[3])] # 34

    except ValueError:
        print("ERROR in view_gradcam. Incorrect argument provided.\n"
              "target_layer needs to be an integer")
        quit()



    for images, labels, filenames in dl:
        for i, image in enumerate(images):
            label = labels[i]

            with torch.no_grad():
                prob = torch.sigmoid(model(image.unsqueeze(0).to(device))).item()

            pred = int(prob>0.5)

            heatmap, overlay = visualize_gradcam(model,image, target_layer, positive=bool(pred))

            disp = image.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(disp, cmap="gray")
            axes[0].set_title("Original CXR")
            axes[0].axis("off")
            # axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
            axes[1].imshow(overlay)
            axes[1].set_title("Heatmap overlay")
            axes[1].axis("off")
            # fig.suptitle(f"{utilities.string_manipulation.extract_filename_from_path(filenames[i], 0)} | "
            #              f"label={label} pred={pred}| P(Positive)={prob:.3f}")

            plt.show()

            plt.close(fig)