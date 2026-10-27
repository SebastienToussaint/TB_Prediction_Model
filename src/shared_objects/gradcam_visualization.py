import torch.cuda
from pytorch_grad_cam import GradCAM
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
import utilities.string_manipulation
from shared_objects.data_handling import CustomDataset, DataStructure
import shared_objects.models

class BinaryTarget:
    """positive=True explains 'TB', False explains 'not TB'."""
    def __init__(self, positive=True):
        self.sign = 1.0 if positive else -1.0
    def __call__(self, model_output):
        return self.sign * model_output[0]

def visualize_gradcam(model, image, target_layer, positive=True):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = image.float()
    while x.ndim < 4:              # HxW -> 1x1xHxW, 1xHxW -> 1x1xHxW
        x = x.unsqueeze(0)
    x = x.to(device)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        heatmap = cam(input_tensor=x, targets=[BinaryTarget(positive)])[0]

    disp = image.squeeze().cpu().numpy().astype(np.float32)   # HxW, display only
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    overlay = show_cam_on_image(np.repeat(disp[..., None], 3, -1), heatmap, use_rgb=True)
    return heatmap, overlay



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = shared_objects.models.CNN()
    model.load("E:\Project\Report/figures\mask_shortcutting\Model\intervals\model_7.pkl")
    model.eval()
    model.to(device)

    model.print_layer_stack()

    ds = DataStructure()
    ds.set_native_datapath("E:\Project\Report/figures\mask_shortcutting\eval_dl")
    dl = ds.get_next_dataloader()


    target_layer = model.layers[34] # 34

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
