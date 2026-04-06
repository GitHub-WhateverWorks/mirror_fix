import os
import glob
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


TEST_IMAGE_DIR  = "./img/MSD/MSD/test/image/"
TEST_MASK_DIR   = "./img/MSD/MSD/test/mask/"
TRAIN_IMAGE_DIR = "./img/MSD/MSD/train/image/"
TRAIN_MASK_DIR  = "./img/MSD/MSD/train/mask/"

PANEL_OUTPUT_ROOT = "./unet_eval"
MASK_OUTPUT_ROOT  = "./seg_outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
THRESHOLD = 0.3
MAX_IMAGES = 20  # set to None for all images


transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)),
])


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(DEVICE)

model.load_state_dict(torch.load("mirror_seg_baseline.pth", map_location=DEVICE))
model.eval()


def save_mask_png(mask_array: np.ndarray, out_path: str) -> None:
    """
    mask_array: uint8 array with values 0 or 255
    """
    Image.fromarray(mask_array, mode="L").save(out_path)


def run_split(split_name, image_dir, mask_dir):
    panel_dir = os.path.join(PANEL_OUTPUT_ROOT, split_name)
    mask_dir_out = os.path.join(MASK_OUTPUT_ROOT, split_name)

    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(mask_dir_out, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    print(f"[INFO] {split_name}: found {len(image_paths)} images")

    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            stem = os.path.splitext(filename)[0]
            gt_path = os.path.join(mask_dir, f"{stem}.png")

            img = Image.open(img_path).convert("RGB")
            gt = Image.open(gt_path).convert("L") if os.path.exists(gt_path) else None
            orig_w, orig_h = img.size

            x = transform(img).unsqueeze(0).to(DEVICE)
            logits = model(x)
            prob = torch.sigmoid(logits)

            prob_up = F.interpolate(
                prob,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )[0, 0].cpu().numpy().astype(np.float32)

            pred = (prob_up > THRESHOLD).astype(np.uint8) * 255

            # save usable outputs
            save_mask_png(pred, os.path.join(mask_dir_out, f"{stem}.png"))
            np.save(os.path.join(mask_dir_out, f"{stem}.npy"), prob_up)

            # keep panel output
            fig, axes = plt.subplots(1, 4 if gt is not None else 3, figsize=(16, 4))

            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(prob_up, cmap="magma")
            axes[1].set_title("Prob")
            axes[1].axis("off")

            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title("Pred")
            axes[2].axis("off")

            if gt is not None:
                axes[3].imshow(gt, cmap="gray")
                axes[3].set_title("GT")
                axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(panel_dir, f"{stem}_panel.png"))
            plt.close(fig)

            print(f"[OK] {split_name}/{stem}")


if __name__ == "__main__":
    os.makedirs(PANEL_OUTPUT_ROOT, exist_ok=True)
    os.makedirs(MASK_OUTPUT_ROOT, exist_ok=True)

    run_split("train", TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
    run_split("test", TEST_IMAGE_DIR, TEST_MASK_DIR)