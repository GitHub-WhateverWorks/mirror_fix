import os
import glob
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from train_mirror_student import TinyMirrorNet

TRAIN_IMAGE_DIR = "./img/MSD/MSD/train/image/"
TRAIN_MASK_DIR  = "./img/MSD/MSD/train/mask/"
TEST_IMAGE_DIR  = "./img/MSD/MSD/test/image/"
TEST_MASK_DIR   = "./img/MSD/MSD/test/mask/"

OUTPUT_ROOT = "./eval_outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
THRESHOLD = 0.3
MAX_IMAGES = 20   # set to None for all

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)),
])

model = TinyMirrorNet().to(DEVICE)
model.load_state_dict(torch.load("mirror_student_best.pth", map_location=DEVICE))
model.eval()

def run_split(split_name, image_dir, mask_dir):
    out_dir = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    print(f"{split_name}: found {len(image_paths)} images")

    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            stem = os.path.splitext(filename)[0]
            mask_path = os.path.join(mask_dir, f"{stem}.png")

            img = Image.open(img_path).convert("RGB")
            gt = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else None
            orig_w, orig_h = img.size

            x = transform(img).unsqueeze(0).to(DEVICE)
            logits = model(x)
            prob = torch.sigmoid(logits)

            prob_up = F.interpolate(
                prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()

            pred = (prob_up > THRESHOLD).astype("uint8") * 255

            # save raw probability
            plt.imsave(os.path.join(out_dir, f"{stem}_prob.png"), prob_up, cmap="magma")

            # save binary prediction
            Image.fromarray(pred).save(os.path.join(out_dir, f"{stem}_pred.png"))

            # save side-by-side comparison
            fig, axes = plt.subplots(1, 4 if gt is not None else 3, figsize=(16, 4))

            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(prob_up, cmap="magma")
            axes[1].set_title("Prob")
            axes[1].axis("off")

            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title(f"Pred @ {THRESHOLD}")
            axes[2].axis("off")

            if gt is not None:
                axes[3].imshow(gt, cmap="gray")
                axes[3].set_title("GT")
                axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{stem}_panel.png"))
            plt.close(fig)

    print(f"Saved {split_name} outputs to {out_dir}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    run_split("train", TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
    run_split("test", TEST_IMAGE_DIR, TEST_MASK_DIR)