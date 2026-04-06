import torch
import os
import glob
import time
from PIL import Image
from torchvision.transforms import v2

# --- Configuration ---
REPO_DIR = "/home/charlieof604/project2/dinov3"
INPUT_DIR = "./img"
OUTPUT_DIR = "./output_masks"
MIRROR_CLASS_IDX = 27
WORK_SIZE = 384   # try 384 first; if quality drops too much, go back to 512

import sys
sys.path.append(REPO_DIR)
from dinov3.eval.segmentation.inference import make_inference

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    print("[1/4] Loading model...")
    t0 = time.time()
    model = torch.hub.load(
        REPO_DIR,
        'dinov3_vit7b16_ms',
        source="local",
        weights="./dinov3_weights/dinov3_vit7b16_segmentor.pth",
        backbone_weights="./dinov3_weights/dinov3_vit7b16_pretrain_backbone.pth",
        torch_dtype=torch.bfloat16
    )
    model = model.cuda().eval()
    print(f"[1/4] Model loaded in {time.time() - t0:.2f}s")
    return model

def get_transform(size=512):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

model = load_model()
transform = get_transform(WORK_SIZE)
crop_res = (WORK_SIZE, WORK_SIZE)

# find images
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))[:400]
print(f"[2/4] Found {len(image_paths)} jpg images in {INPUT_DIR}")

if len(image_paths) == 0:
    raise RuntimeError("No .jpg images found. Check INPUT_DIR and file extensions.")

for idx, path in enumerate(image_paths, 1):
    start_time = time.time()
    filename = os.path.basename(path)
    out_path = os.path.join(OUTPUT_DIR, f"mask_{filename}.png")

    print(f"[3/4] ({idx}/{len(image_paths)}) Processing {filename}")

    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    input_tensor = transform(img).unsqueeze(0).to("cuda", non_blocking=True)

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            seg_logits = make_inference(
                input_tensor,
                model,
                inference_mode="whole",
                decoder_head_type="m2f",
                rescale_to=crop_res,
                n_output_channels=150,
            )

            pred_gpu = seg_logits.argmax(dim=1)  # [1, H, W]

            upscaled = torch.nn.functional.interpolate(
                pred_gpu.unsqueeze(1).float(),
                size=(orig_h, orig_w),
                mode="nearest"
            ).squeeze(0).squeeze(0)

            predictions = upscaled.byte().cpu().numpy()

    mask = (predictions == MIRROR_CLASS_IDX)
    mask_img = Image.fromarray((mask * 255).astype("uint8"))

    print(f"[4/4] Saving -> {out_path}")
    mask_img.save(out_path)

    # cleanup references only
    del seg_logits, pred_gpu, upscaled, input_tensor, predictions, mask_img, mask

    elapsed = time.time() - start_time
    print(f"✅ Done {filename} in {elapsed:.2f}s")

print("All done.")