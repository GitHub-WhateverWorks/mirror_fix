import torch
import os
import glob
import time
from PIL import Image
from torchvision.transforms import v2
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
REPO_DIR = "/home/charlieof604/project2/dinov3" # Use your Linux path!
INPUT_DIR = "./img"
OUTPUT_DIR = "./output_masks"
MIRROR_CLASS_IDX = 27

import sys
sys.path.append(REPO_DIR)
from dinov3.eval.segmentation.inference import make_inference

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    print("Starting massive model load (this will take a while)...")
    # Using your previous successful loading logic
    model = torch.hub.load(
        REPO_DIR, 
        'dinov3_vit7b16_ms', 
        source="local", 
        weights="./dinov3_weights/dinov3_vit7b16_segmentor.pth",
        backbone_weights="./dinov3_weights/dinov3_vit7b16_pretrain_backbone.pth",
        torch_dtype=torch.bfloat16
    )
    return model.cuda().eval()

def get_transform(size=896):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

# 1. Load Model ONCE
crop_res = (512, 512)

model = load_model()
print("Model Loaded")
transform = get_transform(512)

# 2. Get all .jpg files
#image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))[:300]
all_image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
image_paths = []

for path in all_image_paths:
    filename = os.path.basename(path)
    out_path = os.path.join(OUTPUT_DIR, f"mask_{filename}.png")
    if not os.path.exists(out_path):
        image_paths.append(path)

# 👇 LIMIT to 200 new imagess
image_paths = image_paths[:200]

print(f"Processing {len(image_paths)} new images")
#print(f"Found {len(image_paths)} images. Starting inference...")

# 3. Process Loop
for path in image_paths:
    start_time = time.time()
    filename = os.path.basename(path)
    
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    input_tensor = transform(img).unsqueeze(0).cuda()

    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            seg_logits = make_inference(
                input_tensor, model,
                inference_mode="whole", #slide
                decoder_head_type="m2f",
                rescale_to=crop_res,
                n_output_channels=150,
                
                #crop_size=(896, 896),
                #stride=(896, 896),
                #output_activation=partial(torch.nn.functional.softmax, dim=1),
                
            )
            
            # --- DEBUGGING STEPS ---
            # 1. Get the raw class predictions (0-149)
            predictions = seg_logits.argmax(dim=1) # [1, H, W]
            predictions = torch.nn.functional.interpolate(
                predictions.unsqueeze(1).float(), 
                size=(orig_h, orig_w), 
                mode="nearest"
            ).squeeze().cpu().numpy()
            # 2. Check which classes were actually found
            detected_classes = np.unique(predictions)
            print(f"Image: {filename} | Detected Class Indices: {detected_classes}")
            
            # 3. Create the binary mirror mask
            mask = (predictions == MIRROR_CLASS_IDX)
            
            # 4. Save a 'Heatmap' of all classes to see what it DID find
            # (If this is all one color, the model is 'blind')
            #plt.imsave(os.path.join(OUTPUT_DIR, f"heatmap_{filename}.png"), predictions, cmap='nipy_spectral')
            
            # 5. Save the actual mirror mask
            mask_img = Image.fromarray((mask * 255).astype('uint8'))
            mask_img.save(os.path.join(OUTPUT_DIR, f"mask_{filename}.png"))
    
    elapsed = time.time() - start_time
    print(f"✅ Processed {filename} in {elapsed:.2f}s")
    del seg_logits
    del input_tensor

print("Done! Check the /output_masks folder.")