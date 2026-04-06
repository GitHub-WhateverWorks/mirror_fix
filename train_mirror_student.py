import os
import glob
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
import segmentation_models_pytorch as smp

# -----------------------------
# Config
# -----------------------------
TRAIN_IMAGE_DIR = "./img/MSD/MSD/train/image/"
TRAIN_MASK_DIR  = "./img/MSD/MSD/train/mask/"
TEST_IMAGE_DIR  = "./img/MSD/MSD/test/image/"
TEST_MASK_DIR   = "./img/MSD/MSD/test/mask/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 384
VAL_RATIO = 0.2
SEED = 42

# -----------------------------
# Dataset
# -----------------------------
class MirrorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=384, augment=False):
        self.samples = []
        self.img_size = img_size
        self.augment = augment

        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        mask_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in mask_paths
        }

        for img_path in image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if stem in mask_map:
                self.samples.append((img_path, mask_map[stem]))

        self.img_tf = v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
        ])

        self.mask_tf = v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=False),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = (mask > 0.5).float()

        return img, mask

# -----------------------------
# Loss / metric
# -----------------------------
def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1,2,3))
    denom = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return 1 - ((2 * inter) / denom).mean()

def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    return bce + dice

@torch.no_grad()
def dice_score(logits, target, threshold=0.3, eps=1e-6):
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return ((2 * inter) / denom).mean().item()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        total_loss += loss.item()
        total_dice += dice_score(logits, masks)
        n += 1
    return total_loss / max(n, 1), total_dice / max(n, 1)

# -----------------------------
# Train
# -----------------------------
def train():
    full_dataset = MirrorDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, img_size=IMG_SIZE, augment=False)
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataset = MirrorDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, img_size=IMG_SIZE, augment=True)
    val_dataset   = MirrorDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, img_size=IMG_SIZE, augment=False)

    train_dataset.samples = [full_dataset.samples[i] for i in train_subset.indices]
    val_dataset.samples   = [full_dataset.samples[i] for i in val_subset.indices]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Train pairs:", len(train_dataset))
    print("Val pairs:", len(val_dataset))

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_train = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            total_train += loss.item()

        train_loss = total_train / max(len(train_loader), 1)
        val_loss, val_dice = evaluate(model, val_loader)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        torch.save(model.state_dict(), "mirror_unet_last.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "mirror_unet_best.pth")
            print("Saved mirror_unet_best.pth")

if __name__ == "__main__":
    train()