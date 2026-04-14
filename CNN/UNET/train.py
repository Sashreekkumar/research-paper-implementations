import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import BrainTumorDataset
from UNET import UNet
from dice_loss import DiceLoss
from metrics import (
    dice_score,
    iou_score,
    rand_error,
    hausdorff_error
)

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(
    BrainTumorDataset("data/images", "data/masks"),
    batch_size=4,
    shuffle=True
)

model = UNet(in_channels=1, num_classes=1).to(device)

criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -------------------------
# Training
# -------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")


# -------------------------
# Evaluation (after training)
# -------------------------
model.eval()

images, masks = next(iter(train_loader))
images, masks = images.to(device), masks.to(device)

with torch.no_grad():
    preds = model(images)
    preds = torch.sigmoid(preds)
    preds_bin = (preds > 0.5).float()


# -------------------------
# Metrics
# -------------------------
dice = dice_score(preds_bin, masks).item()
iou = iou_score(preds_bin, masks).item()
rand = rand_error(preds_bin, masks).item()
haus = hausdorff_error(preds_bin[0][0], masks[0][0])

print("\n===== RESULTS =====")
print(f"Dice Score    : {dice:.4f}")
print(f"IoU Score     : {iou:.4f}")
print(f"Rand Error    : {rand:.4f}")
print(f"Hausdorff Err : {haus:.4f}")


# -------------------------
# Visualization (paper-style figure)
# -------------------------
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Input MRI")
plt.imshow(images[0][0].cpu(), cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(masks[0][0].cpu(), cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(preds_bin[0][0].cpu(), cmap="gray")

plt.savefig("results/sample_result.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved result to results/sample_result.png")