"""
CNN Training Pipeline
=====================
Supports: ResNet (18/34/50/101/152), VGG (11/13/16/19), AlexNet, MobileNetV2
Datasets : CIFAR-10, CIFAR-100, MNIST, FashionMNIST, ImageNet-style (custom folder)
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ─────────────────────────────────────────────
# 1. DATASET REGISTRY
# ─────────────────────────────────────────────

def get_transforms(dataset_name: str, img_size: int = 224):
    """Returns (train_transform, val_transform) for a given dataset."""
    normalize_params = {
        "cifar10":       ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        "cifar100":      ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        "mnist":         ([0.1307],                  [0.3081]),
        "fashionmnist":  ([0.2860],                  [0.3530]),
        "imagenet":      ([0.485,  0.456,  0.406],   [0.229,  0.224,  0.225]),
    }
    mean, std = normalize_params.get(dataset_name.lower(), normalize_params["imagenet"])

    if dataset_name.lower() in ("mnist", "fashionmnist"):
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),   # CNNs expect 3ch
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean * 3, std * 3),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean * 3, std * 3),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return train_tf, val_tf


def get_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    val_split: float = 0.1,
    img_size: int = 224,
):
    """
    Returns (train_loader, val_loader, test_loader, num_classes, class_names).

    Supported dataset_name values:
        "cifar10", "cifar100", "mnist", "fashionmnist",
        "imagenet"  →  expects data_dir/<train|val> folders (ImageFolder layout)
    """
    train_tf, val_tf = get_transforms(dataset_name, img_size)
    name = dataset_name.lower()

    ds_map = {
        "cifar10":      (datasets.CIFAR10,       10),
        "cifar100":     (datasets.CIFAR100,      100),
        "mnist":        (datasets.MNIST,          10),
        "fashionmnist": (datasets.FashionMNIST,  10),
    }

    if name in ds_map:
        cls, num_classes = ds_map[name]
        train_full = cls(data_dir, train=True,  download=True, transform=train_tf)
        test_ds    = cls(data_dir, train=False, download=True, transform=val_tf)

        val_size   = int(len(train_full) * val_split)
        train_size = len(train_full) - val_size
        train_ds, val_ds = random_split(
            train_full, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        # Apply val transforms to the val split
        val_ds.dataset = copy.deepcopy(train_full)
        val_ds.dataset.transform = val_tf

        class_names = train_full.classes if hasattr(train_full, "classes") else [str(i) for i in range(num_classes)]

    elif name == "imagenet":
        train_ds   = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
        val_ds     = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)
        test_ds    = val_ds
        num_classes = len(train_ds.classes)
        class_names = train_ds.classes

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: cifar10, cifar100, mnist, fashionmnist, imagenet")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[Dataset] {dataset_name.upper()} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | Classes: {num_classes}")
    return train_loader, val_loader, test_loader, num_classes, class_names


# ─────────────────────────────────────────────
# 2. MODEL REGISTRY
# ─────────────────────────────────────────────

def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """
    Returns a torchvision model with the classifier head swapped for num_classes.

    Supported model_name values:
        resnet18, resnet34, resnet50, resnet101, resnet152
        vgg11, vgg13, vgg16, vgg19
        alexnet
        mobilenet_v2
    """
    weights_arg = "DEFAULT" if pretrained else None
    name = model_name.lower()

    model_fn = {
        "resnet18":    models.resnet18,
        "resnet34":    models.resnet34,
        "resnet50":    models.resnet50,
        "resnet101":   models.resnet101,
        "resnet152":   models.resnet152,
        "vgg11":       models.vgg11,
        "vgg13":       models.vgg13,
        "vgg16":       models.vgg16,
        "vgg19":       models.vgg19,
        "alexnet":     models.alexnet,
        "mobilenet_v2": models.mobilenet_v2,
    }

    if name not in model_fn:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_fn.keys())}")

    model = model_fn[name](weights=weights_arg)

    # ── Freeze backbone (useful for transfer learning warm-up) ──
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ── Swap the final classification head ──
    if "resnet" in name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name in ("vgg11", "vgg13", "vgg16", "vgg19"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name == "alexnet":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name == "mobilenet_v2":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} | Total params: {total:,} | Trainable: {trainable:,} | Pretrained: {pretrained}")
    return model


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds     = outputs.max(1)
        correct      += preds.eq(labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds     = outputs.max(1)
        correct      += preds.eq(labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def train(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    save_path: str = "best_model.pth",
    scheduler_type: str = "cosine",   # "cosine" | "step" | "none"
):
    """
    Full training loop with:
      - Cosine / StepLR / no scheduler
      - Best-model checkpointing
      - Live loss & accuracy logging
    Returns history dict with train/val loss and accuracy per epoch.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device} | Epochs: {num_epochs} | LR: {lr}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        scheduler = None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:3d}/{num_epochs}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}%  |  "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.2f}%  |  "
              f"Time: {elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            print(f"  ✓ Best model saved (val acc: {best_val_acc:.2f}%)")

    model.load_state_dict(best_weights)
    print(f"\n[Train] Done. Best Val Accuracy: {best_val_acc:.2f}%")
    return model, history


# ─────────────────────────────────────────────
# 4. EVALUATION & METRICS
# ─────────────────────────────────────────────

@torch.no_grad()
def get_all_preds_labels(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def eval_metrics(model, test_loader, class_names, device="auto"):
    """
    Prints:
      - Top-1 test accuracy
      - Per-class precision, recall, F1
      - Confusion matrix heatmap (saved as confusion_matrix.png)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    preds, labels = get_all_preds_labels(model, test_loader, device)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # ── Confusion matrix ──
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(
        cm, annot=len(class_names) <= 20,
        fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("[Eval] Confusion matrix saved → confusion_matrix.png")
    return preds, labels


def plot_history(history: dict, save_path: str = "training_curves.png"):
    """Plots train/val loss and accuracy curves side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    ax1.set_title("Loss");  ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Val",   linewidth=2, linestyle="--")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[Plot] Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# 5. FULL PIPELINE  (edit this section to run)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── CONFIG ──────────────────────────────
    DATASET      = "cifar10"       # cifar10 | cifar100 | mnist | fashionmnist | imagenet
    MODEL_NAME   = "resnet50"      # resnet18/34/50/101/152 | vgg11/13/16/19 | alexnet | mobilenet_v2
    PRETRAINED   = True            # Use ImageNet pretrained weights
    FREEZE       = False           # Freeze backbone (linear-probe style transfer)
    NUM_EPOCHS   = 20
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    SCHEDULER    = "cosine"        # cosine | step | none
    IMG_SIZE     = 224             # 224 for most torchvision models
    DATA_DIR     = "./data"
    SAVE_PATH    = f"best_{MODEL_NAME}_{DATASET}.pth"
    # ────────────────────────────────────────

    # 1. Load dataset
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataset(
        DATASET, DATA_DIR, val_split=0.1, img_size=IMG_SIZE
    )

    # 2. Build model
    model = get_model(MODEL_NAME, num_classes, pretrained=PRETRAINED, freeze_backbone=FREEZE)

    # 3. Train
    model, history = train(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        scheduler_type=SCHEDULER,
        save_path=SAVE_PATH,
    )

    # 4. Plot training curves
    plot_history(history)

    # 5. Evaluate on test set
    eval_metrics(model, test_loader, class_names)