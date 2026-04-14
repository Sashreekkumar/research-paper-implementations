import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Dataset Class
# -------------------------
class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        image = image / 255.0
        mask = mask / 255.0

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        mask = (mask > 0.5).float()

        return image, mask


# -------------------------
# DataLoader helper
# -------------------------
def get_loader(image_dir, mask_dir, batch_size=4, shuffle=True, img_size=256):
    dataset = BrainTumorDataset(image_dir, mask_dir, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader