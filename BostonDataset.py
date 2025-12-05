import os
import glob
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


class BostonDataset(Dataset):
    """
    Custom dataset for the Boston semantic segmentation project.

    This dataset:
    - Loads image/mask pairs from two folders.
    - Returns images as tensors [3, H, W] in float32, normalized to [0, 1].
    - Returns masks as tensors [H, W] in long, where each value is a class index.
    """

    # Semantic classes in the dataset (order defines the class index)
    CLASSES = [
        "sky",             # 0
        "water",           # 1
        "bridge",          # 2
        "obstacle",        # 3
        "living_obstacle", # 4
        "background",      # 5
        "self",            # 6
    ]

    # Color assigned to each class (R, G, B)
    CLASS_COLOR_MAP = {
        "sky":             (135, 206, 235),  # light sky blue
        "water":           (0, 0, 139),      # dark blue
        "bridge":          (255, 0, 255),    # magenta
        "obstacle":        (255, 0, 0),      # red
        "living_obstacle": (0, 255, 0),      # green
        "background":      (128, 128, 128),  # gray
        "self":            (255, 255, 0),    # yellow
    }

    def __init__(self, img_dir, msk_dir, transform=None, target_transform=None):
        """
        Initialize the dataset.

        Args:
            img_dir (str): Folder path where the images are stored.
            msk_dir (str): Folder path where the masks are stored.
            transform (callable, optional): Transform applied to the image (PIL -> tensor).
            target_transform (callable, optional): Transform applied to the mask (PIL -> tensor).
        """
        self.img_dir = img_dir
        self.msk_dir = msk_dir

        self.transform = transform
        self.target_transform = target_transform

        # Build the list of (image_path, mask_path) pairs
        self.samples = self.load_samples()

    def load_samples(self, img_extension="*.png"):
        """
        Create a list with all (image_path, mask_path) pairs.

        This assumes that:
        - Images are in `img_dir`.
        - Masks are in `msk_dir`.
        - Image and mask share the same file name (e.g., '123.png').
        """
        img_files = glob.glob(os.path.join(self.img_dir, img_extension))
        img_files = sorted(img_files)

        samples = []
        for img_path in img_files:
            filename = os.path.basename(img_path)
            msk_path = os.path.join(self.msk_dir, filename)

            if os.path.isfile(msk_path):
                samples.append((img_path, msk_path))
            else:
                print(f"[Warning] Mask not found for: {filename}")

        return samples

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return one (image, mask) pair as tensors.

        Image:
            - Loaded as RGB.
            - Converted to tensor [3, H, W], float32, values in [0, 1].

        Mask:
            - Loaded as grayscale ('L').
            - Converted to tensor [H, W], long.
            - Each pixel value is a class index (0 to num_classes-1).
        """
        # 1. Get paths for image and mask
        img_path, msk_path = self.samples[idx]

        # 2. Load image and mask as PIL Images
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")  # 'L' = 8-bit pixels, 0..255

        # 3. Apply optional transforms if provided by the user
        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: convert PIL image to tensor [3, H, W] in [0, 1]
            img = np.array(img, dtype=np.uint8)           # (H, W, 3)
            img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
            img = img.float() / 255.0                     # scale to [0, 1]

        if self.target_transform is not None:
            msk = self.target_transform(msk)
        else:
            # Default: convert PIL mask to tensor [H, W] with class indices
            # Here we assume the PNG already stores the correct class index per pixel.
            msk = np.array(msk, dtype=np.int64)    # (H, W)
            msk = torch.from_numpy(msk)            # tensor long [H, W]

        return img, msk
