import time
import torch
import torch.nn as nn
import torch.optim as optim
from BostonFiles import BostonFiles
from BostonDataset import BostonDataset
from BostonDataModule import BostonDataModule
from SegmentationVisualizer import SegmentationVisualizer
from Training import train_one_epoch
from Validation import one_forward, validate_one_epoch
from Unet import UNet
import os

def main():
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Project basic information")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("\n")

    # Download and unzip images
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Download & Unzip images")
    files = BostonFiles('1T572f0oqy5JmuTvVEwkSUeXLWOSHl4hL','1pHp480_Q-s72RoDf1nD7ERzsv9yZTDE1')
    files.download()
    files.unzip()
    current_path = os.getcwd()
    img_dir = os.path.join(current_path, "images", "Images")
    msk_dir = os.path.join(current_path, "masks", "Segmentation_Masks")
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("\n")


    # Create the Boston Dataset
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Create the Boston dataset")
    dataset = BostonDataset(img_dir=img_dir, msk_dir=msk_dir)
    print("Dataset size:", len(dataset))
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("\n")

    # Split train - validation - test dataloaders and perform transformations
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Split train - validation - test & create dataloaders")
    data_module = BostonDataModule(dataset)
    trn_loader, val_loader, tst_loader = data_module.get_loaders()
    print("Batch Size:", data_module.batch_size)
    print("Train batches:", len(trn_loader))
    print("Val batches  :", len(val_loader))
    print("Test batches :", len(tst_loader))
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("\n")

    # Visualize an image/mask per loader
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Image examples & img/msk shapes")
    visualizer = SegmentationVisualizer(class_names=BostonDataset.CLASSES, class_color_map=BostonDataset.CLASS_COLOR_MAP)
    images, masks = next(iter(trn_loader))
    visualizer.show_triplet(images[0], masks[0], suptitle="Random test image & mask")
    print("Test image shape:", images.shape)  # [B, 3, H, W]
    print("Test mask shape :", masks.shape)  # [B, H, W]
    images, masks = next(iter(val_loader))
    visualizer.show_triplet(images[0], masks[0], suptitle="Random validation image & mask")
    print("Validation image shape:", images.shape)  # [B, 3, H, W]
    print("Validation mask shape :", masks.shape)  # [B, H, W]
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("\n")

    # Define the U-Net model
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Define the model & run one forward pass")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = UNet(in_channels=3, num_classes=len(BostonDataset.CLASSES)).to(device)
    images, masks = next(iter(trn_loader))
    images = images.to(device)
    masks = masks.to(device)
    logits = one_forward(model, images)
    visualizer.show_triplet(images[0], masks[0], logits[0], suptitle="Random test image & mask")
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("\n")

    # Define the training loop
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Define the training loop")
    num_classes = len(BostonDataset.CLASSES)
    loss_fn = nn.CrossEntropyLoss()  # expects logits + targets as long
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(1, 5, device, trn_loader, val_loader, model, optimizer, loss_fn)
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("\n")


def train(num_epochs, patience, device, trn_loader, val_loader, model, optimizer, loss_fn):
    num_epochs = 1
    best_val_loss = float("inf")
    patience = 5
    wait = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, trn_loader, optimizer, loss_fn, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet.pth")
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping triggered!")
            break

    print("Training finished!")
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("\n")


if __name__ == '__main__':
    main()


