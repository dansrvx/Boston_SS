import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

from BostonFiles import BostonFiles
from BostonDataset import BostonDataset
from BostonDataModule import BostonDataModule
from SegmentationVisualizer import SegmentationVisualizer
from Execution import train
from Validation import one_forward
from Unet import UNet


def _print_section(title: str):
    """
    Utility function to print a formatted section header.
    """
    line = "*" * 40
    print(line)
    print(title)
    print(line)
    print()  # blank line for readability


def main():
    # ------------------------------------------------------------
    # 1) Basic project / environment information
    # ------------------------------------------------------------
    _print_section("Project information & environment")

    print("PyTorch version:", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device   :", torch.cuda.get_device_name(0))
    else:
        print("CUDA device   : CPU only (no GPU detected)")
    print()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # 2) Download and unzip dataset (images + masks)
    # ------------------------------------------------------------
    _print_section("Downloading & unzipping dataset")

    # These IDs should point to the images and masks ZIPs (e.g. on Google Drive)
    files = BostonFiles(
        '1T572f0oqy5JmuTvVEwkSUeXLWOSHl4hL',   # images ZIP id
        '1pHp480_Q-s72RoDf1nD7ERzsv9yZTDE1'    # masks ZIP id
    )

    print("Downloading files...")
    files.download()
    print("Unzipping files...")
    files.unzip()

    # Build paths for images and masks based on current working directory
    current_path = os.getcwd()
    img_dir = os.path.join(current_path, "images", "Images")
    msk_dir = os.path.join(current_path, "masks", "Segmentation_Masks")

    print("Images directory:", img_dir)
    print("Masks directory :", msk_dir)
    print()

    # ------------------------------------------------------------
    # 3) Create the Boston dataset
    # ------------------------------------------------------------
    _print_section("Creating BostonDataset instance")

    dataset = BostonDataset(img_dir=img_dir, msk_dir=msk_dir)
    print("Total dataset size:", len(dataset))
    print()

    # ------------------------------------------------------------
    # 4) Split dataset and create DataLoaders
    # ------------------------------------------------------------
    _print_section("Splitting dataset & creating DataLoaders")

    data_module = BostonDataModule(dataset, batch_size=4)
    trn_loader, val_loader, tst_loader = data_module.get_loaders()

    print("Batch size      :", data_module.batch_size)
    print("Train batches   :", len(trn_loader))
    print("Validation batch:", len(val_loader))
    print("Test batches    :", len(tst_loader))
    print()

    # ------------------------------------------------------------
    # 5) Visualize sample images and masks from train/val
    # ------------------------------------------------------------
    _print_section("Visualizing sample images & masks")

    visualizer = SegmentationVisualizer(
        class_names=BostonDataset.CLASSES,
        class_color_map=BostonDataset.CLASS_COLOR_MAP,
    )

    # Sample from training loader
    images, masks = next(iter(trn_loader))
    visualizer.show_triplet(
        image_tensor=images[0],
        true_mask_tensor=masks[0],
        pred_mask_tensor=None,
        suptitle="Sample training image & mask",
    )
    print("Training batch image shape:", images.shape)  # [B, 3, H, W]
    print("Training batch mask shape :", masks.shape)   # [B, H, W]")
    print()

    # Sample from validation loader
    images, masks = next(iter(val_loader))
    visualizer.show_triplet(
        image_tensor=images[0],
        true_mask_tensor=masks[0],
        pred_mask_tensor=None,
        suptitle="Sample validation image & mask",
    )
    print("Validation batch image shape:", images.shape)  # [B, 3, H, W]
    print("Validation batch mask shape :", masks.shape)   # [B, H, W]")
    print()


    # ------------------------------------------------------------
    # 6) Define U-Net model and run a single forward pass
    # ------------------------------------------------------------
    _print_section("Defining U-Net model & running one forward pass")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = len(BostonDataset.CLASSES)
    model = UNet(in_channels=3, num_classes=num_classes).to(device)

    # Take a small batch from train loader and run through the model
    images, masks = next(iter(trn_loader))
    images = images.to(device)
    masks = masks.to(device)

    logits = one_forward(model, images)

    # Visualize image, ground truth and predicted mask (logits)
    visualizer.show_triplet(
        image_tensor=images[0],
        true_mask_tensor=masks[0],
        pred_mask_tensor=logits[0],
        suptitle="Sample train image: ground truth vs prediction (untrained model)",
    )
    print()

    # ------------------------------------------------------------
    # 7) Define loss, optimizer and launch training loop
    # ------------------------------------------------------------
    _print_section("Defining training loop & starting training")

    loss_fn = nn.CrossEntropyLoss()  # expects logits + targets as long indices
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train configuration
    num_epochs = 40


    patience = 5

    print(f"Starting training for {num_epochs} epochs (patience = {patience})...")
    best_val_loss, history_df = train(
        num_epochs=num_epochs,
        patience=patience,
        device=device,
        trn_loader=trn_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_classes=num_classes,
        ignore_index=None,
        class_names=BostonDataset.CLASSES,
    )

    print("\nTraining completed.")
    print("Best validation loss:", best_val_loss)
    print("History DataFrame head:")
    print(history_df.head())
    print()

    # ------------------------------------------------------------
    # 8) Save training history DataFrame for later analysis
    # ------------------------------------------------------------
    excel_path = os.path.join(current_path, "training_metrics.xlsx")
    history_df.to_excel(excel_path, index=False)
    print(f"Training metrics also saved as Excel: {excel_path}")

    # ------------------------------------------------------------
    # 9) Reload best model and visualize a random test example
    # ------------------------------------------------------------
    _print_section("Reloading best model & visualizing random test sample")

    model_path = os.path.join(current_path, "best_unet.pth")

    # Load best model from disk
    best_model = load_best_model(model_path=model_path, device=device)

    # Create a new visualizer (if not already available)
    visualizer = SegmentationVisualizer(
        class_names=BostonDataset.CLASSES,
        class_color_map=BostonDataset.CLASS_COLOR_MAP,
    )

    # Use test loader if you want to see true generalization,
    # or val_loader if test_loader is small.
    visualize_random_example(
        model=best_model,
        dataloader=val_loader,       # or val_loader
        device=device,
        visualizer=visualizer,
        title="Random test sample: GT vs prediction vs diff",
    )


def load_best_model(model_path: str, device: str | torch.device) -> torch.nn.Module:
    """
    Load the best saved U-Net model from disk.

    Args:
        model_path: Path to the saved state_dict file (e.g. 'best_unet.pth').
        device: Target device ('cpu' or 'cuda').

    Returns:
        model: UNet model loaded with the saved weights, in eval mode.
    """
    device = torch.device(device)

    num_classes = len(BostonDataset.CLASSES)
    model = UNet(in_channels=3, num_classes=num_classes)

    # Load state dict to the correct device
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def visualize_random_example(
    model: torch.nn.Module,
    dataloader,
    device: str | torch.device,
    visualizer: SegmentationVisualizer,
    title: str | None = "Random sample: image, GT, prediction, diff",
):
    """
    Pick a random sample from the given dataloader, run inference with the model,
    and visualize:
        - Input image
        - Ground truth mask
        - Predicted mask
        - Match/mismatch map (green = correct, red = incorrect)

    Args:
        model: Trained segmentation model (U-Net) in eval mode.
        dataloader: DataLoader to sample from (e.g., test or validation loader).
        device: Target device ('cpu' or 'cuda').
        visualizer: Instance of SegmentationVisualizer.
        title: Optional global title for the figure.
    """
    device = torch.device(device)

    # Get one random batch from the dataloader
    images, masks = next(iter(dataloader))

    # Select a random index inside the batch
    batch_size = images.size(0)
    idx = random.randint(0, batch_size - 1)

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)

    # Run inference (no gradients needed)
    with torch.no_grad():
        logits = model(images)  # [B, C, H, W]

    # Select the random sample
    img_sample = images[idx].cpu()
    mask_sample = masks[idx].cpu()
    pred_sample = logits[idx].cpu()  # logits for this sample; visualizer will argmax

    # Use the visualizer to show image, GT mask, prediction and diff map
    visualizer.show_triplet(
        image_tensor=img_sample,
        true_mask_tensor=mask_sample,
        pred_mask_tensor=pred_sample,
        suptitle=title,
    )


if __name__ == "__main__":
    main()
