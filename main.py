import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2  # New import for augmentation

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


def get_training_augmentation():
    """
    Augmentation pipeline optimized for 640x512 images (W x H).
    It focuses on applying "zoom-in" effects to help detect small objects
    without distorting the rectangular aspect ratio.
    """
    return v2.Compose([
        # 1. Horizontal flip: completely safe for maritime landscapes.
        v2.RandomHorizontalFlip(p=0.5),

        # 2. Rectangular zoom (key for small object detection):
        # - size=(512, 640): (Height, Width) fixes the exact output size.
        # - scale=(0.75, 1.0): crops between 75% and 100% of the original area
        #   (this simulates getting closer to the objects).
        # - ratio=(1.1, 1.4): enforces a rectangular crop (close to your original ~1.25).
        #   Prevents ships from being stretched and appearing square.
        v2.RandomResizedCrop(
            size=(512, 640),
            scale=(0.75, 1.0),
            ratio=(1.1, 1.4),
            antialias=True
        ),

        # 3. Color variation:
        # Helps the model avoid relying only on the exact blue tones of sea and sky.
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.01
        ),
    ])

def load_best_model(model_path: str, device: str | torch.device) -> torch.nn.Module:
    """
    Load the best saved U-Net model from disk.
    """
    device = torch.device(device)
    num_classes = len(BostonDataset.CLASSES)
    model = UNet(in_channels=3, num_classes=num_classes)
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
    Pick a random sample from the given dataloader and visualize.
    """
    device = torch.device(device)
    images, masks = next(iter(dataloader))
    batch_size = images.size(0)
    idx = random.randint(0, batch_size - 1)

    images = images.to(device)
    masks = masks.to(device)  # Ensure mask is on device

    with torch.no_grad():
        logits = model(images)  # [B, C, H, W]

    img_sample = images[idx].cpu()
    mask_sample = masks[idx].cpu()
    pred_sample = logits[idx].cpu()

    visualizer.show_triplet(
        image_tensor=img_sample,
        true_mask_tensor=mask_sample,
        pred_mask_tensor=pred_sample,
        suptitle=title,
    )


def run_experiment(
        experiment_name: str,
        use_augmentation: bool,
        dataset: BostonDataset,
        device: torch.device,
        current_path: str
):
    """
    Encapsulates the full training and validation lifecycle for a specific experiment configuration.
    """
    print(f"\n{'=' * 60}")
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print(f"Augmentation Enabled: {use_augmentation}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------
    # 4) Split dataset and create DataLoaders (Modified for Experiment)
    # ------------------------------------------------------------
    _print_section(f"[{experiment_name}] Splitting dataset & creating DataLoaders")

    # LOGIC FOR AUGMENTATION (Global parameter check)
    if use_augmentation:
        train_transform = get_training_augmentation()
        print(f"[{experiment_name}] Applying data augmentation to training set.")
    else:
        train_transform = None
        print(f"[{experiment_name}] No augmentation applied (Baseline).")

    # Create DataModule with specific transform
    data_module = BostonDataModule(dataset, batch_size=4, train_transform=train_transform)
    trn_loader, val_loader, tst_loader = data_module.get_loaders()

    print("Batch size      :", data_module.batch_size)
    print("Train batches   :", len(trn_loader))
    print("Validation batch:", len(val_loader))
    print()

    # ------------------------------------------------------------
    # 5) Visualize sample images (Optional: to verify augmentation)
    # ------------------------------------------------------------
    _print_section(f"[{experiment_name}] Visualizing sample images & masks")

    visualizer = SegmentationVisualizer(
        class_names=BostonDataset.CLASSES,
        class_color_map=BostonDataset.CLASS_COLOR_MAP,
    )

    # Sample from training loader (to see augmentation effects if enabled)
    images, masks = next(iter(trn_loader))
    visualizer.show_triplet(
        image_tensor=images[0],
        true_mask_tensor=masks[0],
        pred_mask_tensor=None,
        suptitle=f"[{experiment_name}] Sample training image & mask",
    )

    # ------------------------------------------------------------
    # 6) Define U-Net model
    # ------------------------------------------------------------
    _print_section(f"[{experiment_name}] Defining U-Net model")

    num_classes = len(BostonDataset.CLASSES)
    # Re-initialize model for every experiment to start with fresh weights
    model = UNet(in_channels=3, num_classes=num_classes).to(device)

    # ------------------------------------------------------------
    # 7) Define loss, optimizer and launch training loop
    # ------------------------------------------------------------
    _print_section(f"[{experiment_name}] Starting training")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 40
    patience = 10

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
        class_names=BostonDataset.CLASSES,
    )

    # ------------------------------------------------------------
    # 8) Save training history & Model
    # ------------------------------------------------------------
    # Save Metrics
    excel_filename = f"training_metrics_{experiment_name}.xlsx"
    excel_path = os.path.join(current_path, excel_filename)
    history_df.to_excel(excel_path, index=False)
    print(f"[{experiment_name}] Metrics saved to: {excel_path}")

    # Rename the best model to include experiment name (train function saves as 'best_unet.pth')
    default_model_name = "best_unet.pth"
    specific_model_name = f"best_unet_{experiment_name}.pth"

    if os.path.exists(default_model_name):
        os.rename(default_model_name, specific_model_name)
        print(f"[{experiment_name}] Best model renamed to: {specific_model_name}")

    # ------------------------------------------------------------
    # 9) Reload best model and visualize a random test example
    # ------------------------------------------------------------
    _print_section(f"[{experiment_name}] Visualizing random validation sample with best model")

    model_path = os.path.join(current_path, specific_model_name)
    best_model = load_best_model(model_path=model_path, device=device)

    visualize_random_example(
        model=best_model,
        dataloader=val_loader,
        device=device,
        visualizer=visualizer,
        title=f"[{experiment_name}] Random val sample: GT vs prediction",
    )

    return best_val_loss


def main():
    # ------------------------------------------------------------
    # 1) Basic project / environment information
    # ------------------------------------------------------------
    _print_section("Project information & environment")

    print("PyTorch version:", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # ------------------------------------------------------------
    # 2) Download and unzip dataset (images + masks)
    # ------------------------------------------------------------
    _print_section("Downloading & unzipping dataset")

    files = BostonFiles(
        '1T572f0oqy5JmuTvVEwkSUeXLWOSHl4hL',  # images ZIP id
        '1pHp480_Q-s72RoDf1nD7ERzsv9yZTDE1'  # masks ZIP id
    )

    # files.download() # Uncomment if files are not present
    # files.unzip()    # Uncomment if files are not unzipped

    current_path = os.getcwd()
    img_dir = os.path.join(current_path, "images", "Images")
    msk_dir = os.path.join(current_path, "masks", "Segmentation_Masks")

    # ------------------------------------------------------------
    # 3) Create the Boston dataset (BASE)
    # ------------------------------------------------------------
    _print_section("Creating Base BostonDataset instance")
    # This dataset contains all images. Splitting happens inside the experiment.
    dataset = BostonDataset(img_dir=img_dir, msk_dir=msk_dir)
    print("Total dataset size:", len(dataset))
    print()

    # ------------------------------------------------------------
    # EXECUTION FLOWS
    # ------------------------------------------------------------

    # FLOW 1: Baseline (No Augmentation)
    ''' 
    loss_no_aug = run_experiment(
        experiment_name="no_aug",
        use_augmentation=False,
        dataset=dataset,
        device=device,
        current_path=current_path
    )
    '''

    # FLOW 2: With Data Augmentation
    loss_with_aug = run_experiment(
        experiment_name="with_aug",
        use_augmentation=True,
        dataset=dataset,
        device=device,
        current_path=current_path
    )

    # ------------------------------------------------------------
    # Final Comparison
    # ------------------------------------------------------------
    '''
    _print_section("Final Results Comparison")
    print(f"Best Val Loss (No Aug)  : {loss_no_aug:.4f}")
    print(f"Best Val Loss (With Aug): {loss_with_aug:.4f}")

    if loss_with_aug < loss_no_aug:
        print("\nConclusion: Data Augmentation improved performance.")
    else:
        print("\nConclusion: Data Augmentation did not improve performance (or needs tuning).")
    '''

if __name__ == "__main__":
    main()