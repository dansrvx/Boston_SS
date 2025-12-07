import time
import torch
import torch.nn as nn
from Metrics import Metrics
from BostonDataset import BostonDataset


def one_forward(model, images):
    """
    Utility function to perform a forward pass without gradient computation.
    Useful for quick debugging of output shapes.

    Args:
        model: PyTorch model.
        images: Input batch of shape [B, C, H, W].

    Returns:
        logits: Model raw outputs of shape [B, num_classes, H, W].
    """
    with torch.no_grad():
        logits = model(images)
    print("Logits shape:", logits.shape)
    return logits


def validate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    epoch: int = 1,
    print_every: int = 10,
):
    """
    Run one full validation epoch, computing loss and segmentation metrics.

    This function:
        - Processes the entire validation set with the model in eval mode.
        - Computes per-batch and epoch-level loss.
        - Accumulates a confusion matrix for IoU / Precision / Recall.
        - Prints useful debug information for the first batch.
        - Returns the epoch loss and a metrics dictionary.

    Args:
        model: PyTorch model (U-Net or similar).
        dataloader: Validation DataLoader providing (images, masks).
        loss_fn: Loss function for segmentation (e.g., CrossEntropyLoss).
        device: Device string ('cpu' or 'cuda').
        epoch: Current epoch index (for logging).
        print_every: Print metrics every N batches.

    Returns:
        epoch_loss (float)
        metrics_results (dict): Contains per-class + mean IoU, Precision, Recall.
    """

    # ------------------------------------------------------------
    # Setup model and bookkeeping structures
    # ------------------------------------------------------------
    model.eval()  # disable dropout and batchnorm statistics updates
    running_loss = 0.0
    num_batches = len(dataloader)

    # Create metrics object using the number of classes from the dataset
    metrics = Metrics(num_classes=len(BostonDataset.CLASSES), device=device)
    metrics.reset()

    epoch_start_time = time.time()

    # ------------------------------------------------------------
    # Disable gradient calculation for validation
    # ------------------------------------------------------------
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader, start=1):
            batch_start_time = time.time()

            # ------------------------------------------------------------
            # Move data to target device
            # ------------------------------------------------------------
            images = images.to(device)
            masks = masks.to(device)

            # ------------------------------------------------------------
            # Debug info for the first batch (shapes, ranges, devices)
            # ------------------------------------------------------------
            if batch_idx == 1:
                print(f"[VAL Epoch {epoch}] First batch debug:")
                print(f"  images.shape = {images.shape}, dtype = {images.dtype}, device = {images.device}")
                print(f"  masks.shape  = {masks.shape}, dtype  = {masks.dtype}, device  = {masks.device}")

                print(f"  images.min={images.min().item():.4f}, images.max={images.max().item():.4f}")
                print(f"  masks.min ={masks.min().item():.4f}, masks.max ={masks.max().item():.4f}")

            # ------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------
            logits = model(images)

            # ------------------------------------------------------------
            # Loss computation
            # ------------------------------------------------------------
            loss = loss_fn(logits, masks)

            # Validation safety check (NAN/INF values)
            if not torch.isfinite(loss):
                print(f"[VAL WARNING] Non-finite loss at batch {batch_idx}: {loss.item()}")
                break

            batch_loss = loss.item()
            running_loss += batch_loss

            # ------------------------------------------------------------
            # Update segmentation metrics (IoU / Precision / Recall)
            # ------------------------------------------------------------
            metrics.update(logits, masks)

            # ------------------------------------------------------------
            # Logging for batch-level progress
            # ------------------------------------------------------------
            if (batch_idx % print_every == 0) or (batch_idx == num_batches):
                avg_loss_so_far = running_loss / batch_idx
                batch_time = time.time() - batch_start_time

                print(
                    f"[VAL Epoch {epoch}] "
                    f"Batch {batch_idx}/{num_batches} | "
                    f"batch_loss = {batch_loss:.4f} | "
                    f"avg_loss = {avg_loss_so_far:.4f} | "
                    f"batch_time = {batch_time:.3f}s"
                )

    # ------------------------------------------------------------
    # End of epoch: compute final loss & metrics
    # ------------------------------------------------------------
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / num_batches

    metrics_results = metrics.compute()

    print(
        f"[VAL Epoch {epoch} DONE] "
        f"avg_loss = {epoch_loss:.4f} | "
        f"epoch_time = {epoch_time:.2f}s"
    )

    print(
        f"[VAL Epoch {epoch} METRICS] "
        f"mIoU = {metrics_results['mean_iou']:.4f} | "
        f"mPrecision = {metrics_results['mean_precision']:.4f} | "
        f"mRecall = {metrics_results['mean_recall']:.4f}"
    )

    return epoch_loss, metrics_results
