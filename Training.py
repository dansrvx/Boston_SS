import time
import torch
import torch.nn as nn


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
    epoch: int = 1,
    print_every: int = 10,
):
    """
    Train the model for one full epoch and print useful debug information.

    This function:
        - Performs forward + backward + optimizer steps.
        - Computes and logs batch-level and epoch-level loss.
        - Prints diagnostic information for the first batch.
        - Monitors numerical stability (NaN / Inf losses).

    Args:
        model: PyTorch segmentation model (e.g., U-Net).
        dataloader: Training DataLoader yielding (images, masks).
        optimizer: Optimizer instance (Adam, SGD, etc.).
        loss_fn: Loss function used for training.
        device: Device string ('cpu' or 'cuda').
        epoch: Current epoch index (for logging).
        print_every: Print status every N batches.

    Returns:
        epoch_loss (float): Mean loss for the entire epoch.
    """

    # ------------------------------------------------------------
    # Setup model in training mode
    # ------------------------------------------------------------
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    epoch_start_time = time.time()

    # ------------------------------------------------------------
    # Iterate over all batches
    # ------------------------------------------------------------
    for batch_idx, (images, masks) in enumerate(dataloader, start=1):
        batch_start_time = time.time()

        # ------------------------------------------------------------
        # Move input data to the correct device
        # ------------------------------------------------------------
        images = images.to(device)
        masks = masks.to(device)

        # ------------------------------------------------------------
        # Print detailed debug information for the first batch only
        # ------------------------------------------------------------
        if batch_idx == 1:
            print(f"[Epoch {epoch}] First batch debug:")
            print(f"  images.shape = {images.shape}, dtype = {images.dtype}, device = {images.device}")
            print(f"  masks.shape  = {masks.shape}, dtype  = {masks.dtype}, device  = {masks.device}")

        # ------------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------------
        logits = model(images)

        # ------------------------------------------------------------
        # Compute loss
        # ------------------------------------------------------------
        loss = loss_fn(logits, masks)

        # Numerical stability check
        if not torch.isfinite(loss):
            print(f"[WARNING] Non-finite loss detected at batch {batch_idx}: {loss.item()}")
            print("Stopping epoch early for safety.")
            break

        # ------------------------------------------------------------
        # Backpropagation and optimization step
        # ------------------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------------------------------------------------
        # Accumulate loss statistics
        # ------------------------------------------------------------
        batch_loss = loss.item()
        running_loss += batch_loss

        # ------------------------------------------------------------
        # Logging: print progress every 'print_every' batches
        # ------------------------------------------------------------
        if (batch_idx % print_every == 0) or (batch_idx == num_batches):
            avg_loss_so_far = running_loss / batch_idx
            batch_time = time.time() - batch_start_time

            print(
                f"[Epoch {epoch}] "
                f"Batch {batch_idx}/{num_batches} | "
                f"batch_loss = {batch_loss:.4f} | "
                f"avg_loss = {avg_loss_so_far:.4f} | "
                f"batch_time = {batch_time:.3f}s"
            )

    # ------------------------------------------------------------
    # End-of-epoch statistics
    # ------------------------------------------------------------
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / num_batches

    print(
        f"[Epoch {epoch} DONE] "
        f"avg_loss = {epoch_loss:.4f} | "
        f"epoch_time = {epoch_time:.2f}s"
    )

    return epoch_loss
