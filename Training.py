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
    Train the model for one epoch and print useful debug information.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for the training set.
        optimizer: Optimizer.
        loss_fn: Loss function.
        device: Device ('cpu' or 'cuda').
        epoch: Current epoch index (for logging).
        print_every: Print status every N batches.
    """
    model.train()  # put model in training mode
    running_loss = 0.0
    num_batches = len(dataloader)

    epoch_start_time = time.time()

    for batch_idx, (images, masks) in enumerate(dataloader, start=1):
        batch_start_time = time.time()

        # --- Move data to device ---
        images = images.to(device)
        masks  = masks.to(device)

        # --- Optional: debug first batch shapes/dtypes/devices ---
        if batch_idx == 1:
            print(f"[Epoch {epoch}] First batch debug:")
            print(f"  images.shape = {images.shape}, dtype = {images.dtype}, device = {images.device}")
            print(f"  masks.shape  = {masks.shape}, dtype  = {masks.dtype}, device  = {masks.device}")

        # --- Forward pass ---
        logits = model(images)

        # --- Compute loss ---
        loss = loss_fn(logits, masks)

        # --- Check for NaNs or infs in loss ---
        if not torch.isfinite(loss):
            print(f"[WARNING] Non-finite loss detected at batch {batch_idx}: {loss.item()}")
            print("Stopping epoch early to inspect the issue.")
            break

        # --- Backpropagation ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss

        # --- Print status every 'print_every' batches ---
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

    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / num_batches

    print(
        f"[Epoch {epoch} DONE] "
        f"avg_loss = {epoch_loss:.4f} | "
        f"epoch_time = {epoch_time:.2f}s"
    )
    return epoch_loss