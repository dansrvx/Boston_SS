import time
import torch
import torch.nn as nn

def one_forward(model, images):
    with torch.no_grad():  # no gradient calculation
        logits = model(images)  # forward pass
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
    Validate the model for one epoch and print useful debug information.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        loss_fn: Loss function.
        device: Device ('cpu' or 'cuda').
        epoch: Current epoch index (for logging).
        print_every: How often to print batch progress.
    """

    model.eval()  # evaluation mode: disable dropout/batchnorm updates
    running_loss = 0.0
    num_batches = len(dataloader)

    epoch_start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader, start=1):
            batch_start_time = time.time()

            # --- Move data to device ---
            images = images.to(device)
            masks  = masks.to(device)

            # --- Debug only for the first batch ---
            if batch_idx == 1:
                print(f"[VAL Epoch {epoch}] First batch debug:")
                print(f"  images.shape = {images.shape}, dtype = {images.dtype}, device = {images.device}")
                print(f"  masks.shape  = {masks.shape}, dtype  = {masks.dtype}, device  = {masks.device}")

                # Optional: inspect ranges
                print(f"  images.min={images.min().item():.4f}, images.max={images.max().item():.4f}")
                print(f"  masks.min ={masks.min().item():.4f}, masks.max ={masks.max().item():.4f}")

            # --- Forward pass ---
            logits = model(images)

            # --- Compute loss ---
            loss = loss_fn(logits, masks)

            if not torch.isfinite(loss):
                print(f"[VAL WARNING] Non-finite loss at batch {batch_idx}: {loss.item()}")
                break

            batch_loss = loss.item()
            running_loss += batch_loss

            # --- Print every N batches ---
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

    # --- End of epoch ---
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / num_batches

    print(
        f"[VAL Epoch {epoch} DONE] "
        f"avg_loss = {epoch_loss:.4f} | "
        f"epoch_time = {epoch_time:.2f}s"
    )
    return epoch_loss