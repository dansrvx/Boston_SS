from Training import train_one_epoch
from Validation import validate_one_epoch
import pandas as pd
import torch

def train(
    num_epochs,
    patience,
    device,
    trn_loader,
    val_loader,
    model,
    optimizer,
    loss_fn,
    num_classes: int,
    ignore_index: int | None = None,
    class_names: list[str] | None = None,   # Optional: human-readable class labels
):
    """
    Full training loop with:
    - per-epoch training loss
    - per-epoch validation loss
    - semantic segmentation metrics (IoU, Precision, Recall)
    - per-class metrics
    - early stopping
    - automatic saving of best model
    - final Pandas DataFrame containing all metrics

    Returns:
        best_val_loss (float)
        history_df (pd.DataFrame)
    """

    best_val_loss = float("inf")
    wait = 0

    # List of dicts: one row per epoch → later converted to DataFrame
    history = []

    for epoch in range(num_epochs):
        current_epoch = epoch + 1

        # ----------------------------------------------------------
        # TRAINING PHASE
        # ----------------------------------------------------------
        train_loss = train_one_epoch(
            model,
            trn_loader,
            optimizer,
            loss_fn,
            device,
            epoch=current_epoch,
            print_every=1,
        )

        # ----------------------------------------------------------
        # VALIDATION PHASE (includes metric calculation)
        # ----------------------------------------------------------
        val_loss, val_metrics = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            epoch=current_epoch,
            print_every=10
        )

        print(
            f"[SUMMARY Epoch {current_epoch}] "
            f"train_loss = {train_loss:.4f} | "
            f"val_loss = {val_loss:.4f} | "
            f"mIoU = {val_metrics['mean_iou']:.4f} | "
            f"mPre = {val_metrics['mean_precision']:.4f} | "
            f"mRec = {val_metrics['mean_recall']:.4f}"
        )

        # ----------------------------------------------------------
        # BUILD EPOCH RECORD (GLOBAL METRICS)
        # ----------------------------------------------------------
        record = {
            "epoch": current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_iou": val_metrics["mean_iou"],
            "val_mean_precision": val_metrics["mean_precision"],
            "val_mean_recall": val_metrics["mean_recall"],
        }

        # ----------------------------------------------------------
        # ADD PER-CLASS METRICS
        # ----------------------------------------------------------
        iou_per_class   = val_metrics["iou_per_class"]
        pre_per_class  = val_metrics["precision_per_class"]
        rec_per_class   = val_metrics["recall_per_class"]

        for i in range(num_classes):
            # Choose readable class label when available
            if class_names is not None and i < len(class_names):
                class_label = class_names[i]
            else:
                class_label = f"class_{i}"

            # Store per-class metrics
            record[f"iou_{class_label}"] = float(iou_per_class[i])
            record[f"precision_{class_label}"] = float(pre_per_class[i])
            record[f"recall_{class_label}"] = float(rec_per_class[i])

        # Save metrics for this epoch
        history.append(record)

        # ----------------------------------------------------------
        # EARLY STOPPING + MODEL CHECKPOINTING
        # ----------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet.pth")
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping triggered!")
            break

    print("\nTraining finished!")
    print("****************************************")
    print("\n")

    # ----------------------------------------------------------
    # CONVERT LOGGED METRICS TO A DATAFRAME
    # ----------------------------------------------------------
    history_df = pd.DataFrame(history)

    return best_val_loss, history_df
