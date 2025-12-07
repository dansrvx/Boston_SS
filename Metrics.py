import torch
import numpy as np


class Metrics:
    """
    Utility class for computing semantic segmentation metrics using
    a confusion matrix accumulated over batches.

    Supported metrics:
        - IoU (Intersection over Union)
        - Precision
        - Recall

    The class supports:
        - Per-class metrics
        - Mean metrics
        - Ignored labels via `ignore_index`
    """

    def __init__(self, num_classes: int, ignore_index: int | None = None, device: str = "cpu"):
        """
        Initialize the Metrics object.

        Args:
            num_classes: Total number of semantic classes.
            ignore_index: Label value to be ignored when computing metrics
                          (e.g., 255 in many segmentation datasets).
            device: Device where the confusion matrix will be stored ('cpu' or 'cuda').
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.confusion_matrix = self.reset()

    def reset(self):
        """
        Reset the internal confusion matrix to all zeros.

        Returns:
            A zero-initialized confusion matrix of shape [num_classes, num_classes].
        """
        cm = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device
        )
        return cm

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update the confusion matrix with a new batch of predictions.

        Args:
            logits: Raw model outputs of shape [B, C, H, W].
                    C = num_classes.
            targets: Ground-truth labels of shape [B, H, W].
                    Each pixel contains a class index in [0, num_classes-1]
                    or ignore_index.
        """

        # --------------------------------------------------
        # 1. Compute predicted class per pixel → [B, H, W]
        # --------------------------------------------------
        predictions = torch.argmax(logits, dim=1)

        # --------------------------------------------------
        # 2. Flatten predictions and targets → [B*H*W]
        # --------------------------------------------------
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # --------------------------------------------------
        # 3. Remove ignored pixels, if specified
        # --------------------------------------------------
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]

        # --------------------------------------------------
        # 4. Remove invalid target values (outside class range)
        # --------------------------------------------------
        valid_mask = (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        # --------------------------------------------------
        # 5. Compute flat indices for bincount
        #    index = target_class * num_classes + predicted_class
        # --------------------------------------------------
        indices = self.num_classes * targets + predictions
        cm = torch.bincount(
            indices,
            minlength=self.num_classes ** 2
        )

        # --------------------------------------------------
        # 6. Reshape to [num_classes, num_classes] and accumulate
        # --------------------------------------------------
        cm = cm.view(self.num_classes, self.num_classes)
        self.confusion_matrix += cm

    def compute(self):
        """
        Compute IoU, Precision, and Recall from the accumulated confusion matrix.

        Definitions:
            TP = True Positives = diagonal of confusion matrix
            FP = column sum minus TP
            FN = row sum minus TP

        Returns:
            dict containing:
                - IoU per class
                - Precision per class
                - Recall per class
                - Mean IoU
                - Mean Precision
                - Mean Recall
        """

        cm = self.confusion_matrix.float()

        # True Positives → diagonal of confusion matrix
        tp = torch.diag(cm)

        # Actual ground-truth pixels per class → row sum
        actual = cm.sum(dim=1)

        # Predicted pixels per class → column sum
        predicted = cm.sum(dim=0)

        # IoU denominator = TP + FP + FN = actual + predicted - TP
        union = actual + predicted - tp

        # Compute metrics safely using clamped denominators
        iou_per_class = tp / torch.clamp(union, min=1.0)
        precision_per_class = tp / torch.clamp(predicted, min=1.0)
        recall_per_class = tp / torch.clamp(actual, min=1.0)

        # Aggregate mean metrics
        mean_iou = iou_per_class.mean().item()
        mean_precision = precision_per_class.mean().item()
        mean_recall = recall_per_class.mean().item()

        return {
            "iou_per_class": iou_per_class.cpu(),
            "precision_per_class": precision_per_class.cpu(),
            "recall_per_class": recall_per_class.cpu(),
            "mean_iou": mean_iou,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
        }
