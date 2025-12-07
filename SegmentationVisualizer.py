import numpy as np
import torch
import matplotlib.pyplot as plt


class SegmentationVisualizer:
    """
    Helper class to visualize semantic segmentation results.

    This class:
        - Receives class names and a color map (class_name -> (R, G, B)).
        - Can decode masks of class indices into RGB color images.
        - Can display:
            * the original image
            * the ground truth mask
            * the predicted mask
            * a match/mismatch map (green = correct, red = incorrect)
    """

    def __init__(self, class_names, class_color_map):
        """
        Initialize the visualizer with class labels and colors.

        Args:
            class_names (list of str): List of class names. The index in this list
                is the class index used in the masks (0..num_classes-1).
            class_color_map (dict): Mapping {class_name: (R, G, B)} for each class.
                                    Color channels are expected in [0, 255].
        """
        self.class_names = class_names
        self.class_color_map = class_color_map

        # Precompute the colors in the same order as class_names
        self.class_colors = [self.class_color_map[c] for c in self.class_names]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy_image(image_tensor):
        """
        Convert an image tensor [C, H, W] or [H, W, C] into a numpy array
        [H, W, C] in range [0, 1] suitable for matplotlib.

        Assumptions:
            - If input is [C, H, W] and values are in [0, 1], they are used directly.
            - If values are in [0, 255], they are normalized to [0, 1].
        """
        if isinstance(image_tensor, torch.Tensor):
            img = image_tensor.detach().cpu()

            # If shape is [C, H, W], permute to [H, W, C]
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)  # [H, W, C]

            img_np = img.numpy().astype("float32")

            # If image is in [0, 255], normalize to [0, 1]
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

        else:
            # If it is already a numpy array or list-like
            img_np = np.array(image_tensor).astype("float32")
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

        # Ensure values are clipped to [0, 1]
        img_np = np.clip(img_np, 0.0, 1.0)
        return img_np

    @staticmethod
    def _to_numpy_mask_indices(mask_tensor):
        """
        Convert a mask tensor into a 2D numpy array [H, W] of class indices.

        Supported formats:
            - [H, W] with integer class indices.
            - [C, H, W] with logits or probabilities (argmax over channel dimension).
        """
        if isinstance(mask_tensor, torch.Tensor):
            mask = mask_tensor.detach().cpu()

            # If mask has shape [C, H, W], assume it is logits or probabilities
            # and take argmax over channels.
            if mask.ndim == 3 and mask.shape[0] > 1:
                mask = torch.argmax(mask, dim=0)  # [H, W]

            # Now we expect [H, W] with integer indices
            if mask.ndim != 2:
                raise ValueError(
                    "Mask tensor must be [H, W] or [C, H, W] (logits). "
                    f"Got shape: {tuple(mask.shape)}"
                )

            mask_np = mask.numpy().astype(np.int64)

        else:
            mask_np = np.array(mask_tensor).astype(np.int64)

        return mask_np

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def decode_mask_to_color(self, mask_tensor):
        """
        Convert a mask of class indices [H, W] (or logits [C, H, W]) into a
        color image [H, W, 3] (uint8) using the predefined color map.

        Args:
            mask_tensor (Tensor or np.ndarray):
                - [H, W] with class indices, or
                - [C, H, W] with logits or probabilities.

        Returns:
            color_mask (np.ndarray): RGB image [H, W, 3], dtype uint8.
        """
        mask_np = self._to_numpy_mask_indices(mask_tensor)

        h, w = mask_np.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Assign RGB color for each class index
        for class_idx, color in enumerate(self.class_colors):
            color_mask[mask_np == class_idx] = color  # color is (R, G, B) in [0, 255]

        return color_mask

    def _compute_match_mismatch_map(self, true_mask_tensor, pred_mask_tensor):
        """
        Create a match/mismatch RGB map comparing ground truth and predictions.

        For each pixel:
            - Green  (0, 1, 0) if predicted class == true class.
            - Red    (1, 0, 0) if predicted class != true class.

        Args:
            true_mask_tensor: Ground truth mask [H, W] or [C, H, W].
            pred_mask_tensor: Predicted mask [H, W] or [C, H, W].

        Returns:
            diff_map (np.ndarray): Float RGB image [H, W, 3] in [0, 1].
        """
        true_indices = self._to_numpy_mask_indices(true_mask_tensor)
        pred_indices = self._to_numpy_mask_indices(pred_mask_tensor)

        if true_indices.shape != pred_indices.shape:
            raise ValueError(
                f"Shape mismatch between true and predicted masks: "
                f"{true_indices.shape} vs {pred_indices.shape}"
            )

        h, w = true_indices.shape
        diff_map = np.zeros((h, w, 3), dtype=np.float32)

        # Boolean mask of correct predictions
        match = (true_indices == pred_indices)

        # Correct pixels → green
        diff_map[match] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # Incorrect pixels → red
        diff_map[~match] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        return diff_map

    def show_triplet(
        self,
        image_tensor,
        true_mask_tensor,
        pred_mask_tensor=None,
        figsize=(18, 5),
        suptitle=None,
    ):
        """
        Show the original image, the ground truth mask, the predicted mask,
        and (if prediction is provided) a match/mismatch map.

        If pred_mask_tensor is None:
            - Only three panels are shown:
                * Image
                * Ground truth mask
                * Blank predicted mask (black)
        If pred_mask_tensor is not None:
            - Four panels are shown:
                * Image
                * Ground truth mask
                * Predicted mask
                * Match/mismatch (green = correct, red = incorrect)

        Args:
            image_tensor: Input image [C, H, W], [H, W, C], or numpy array.
            true_mask_tensor: Ground truth mask [H, W] or [C, H, W].
            pred_mask_tensor: Predicted mask [H, W] or [C, H, W], or None.
            figsize: Figure size for matplotlib.
            suptitle: Optional global title for the figure.
        """

        # --------------------------------------------------------------
        # Prepare input image for visualization
        # --------------------------------------------------------------
        img_np = self._to_numpy_image(image_tensor)

        # Ground truth mask (decoded to color, normalized to [0, 1])
        true_color = self.decode_mask_to_color(true_mask_tensor).astype("float32") / 255.0

        # --------------------------------------------------------------
        # Handle predicted mask and match/mismatch map
        # --------------------------------------------------------------
        if pred_mask_tensor is None:
            # Create a black mask same size as true_mask
            h, w = self._to_numpy_mask_indices(true_mask_tensor).shape
            pred_color = np.zeros((h, w, 3), dtype="float32")  # all black
            diff_map = None
            num_panels = 3
        else:
            pred_color = self.decode_mask_to_color(pred_mask_tensor).astype("float32") / 255.0
            diff_map = self._compute_match_mismatch_map(true_mask_tensor, pred_mask_tensor)
            num_panels = 4

        # --------------------------------------------------------------
        # Create subplots: 3 or 4 depending on whether we have predictions
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, num_panels, figsize=figsize)

        # When num_panels == 3, axes is an array of length 3
        # When num_panels == 4, axes is an array of length 4
        if num_panels == 3:
            ax_img, ax_true, ax_pred = axes
        else:
            ax_img, ax_true, ax_pred, ax_diff = axes

        # Original image
        ax_img.imshow(img_np)
        ax_img.set_title("Image")
        ax_img.axis("off")

        # Ground truth mask
        ax_true.imshow(true_color)
        ax_true.set_title("Ground truth mask")
        ax_true.axis("off")

        # Predicted mask or blank
        ax_pred.imshow(pred_color)
        if pred_mask_tensor is not None:
            ax_pred.set_title("Predicted mask")
        else:
            ax_pred.set_title("Predicted mask (blank)")
        ax_pred.axis("off")

        # Match/mismatch map (only if we have predictions)
        if pred_mask_tensor is not None and diff_map is not None:
            ax_diff.imshow(diff_map)
            ax_diff.set_title("Match / Mismatch\n(green = correct, red = wrong)")
            ax_diff.axis("off")

        if suptitle is not None:
            fig.suptitle(suptitle)

        plt.tight_layout()
        plt.show()
