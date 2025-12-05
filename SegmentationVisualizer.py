import numpy as np
import torch
import matplotlib.pyplot as plt


class SegmentationVisualizer:
    """
    Helper class to visualize semantic segmentation results.

    This class:
    - Receives class names and a color map (class_name -> (R, G, B)).
    - Can decode masks of class indices into RGB color images.
    - Can display the original image, ground truth mask, and predicted mask
      side by side.
    """

    def __init__(self, class_names, class_color_map):
        """
        Args:
            class_names (list of str): List of class names. The index in this list
                is the class index used in the masks (0..num_classes-1).
            class_color_map (dict): Mapping {class_name: (R, G, B)} for each class.
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

        Assumes the input tensor is already normalized to [0, 1] if [C, H, W].
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
            # If it is already a numpy array
            img_np = np.array(image_tensor).astype("float32")
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

        # Ensure values are clipped to [0, 1]
        img_np = np.clip(img_np, 0.0, 1.0)
        return img_np

    @staticmethod
    def _to_numpy_mask_indices(mask_tensor):
        """
        Convert a mask tensor (either [H, W] of indices or [C, H, W] logits)
        into a 2D numpy array [H, W] of class indices.
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

        for class_idx, color in enumerate(self.class_colors):
            color_mask[mask_np == class_idx] = color  # color is (R, G, B)

        return color_mask

    def show_triplet(self, image_tensor, true_mask_tensor, pred_mask_tensor=None,
                     figsize=(15, 5), suptitle=None):
        """
        Show the original image, the ground truth mask, and the predicted mask
        side by side. If pred_mask_tensor is None, a blank black mask is shown.
        """

        # Convert image
        img_np = self._to_numpy_image(image_tensor)

        # Ground truth mask (decoded to color)
        true_color = self.decode_mask_to_color(true_mask_tensor)
        true_color = true_color.astype("float32") / 255.0

        # Handle predicted mask
        if pred_mask_tensor is None:
            # Create a black mask same size as true_mask
            h, w = true_mask_tensor.shape
            pred_color = np.zeros((h, w, 3), dtype="float32")  # all black
        else:
            pred_color = self.decode_mask_to_color(pred_mask_tensor)
            pred_color = pred_color.astype("float32") / 255.0

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        # Ground truth mask
        axes[1].imshow(true_color)
        axes[1].set_title("Ground truth mask")
        axes[1].axis("off")

        # Predicted mask or blank
        axes[2].imshow(pred_color)
        axes[2].set_title("Predicted mask" if pred_mask_tensor is not None else "Predicted mask (blank)")
        axes[2].axis("off")

        if suptitle is not None:
            fig.suptitle(suptitle)

        plt.tight_layout()
        plt.show()
