import torch
from BostonFiles import BostonFiles
from BostonDataset import BostonDataset
from BostonDataModule import BostonDataModule
from SegmentationVisualizer import SegmentationVisualizer
import os

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    files = BostonFiles('1T572f0oqy5JmuTvVEwkSUeXLWOSHl4hL',
                        '1pHp480_Q-s72RoDf1nD7ERzsv9yZTDE1')

    files.download()
    files.unzip()

    current_path = os.getcwd()
    img_dir = os.path.join(current_path, "images", "Images")
    msk_dir = os.path.join(current_path, "masks", "Segmentation_Masks")

    dataset = BostonDataset(img_dir=img_dir, msk_dir=msk_dir)
    print("Dataset size:", len(dataset))

    data_module = BostonDataModule(dataset)
    trn_dataset, val_dataset, tst_dataset = data_module.get_datasets()
    trn_loader, val_loader, tst_loader = data_module.get_loaders()

    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Batch statistics")
    print("Train batches:", len(trn_loader))
    print("Val batches  :", len(val_loader))
    print("Test batches :", len(tst_loader))
    print("**-**-**-**-**-**-**-**-**-**-**-**")

    visualizer = SegmentationVisualizer(class_names=BostonDataset.CLASSES,
                                        class_color_map=BostonDataset.CLASS_COLOR_MAP)
    print("**-**-**-**-**-**-**-**-**-**-**-**")
    print("Image examples & statistics")
    images, masks = next(iter(trn_loader))
    visualizer.show_triplet(images[0], masks[0], suptitle="Random test image & mask")
    print("Test image shape:", images.shape)  # [B, 3, H, W]
    print("Test mask shape :", masks.shape)  # [B, H, W]
    images, masks = next(iter(val_loader))
    visualizer.show_triplet(images[0], masks[0], suptitle="Random validation image & mask")
    print("Validation image shape:", images.shape)  # [B, 3, H, W]
    print("Validation mask shape :", masks.shape)  # [B, H, W]


if __name__ == '__main__':
    main()


