import torch
from BostonDataset import BostonDataset
from torch.utils.data import DataLoader, random_split

class BostonDataModule:
    def __init__(self,
                 dataset,
                 batch_size: int = 4,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.0,
                 num_workers: int = 2,
                 shuffle_train: bool = True,
                 transform=None,
                 target_transform=None,
                 seed: int = 42,
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._setup()

    def _setup(self):

        dataset_size = len(self.dataset)

        # Compute sizes for train/val/test
        val_size = int(self.val_ratio * dataset_size)
        test_size = int(self.test_ratio * dataset_size)
        train_size = dataset_size - val_size - test_size

        if train_size <= 0:
            raise ValueError(
                f"Invalid split: train_size={train_size}. "
                f"Check val_ratio={self.val_ratio} and test_ratio={self.test_ratio}."
            )

        print("Splits:", train_size, val_size, test_size)

        # 3. Perform a deterministic random split using a fixed seed
        generator = torch.Generator().manual_seed(self.seed)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            lengths=[train_size, val_size, test_size],
            generator=generator
        )
        self._create_dataloaders()

    def _create_dataloaders(self):
        """
        Create DataLoaders for train/val/test subsets.
        """
        # Training DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Validation DataLoader (no shuffle)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Test DataLoader (no shuffle). Can be empty if test_ratio == 0.
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_loaders(self):
        """
        Return train, validation and test DataLoaders.

        Returns:
            (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader


    def get_datasets(self):
        """
        Return train, validation and test Datasets.

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        return self.train_dataset, self.val_dataset, self.test_dataset




