"""Dataset loading and preprocessing."""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torch

from src.inference import get_train_transform, get_eval_transform


class ShapeDataset(Dataset):
    """Dataset that loads shape images from class-named subdirectories."""

    def __init__(self, data_dir, classes, transform=None):
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []

        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".png") and "(" not in fname:
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        self.class_to_idx[class_name],
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class ShapeDataModule(pl.LightningDataModule):
    """Lightning DataModule for shape classification."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["data"]["data_dir"]
        self.classes = config["data"]["classes"]
        self.image_size = config["data"]["image_size"]
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.train_split = config["data"]["train_split"]
        self.val_split = config["data"]["val_split"]

    def setup(self, stage=None):
        train_transform = get_train_transform(self.image_size)
        eval_transform = get_eval_transform(self.image_size)

        full_dataset = ShapeDataset(self.data_dir, self.classes, transform=None)
        n = len(full_dataset)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)
        n_test = n - n_train - n_val

        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset, test_subset = random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )

        self.train_dataset = TransformSubset(train_subset, train_transform)
        self.val_dataset = TransformSubset(val_subset, eval_transform)
        self.test_dataset = TransformSubset(test_subset, eval_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


class TransformSubset:
    """Wraps a Subset to apply a specific transform."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
