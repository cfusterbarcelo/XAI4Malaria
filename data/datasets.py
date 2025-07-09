# data/datasets.py

import torch
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from data.transforms import get_preprocessing_pipeline
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torchvision import datasets
from data.transforms import get_preprocessing_pipeline

class ImageFolderWithPaths(ImageFolder):
    """
    Custom ImageFolder that returns (image_tensor, label, image_path).
    """
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path

def get_kfold_datasets(
    data_root,
    k_folds=5,
    resize=(32, 32),
    apply_clahe=True,
    apply_dilation=True,
    seed=42
):
    """
    Generates k-fold training and validation dataset splits for cross-validation.

    Args:
        data_root (str): Path to dataset with Parasitized/ and Uninfected/
        k_folds (int): Number of folds (default: 5)
        resize (tuple): Resize dimensions
        apply_clahe (bool): Whether to apply CLAHE
        apply_dilation (bool): Whether to apply morphological dilation
        seed (int): Seed for reproducibility

    Returns:
        List of (train_dataset, val_dataset) tuples, one per fold
    """
    transform = get_preprocessing_pipeline(resize, apply_clahe, apply_dilation)
    full_dataset = ImageFolderWithPaths(root=data_root, transform=transform)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = []

    for train_indices, val_indices in kf.split(full_dataset):
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        folds.append((train_dataset, val_dataset))

    return folds