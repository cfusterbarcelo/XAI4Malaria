# data/datasets.py

import torch
from torchvision import datasets
from torch.utils.data import random_split
from data.transforms import get_preprocessing_pipeline
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torchvision import datasets
from data.transforms import get_preprocessing_pipeline

def load_malaria_dataset(
    data_root,
    split_ratios=(0.8, 0.1, 0.1),
    resize=(32, 32),
    apply_clahe=True,
    apply_dilation=True,
    seed=42
):
    """
    Dynamically splits the raw malaria dataset into train, val, and test sets.

    Args:
        data_root (str): Path to dataset with `Parasitized/` and `Uninfected/` subfolders.
        split_ratios (tuple): Ratios for train, val, and test (must sum to 1.0).
        resize (tuple): Image resize dimensions (default: 32x32).
        apply_clahe (bool): Whether to apply CLAHE.
        apply_dilation (bool): Whether to apply dilation.
        seed (int): Random seed for reproducibility.

    Returns:
        train_dataset (Dataset)
        val_dataset (Dataset)
        test_dataset (Dataset)
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    transform = get_preprocessing_pipeline(resize, apply_clahe, apply_dilation)

    # Load entire dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=data_root, transform=transform)
    total_size = len(full_dataset)

    # Compute split lengths
    train_size = int(split_ratios[0] * total_size)
    val_size = int(split_ratios[1] * total_size)
    test_size = total_size - train_size - val_size  # ensure no rounding loss

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    return train_dataset, val_dataset, test_dataset

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
    full_dataset = datasets.ImageFolder(root=data_root, transform=transform)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = []

    for train_indices, val_indices in kf.split(full_dataset):
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        folds.append((train_dataset, val_dataset))

    return folds