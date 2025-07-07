# data/dataloader_factory.py

from torch.utils.data import DataLoader
from data.datasets import load_malaria_dataset


def get_dataloaders(
    data_root,
    batch_size=32,
    split_ratios=(0.8, 0.1, 0.1),
    resize=(32, 32),
    apply_clahe=True,
    apply_dilation=True,
    num_workers=4,
    seed=42
):
    """
    Creates train, validation, and test dataloaders for malaria classification.

    Args:
        data_root (str): Path to dataset with Parasitized/ and Uninfected/
        batch_size (int): Batch size for all DataLoaders
        split_ratios (tuple): Proportions for train, val, test (must sum to 1.0)
        resize (tuple): Resize target for input images
        apply_clahe (bool): Whether to apply CLAHE
        apply_dilation (bool): Whether to apply dilation
        num_workers (int): Workers for parallel loading
        seed (int): Seed for random splitting

    Returns:
        (train_loader, val_loader, test_loader): Tuple of DataLoaders
    """
    train_set, val_set, test_set = load_malaria_dataset(
        data_root=data_root,
        split_ratios=split_ratios,
        resize=resize,
        apply_clahe=apply_clahe,
        apply_dilation=apply_dilation,
        seed=seed
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def get_kfold_dataloaders(
    data_root,
    k_folds=5,
    batch_size=32,
    resize=(32, 32),
    apply_clahe=True,
    apply_dilation=True,
    num_workers=4,
    seed=42
):
    """
    Creates k-fold (train_loader, val_loader) pairs for cross-validation.

    Args:
        data_root (str): Path to ImageFolder-style dataset
        k_folds (int): Number of folds (default: 5)
        batch_size (int): Batch size for DataLoaders
        resize (tuple): Resize size for images
        apply_clahe (bool): Use CLAHE preprocessing
        apply_dilation (bool): Use dilation preprocessing
        num_workers (int): Data loading workers
        seed (int): Random seed

    Returns:
        List of (train_loader, val_loader) tuples for each fold
    """
    folds = get_kfold_datasets(
        data_root=data_root,
        k_folds=k_folds,
        resize=resize,
        apply_clahe=apply_clahe,
        apply_dilation=apply_dilation,
        seed=seed
    )

    fold_loaders = []
    for train_dataset, val_dataset in folds:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders
