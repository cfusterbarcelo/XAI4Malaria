# training/loss.py

import torch.nn as nn


def get_loss_function(name):
    """
    Returns a PyTorch loss function based on a string name.

    Args:
        name (str): e.g., 'cross_entropy', 'sparse_categorical_crossentropy'

    Returns:
        nn.Module
    """
    name = name.lower()
    if name in ["cross_entropy", "sparse_categorical_crossentropy"]:
        return nn.CrossEntropyLoss()
    elif name == "bce":
        return nn.BCELoss()
    elif name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")
