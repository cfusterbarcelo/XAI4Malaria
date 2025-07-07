# utils/visualization.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision.utils import save_image
import torch


def plot_training_curves(history, save_path):
    """
    Plots training/validation loss and accuracy over epochs.
    """
    epochs = history["epoch"]
    plt.figure(figsize=(10, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path, normalize=True):
    """
    Plots and saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_misclassified_examples(dataset, y_true, y_pred, output_dir, class_names=None):
    """
    Saves FP and FN images into separate folders for visual inspection.
    Only works with Subset datasets where dataset[i] returns (image, label).

    Args:
        dataset (Subset or ImageFolder): Dataset used during eval
        y_true (list[int]): Ground truth labels
        y_pred (list[int]): Predicted labels
        output_dir (str): Output path to save misclassified images
        class_names (list): Optional class label names
    """
    os.makedirs(output_dir, exist_ok=True)
    false_positives = []
    false_negatives = []

    for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            image, label = dataset[idx]  # image: tensor
            image_folder = "false_positives" if pred == 1 and true == 0 else "false_negatives"
            save_folder = os.path.join(output_dir, image_folder)
            os.makedirs(save_folder, exist_ok=True)

            fname = f"idx{idx}_pred{pred}_true{true}.png"
            save_path = os.path.join(save_folder, fname)
            save_image(image, save_path)
