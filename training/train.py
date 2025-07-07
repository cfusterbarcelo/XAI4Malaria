# training/train.py

import os
import torch
from training.eval import evaluate_model
from utils.logging import print_and_log


def train_one_fold(
    fold_idx,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    metrics_fn,
    device,
    num_epochs,
    output_dir,
    save_model=True,
    log_file=None
):
    """
    Trains the model for one fold and evaluates on validation set.

    Args:
        fold_idx (int): Index of current fold
        model (nn.Module): Model to train
        train_loader (DataLoader)
        val_loader (DataLoader)
        optimizer (torch.optim.Optimizer)
        loss_fn (nn.Module)
        metrics_fn (callable): Function to compute metrics
        device (torch.device)
        num_epochs (int)
        output_dir (str): Folder to store logs and models
        save_model (bool)
        log_file (file handle or None)

    Returns:
        dict: Final metrics
    """
    model.to(device)
    best_val_acc = 0.0
    best_model_path = None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        
        # Track for plotting
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        
        avg_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, loss_fn, metrics_fn, device)

        msg = (
            f"[Fold {fold_idx}] Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )
        print_and_log(msg, log_file)

        # Save best model
        if save_model and val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            model_folder = os.path.join(output_dir, f"fold_{fold_idx}")
            os.makedirs(model_folder, exist_ok=True)
            best_model_path = os.path.join(model_folder, "model.pth")
            torch.save(model.state_dict(), best_model_path)

    return {
        "fold": fold_idx,
        "best_val_acc": best_val_acc,
        "metrics": val_metrics,
        "model_path": best_model_path,
        "history": history
    }
