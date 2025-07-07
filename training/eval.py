# training/eval.py

import torch
from tqdm import tqdm


def evaluate_model(model, dataloader, loss_fn, metrics_fn, device, return_preds=False):
    """
    Evaluates the model on the given dataloader.

    Args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): The data loader (val or test)
        loss_fn (callable): Loss function
        metrics_fn (callable): Metric computation function
        device (torch.device)

    Returns:
        dict: {
            'loss': float,
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'auc': float (if computable)
        }
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (images, labels) in tqdm(enumerate(dataloader), desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if return_preds:
                all_indices.extend(range(len(all_preds)))

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = metrics_fn(all_labels, all_preds)
    metrics["loss"] = avg_loss

    if return_preds:
        return metrics, all_labels, all_preds, all_indices
    else:
        return metrics
