# scripts/train_model.py

import os
import yaml
import torch
import datetime
from models.model_factory import get_model
from training.train import train_one_fold
from training.metrics import classification_metrics
from training.loss import get_loss_function  # You’ll create this later
from data.dataloader_factory import get_kfold_dataloaders
from utils.visualization import plot_training_curves


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_run_folder(base_dir, model_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{model_name.upper()}"
    run_dir = os.path.join(base_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config_copy(config_dicts, save_path):
    merged = {}
    for cfg in config_dicts:
        merged.update(cfg)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.safe_dump(merged, f)


def main():
    # === Load configs ===
    model_cfg = load_yaml_config("configs/model_spcnn.yaml")
    train_cfg = load_yaml_config("configs/train_default.yaml")
    paths_cfg = load_yaml_config("configs/paths.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_cfg["model"]["name"]
    data_root = paths_cfg["paths"]["data_root"]
    output_root = paths_cfg["paths"]["output_root"]
    save_model = paths_cfg["paths"].get("save_model", True)
    save_logs = paths_cfg["paths"].get("save_logs", True)

    run_dir = create_run_folder(output_root, model_name)
    save_config_copy([model_cfg, train_cfg, paths_cfg], run_dir)

    # === Dataloaders ===
    folds = get_kfold_dataloaders(
        data_root=data_root,
        k_folds=5,
        batch_size=train_cfg["train"]["batch_size"],
        resize=model_cfg["model"]["input_shape"][1:],  # (H, W)
        apply_clahe=True,
        apply_dilation=True,
        num_workers=4,
        seed=42
    )

    all_results = []

    # === Train each fold ===
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        print(f"\n===== Fold {fold_idx + 1}/5 =====")

        # === Model, Optimizer, Loss ===
        model = get_model(model_cfg["model"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["train"]["learning_rate"]
        )
        loss_fn = get_loss_function(train_cfg["train"]["loss"])

        # === Logging ===
        fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        log_file = open(os.path.join(fold_dir, "log.txt"), "w") if save_logs else None

        result = train_one_fold(
            fold_idx=fold_idx,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics_fn=classification_metrics,
            device=device,
            num_epochs=train_cfg["train"]["epochs"],
            output_dir=run_dir,
            save_model=save_model,
            log_file=log_file
        )

        # Save training curves
        if paths_cfg["paths"].get("save_figures", True):
            plot_path = os.path.join(fold_dir, "training_plot.png")
            plot_training_curves(result["history"], save_path=plot_path)

        # Save metrics
        with open(os.path.join(fold_dir, "metrics.txt"), "w") as f:
            for k, v in result["metrics"].items():
                f.write(f"{k}: {v:.4f}\n")

        all_results.append(result)

        if log_file:
            log_file.close()

    print("\n✅ Training complete across all folds.")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
