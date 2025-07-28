#!/usr/bin/env python3
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import cv2
from data.xai_dataloader import TestCSVImageDataset
from data.transforms import get_preprocessing_pipeline
from models.model_factory import get_model
from explainability.lime import LimeExplainer

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_yaml(path):
    # Read YAML with UTF-8 encoding to avoid decode errors on Windows
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 1) load configs
    xai_cfg   = load_yaml("configs/xai_lime.yaml")
    paths_cfg = load_yaml("configs/paths.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name    = xai_cfg["model"]["run_name"]
    fold        = xai_cfg["model"]["fold"]
    num_samples = xai_cfg["xai"]["num_samples"]
    hide_color  = xai_cfg["xai"].get("hide_color", 0)
    num_features= xai_cfg["xai"].get("num_features", 5)
    top_labels  = xai_cfg["xai"].get("top_labels", 1)

    image_size    = xai_cfg["data"]["image_size"]
    batch_size    = xai_cfg["data"]["batch_size"]
    apply_clahe   = xai_cfg["data"]["apply_clahe"]
    apply_dilation= xai_cfg["data"]["apply_dilation"]

    data_root     = paths_cfg["paths"]["data_root"]
    runs_dir      = paths_cfg["paths"]["model_runs_dir"]
    xai_base      = paths_cfg["paths"]["xai_runs_dir"]

    # 2) paths
    fold_dir        = os.path.join(runs_dir, run_name, f"fold_{fold}")
    csv_path        = os.path.join(fold_dir, "test_predictions.csv")
    checkpoint_path = os.path.join(fold_dir, "model.pth")
    xai_out_dir     = os.path.join(xai_base, run_name, f"fold_{fold}", "lime")
    os.makedirs(xai_out_dir, exist_ok=True)

    # sanity check: CSV exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Couldn’t find predictions CSV at {csv_path}")

    # 3) data loader
    preprocess   = get_preprocessing_pipeline(
        resize=(image_size, image_size),
        apply_clahe=apply_clahe,
        apply_dilation=apply_dilation
    )
    test_dataset = TestCSVImageDataset(csv_path, data_root, transform=preprocess)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 4) model
    model = get_model(xai_cfg["model"])
    model.to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state      = checkpoint.get("state_dict", checkpoint)
    new_state  = {}
    for k, v in state.items():
        if k.startswith("features."):
            new_state[k.replace("features.", "feature_extractor.")] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)

    # 5) explainer
    explainer = LimeExplainer(
        model=model,
        transform=preprocess,
        output_dir=xai_out_dir,
        num_samples=num_samples,
        batch_size=1
    )

    print(f"🔍 Running LIME on {len(test_dataset)} images (samples={num_samples})")

    # 6) loop & save
    for img_tensor, filename, true_lbl, pred_lbl in test_loader:
        img_rel_path = filename[0]
        img_path     = os.path.join(data_root, img_rel_path)
        explainer.explain_image(
            img_path=img_path,
            label=int(pred_lbl[0]),
            hide_color=hide_color,
            num_features=num_features,
            top_labels=top_labels
        )

    print(f"✅ LIME results written to {xai_out_dir}")

if __name__ == "__main__":
    main()
