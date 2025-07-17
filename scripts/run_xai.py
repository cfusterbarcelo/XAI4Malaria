# scripts/run_xai.py

import os
import sys
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from data.xai_dataloader import TestCSVImageDataset
from data.transforms import get_preprocessing_pipeline
from models.model_factory import get_model
from explainability.gradcam import GradCAM
from utils.xai_utils import save_cam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_target_module(model, layer_str):
    parts = layer_str.split(".")
    module = model
    for p in parts:
        module = module[int(p)] if p.isdigit() else getattr(module, p)
    return module

def save_both(cam, orig_rgb, run_dir, filename, true_lbl, pred_lbl, alpha):
    """
    Save a resized heatmap and a translucent overlay,
    each in their own subfolder under run_dir.
    """
    heat_dir    = os.path.join(run_dir, "heatmap")
    overlay_dir = os.path.join(run_dir, "overlay")
    os.makedirs(heat_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # heatmap only (resized to original)
    save_cam(cam, orig_rgb, heat_dir, filename,
             true_lbl, pred_lbl,
             mode="heatmap", alpha=alpha)

    # overlay (resized heatmap + blend)
    save_cam(cam, orig_rgb, overlay_dir, filename,
             true_lbl, pred_lbl,
             mode="overlay", alpha=alpha)

def main():
    # load configs
    xai_cfg   = load_yaml("configs/xai_spcnn.yaml")
    paths_cfg = load_yaml("configs/paths.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name     = xai_cfg["model"]["run_name"]
    fold         = xai_cfg["model"]["fold"]
    method       = xai_cfg["xai"]["method"].lower()
    target_layer = xai_cfg["xai"]["target_layer"]

    image_size     = xai_cfg["data"]["image_size"]
    batch_size     = xai_cfg["data"]["batch_size"]
    apply_clahe    = xai_cfg["data"]["apply_clahe"]
    apply_dilation = xai_cfg["data"]["apply_dilation"]

    data_root    = paths_cfg["paths"]["data_root"]
    runs_dir     = paths_cfg["paths"]["model_runs_dir"]
    xai_base     = paths_cfg["paths"]["xai_runs_dir"]

    # build paths
    fold_dir        = os.path.join(runs_dir, run_name, f"fold_{fold}")
    csv_path        = os.path.join(fold_dir, "test_predictions.csv")
    checkpoint_path = os.path.join(fold_dir, "model.pth")
    xai_out_dir     = os.path.join(xai_base, run_name, f"fold_{fold}", method)
    os.makedirs(xai_out_dir, exist_ok=True)

    # loader
    preprocess = get_preprocessing_pipeline(
        resize=(image_size, image_size),
        apply_clahe=apply_clahe,
        apply_dilation=apply_dilation
    )
    dataset = TestCSVImageDataset(csv_path, data_root, transform=preprocess)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # model
    model = get_model(xai_cfg["model"])
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    new_state = {}
    for k,v in state.items():
        if k.startswith("features."):
            new_state[k.replace("features.", "feature_extractor.")] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()

    # explainer
    if method == "gradcam":
        tgt = resolve_target_module(model, target_layer)
        explainer = GradCAM(model, tgt, device=device)
    else:
        raise NotImplementedError

    print(f"🔍 Found {len(dataset)} images, {len(loader)} batches (bs={batch_size})")

    alpha=0.4  # more transparent
    for img_tensor, filename, true_lbl, pred_lbl in loader:
        fname = os.path.basename(filename[0])
        img   = img_tensor.to(device)

        cam = explainer.generate_heatmap(img, class_idx=int(pred_lbl[0]))

        orig_bgr = cv2.imread(os.path.join(data_root, filename[0]), cv2.IMREAD_COLOR)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        save_both(cam, orig_rgb, xai_out_dir, fname,
                  int(true_lbl[0]), int(pred_lbl[0]), alpha=alpha)

    print(f"✅ XAI results written to {xai_out_dir}")

if __name__=="__main__":
    main()
