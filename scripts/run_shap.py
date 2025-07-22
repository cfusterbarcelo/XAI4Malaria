# scripts/run_shap.py

import os
import sys
import yaml
import torch
import cv2
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import numpy as np
from data.xai_dataloader import TestCSVImageDataset
from data.transforms    import get_preprocessing_pipeline
from models.model_factory import get_model
from utils.xai_utils    import save_cam
from explainability.shap import SHAPExplainer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_both(cam: np.ndarray,
              orig_rgb: np.ndarray,
              run_dir: str,
              filename: str,
              true_lbl: int,
              pred_lbl: int,
              alpha: float = 0.4):
    """
    Exactly like your Grad-CAM runner:
     - writes cam-only heatmap into run_dir/heatmap/
     - writes translucent overlay into run_dir/overlay/
    """
    heat_dir    = os.path.join(run_dir, "heatmap")
    overlay_dir = os.path.join(run_dir, "overlay")
    os.makedirs(heat_dir,    exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # 1) heatmap only
    save_cam(cam, orig_rgb, heat_dir, filename,
             true_lbl, pred_lbl,
             mode="heatmap", alpha=alpha)

    # 2) overlay
    save_cam(cam, orig_rgb, overlay_dir, filename,
             true_lbl, pred_lbl,
             mode="overlay", alpha=alpha)

def main():
    # 1) load configs
    xai_cfg   = load_yaml("configs/xai_shap.yaml")
    paths_cfg = load_yaml("configs/paths.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name       = xai_cfg["model"]["run_name"]
    fold           = xai_cfg["model"]["fold"]
    background_size= xai_cfg["xai"]["background_size"]
    explainer_type = xai_cfg["xai"]["explainer_type"].lower()
    nsamples       = xai_cfg["xai"]["nsamples"]

    image_size     = xai_cfg["data"]["image_size"]
    batch_size     = xai_cfg["data"]["batch_size"]
    apply_clahe    = xai_cfg["data"]["apply_clahe"]
    apply_dilation = xai_cfg["data"]["apply_dilation"]

    data_root    = paths_cfg["paths"]["data_root"]
    runs_dir     = paths_cfg["paths"]["model_runs_dir"]
    xai_base     = paths_cfg["paths"]["xai_runs_dir"]

    # 2) paths
    fold_dir        = os.path.join(runs_dir, run_name, f"fold_{fold}")
    csv_path        = os.path.join(fold_dir, "test_predictions.csv")
    checkpoint_path = os.path.join(fold_dir, "model.pth")
    xai_out_dir     = os.path.join(xai_base, run_name, f"fold_{fold}", "shap")
    os.makedirs(xai_out_dir, exist_ok=True)

    # 3) data loaders
    preprocess = get_preprocessing_pipeline(
        resize=(image_size, image_size),
        apply_clahe=apply_clahe,
        apply_dilation=apply_dilation
    )
    test_dataset = TestCSVImageDataset(csv_path, data_root, transform=preprocess)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # background for SHAP
    bg_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    bg_samples = []
    for img_tensor, *_ in bg_loader:
        bg_samples.append(img_tensor)
        if len(bg_samples) >= background_size:
            break
    background = torch.cat(bg_samples, dim=0).to(device)

    # 4) model
    model = get_model(xai_cfg["model"])
    model.to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    new_state = {}
    for k,v in state.items():
        if k.startswith("features."):
            new_state[k.replace("features.","feature_extractor.")] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)

    # 5) explainer
    explainer = SHAPExplainer(
        model,
        background,
        device,
        explainer_type=explainer_type,
        nsamples=nsamples
    )

    print(f"🔍 Running SHAP on {len(test_dataset)} images (bg={background_size}, samples={nsamples})")

    # 6) loop & save
    alpha = 0.4
    for img_tensor, filename, true_lbl, pred_lbl in test_loader:
        fname = os.path.basename(filename[0])
        img   = img_tensor.to(device)

        shap_map = explainer(img, class_idx=int(pred_lbl[0]))
        cam_raw  = shap_map[0].detach().cpu().numpy()
        cam      = gaussian_filter(cam_raw, sigma=1.5)

        orig_bgr = cv2.imread(os.path.join(data_root, filename[0]), cv2.IMREAD_COLOR)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        save_both(cam, orig_rgb, xai_out_dir, fname,
                  int(true_lbl[0]), int(pred_lbl[0]), alpha=alpha)

    print(f"✅ SHAP results written to {xai_out_dir}")

if __name__ == "__main__":
    main()
