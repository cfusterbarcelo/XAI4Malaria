import os
import cv2
import numpy as np
from PIL import Image

def resolve_layer(model, path):
    cur = model
    for p in path.split('.'):
        cur = getattr(cur, p) if not p.isdigit() else cur[int(p)]
    return cur

def overlay_heatmap(orig_pil: Image.Image,
                    cam: np.ndarray,  # assumed shape [Hcam,Wcam], floats 0–255 or 0–1
                    alpha: float = 0.2):
    """
    Resize cam to orig size, colorize it, and blend with the original PIL image.
    """
    # convert PIL to BGR array
    orig_np = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
    H, W = orig_np.shape[:2]

    # normalize cam to 0–255 uint8
    cam_norm = cam.astype(np.float32)
    cam_norm -= cam_norm.min()
    cam_norm /= cam_norm.max()
    cam_u8   = (cam_norm*255).astype(np.uint8)

    # resize up to match the original
    cam_rs = cv2.resize(cam_u8, (W, H), interpolation=cv2.INTER_LINEAR)

    # color-map it
    heatmap = cv2.applyColorMap(cam_rs, cv2.COLORMAP_JET)

    # blend
    overlay_bgr = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap, alpha, 0)

    # back to PIL RGB
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)


def dispatch_xai_explainer(method, model, target_module, device):
    import importlib
    mapping = {
      'gradcam':   ('gradcam','GradCAM'),
      'gradcam++': ('gradcampp','GradCAMPlusPlus'),
      'scorecam':  ('scorecam','ScoreCAM'),
      # …and any future ones
    }
    if method not in mapping:
      raise ValueError(f"Unknown XAI method {method}")
    mod_name, cls_name = mapping[method]
    mod = importlib.import_module(f'explainability.{mod_name}')
    ExplClass = getattr(mod, cls_name)
    return ExplClass(model, target_module, device)

def save_cam(cam: np.ndarray,
             orig_rgb: np.ndarray,
             out_dir: str,
             filename: str,
             true_label: int,
             pred_label: int,
             mode: str = "overlay",
             alpha: float = 0.4):
    """
    cam: H×W or H×W×C float in [0,1]
    orig_rgb: H×W×3 uint8 (RGB)
    mode: "raw", "heatmap", or "overlay"
    alpha: blending weight for heatmap when overlaying
    """
    base, ext = os.path.splitext(filename)
    out_fname = f"{base}_true{true_label}_pred{pred_label}{ext}"
    out_path = os.path.join(out_dir, out_fname)
    
    H, W, _ = orig_rgb.shape

    # 1) raw float map
    if mode == "raw":
        cv2.imwrite(out_path, (cam * 255).astype(np.uint8))
        return out_path

    # 2) percentile‐based stretch + uint8 map
    low_p, high_p = 5, 99
    p_low, p_high = np.percentile(cam, [low_p, high_p])
    cam_clipped = np.clip(cam, p_low, p_high)
    # rescale that window to [0,1]
    cam_norm = (cam_clipped - p_low) / (p_high - p_low + 1e-8)
    p_high     = np.percentile(cam, 99)
    cam_clipped = np.minimum(cam, p_high)
    cam_norm    = (cam_clipped - cam.min()) / (p_high - cam.min() + 1e-8)
    heatmap_u8  = (cam_norm * 255).astype(np.uint8)

    # collapse any extra channel dimension
    if heatmap_u8.ndim == 3:
        # if it's H×W×1, squeeze; else average across channels
        if heatmap_u8.shape[2] == 1:
            heatmap_u8 = heatmap_u8[:, :, 0]
        else:
            # e.g. SHAP gave two channels—take the absolute‐mean
            heatmap_u8 = np.mean(np.abs(heatmap_u8), axis=2).astype(np.uint8)
    elif heatmap_u8.ndim != 2:
        raise ValueError(f"Unexpected heatmap shape: {heatmap_u8.shape}")

    # apply the colormap
    heat_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    if mode == "heatmap":
        heat_resized = cv2.resize(heat_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_path, heat_resized)
        return out_path

    # 3) overlay
    orig_bgr     = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    heat_resized = cv2.resize(heat_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    overlay_bgr  = cv2.addWeighted(orig_bgr, 1 - alpha, heat_resized, alpha, 0)
    cv2.imwrite(out_path, overlay_bgr)
    return out_path
