# scripts/plot_confusion_examples.py

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
CSV_PATH              = r"D:\Results\XAI4Malaria\runs\2025-07-10_10-47-13_SPCNN\fold_1\test_predictions.csv"
DATA_ROOT             = r"D:\Data\Malaria\MalariaSingleCellDataset\cell_images"
GRADCAM_OVERLAY_DIR   = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\gradcam\overlay_40%"
GRADCAMPP_OVERLAY_DIR = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\gradcam++\overlay_40%"
PLOT_OUT_DIR          = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\plots"

TRUE_COL       = "true_label"
PRED_COL       = "predicted_label"
FNAME_COL      = "image_path"

OVERLAY_SUFFIX = "_true{true}_pred{pred}.png"
# ----------------------------------------------

os.makedirs(PLOT_OUT_DIR, exist_ok=True)

# label map
LABEL_MAP = {
    1: "Uninfected",
    0: "Parasitized"
}

# 1) load csv
df = pd.read_csv(CSV_PATH)

# 2) pick one random sample for each (true,pred) pair
pairs = [(1,1),(1,0),(0,0),(0,1)]
candidates = {}
for t,p in pairs:
    sub = df[(df[TRUE_COL]==t) & (df[PRED_COL]==p)]
    if sub.empty:
        raise ValueError(f"No rows where {TRUE_COL}=={t} and {PRED_COL}=={p}")
    candidates[(t,p)] = sub.sample(1).iloc[0]

# 3) build figure (3 columns: original | gradcam | gradcam++)
fig, axes = plt.subplots(
    len(pairs), 3,
    figsize=(9, 3*len(pairs)),
    squeeze=False
)

for i,(t,p) in enumerate(pairs):
    row = candidates[(t,p)]
    fname = row[FNAME_COL]
    base  = os.path.basename(fname)
    name,_ = os.path.splitext(base)

    # load original RGB
    orig_bgr = cv2.imread(os.path.join(DATA_ROOT, fname), cv2.IMREAD_COLOR)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # load gradcam overlay
    overlay_name = f"{name}{OVERLAY_SUFFIX.format(true=t,pred=p)}"
    ov_bgr = cv2.imread(os.path.join(GRADCAM_OVERLAY_DIR,   overlay_name), cv2.IMREAD_COLOR)
    if ov_bgr is None:
        raise FileNotFoundError(f"GradCAM overlay not found: {overlay_name}")
    ov_rgb = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)

    # load gradcam++ overlay
    ov_pp_bgr = cv2.imread(os.path.join(GRADCAMPP_OVERLAY_DIR, overlay_name), cv2.IMREAD_COLOR)
    if ov_pp_bgr is None:
        raise FileNotFoundError(f"GradCAM++ overlay not found: {overlay_name}")
    ov_pp_rgb = cv2.cvtColor(ov_pp_bgr, cv2.COLOR_BGR2RGB)

    # determine descriptive labels
    orig_label = LABEL_MAP[t]
    pred_label = LABEL_MAP[p]

    # plot original
    ax = axes[i,0]
    ax.imshow(orig_rgb)
    ax.set_title(f"Original {orig_label},\nPredicted {pred_label}")
    ax.axis("off")

    # plot grad-cam overlay
    ax = axes[i,1]
    ax.imshow(ov_rgb)
    ax.set_title("GradCAM Overlay")
    ax.axis("off")

    # plot grad-cam++ overlay
    ax = axes[i,2]
    ax.imshow(ov_pp_rgb)
    ax.set_title("GradCAM++ Overlay")
    ax.axis("off")

plt.tight_layout()
out_path = os.path.join(PLOT_OUT_DIR, "confusion_examples.png")
plt.savefig(out_path, dpi=150)
print(f"✅ Saved grid to {out_path}")
