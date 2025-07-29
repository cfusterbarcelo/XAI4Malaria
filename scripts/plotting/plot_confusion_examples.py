# scripts/plot_confusion_examples.py

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
CSV_PATH               = r"D:\Results\XAI4Malaria\runs\2025-07-10_10-47-13_SPCNN\fold_1\test_predictions.csv"
DATA_ROOT              = r"D:\Data\Malaria\MalariaSingleCellDataset\cell_images"
GRADCAM_OVERLAY_DIR    = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\gradcam\overlay_40%"
GRADCAMPP_OVERLAY_DIR  = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\gradcam++\overlay_40%"
SHAP_GRAD_OVERLAY_DIR  = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\shap-grad\overlay"
SHAP_DEEP_OVERLAY_DIR  = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\shap-deep\overlay"
PLOT_OUT_DIR           = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\plots"
# ----------------------------------------------

TRUE_COL      = "true_label"
PRED_COL      = "predicted_label"
FNAME_COL     = "image_path"
OVERLAY_SUFFIX = "_true{true}_pred{pred}.png"

os.makedirs(PLOT_OUT_DIR, exist_ok=True)

# human‐readable labels
LABEL_MAP = {1: "Uninfected", 0: "Parasitized"}

# 1) load csv
df = pd.read_csv(CSV_PATH)

# 2) pick one random sample for each (true,pred) pair
pairs = [(1,1), (1,0), (0,0), (0,1)]
candidates = {}
for t,p in pairs:
    sub = df[(df[TRUE_COL]==t) & (df[PRED_COL]==p)]
    if sub.empty:
        raise ValueError(f"No rows where {TRUE_COL}=={t} and {PRED_COL}=={p}")
    candidates[(t,p)] = sub.sample(1).iloc[0]

# 3) build figure (5 columns)
fig, axes = plt.subplots(
    len(pairs), 5,
    figsize=(15, 3*len(pairs)),  # wider for 5 columns
    squeeze=False
)

for i,(t,p) in enumerate(pairs):
    row = candidates[(t,p)]
    fname = row[FNAME_COL]
    base, _ = os.path.splitext(os.path.basename(fname))

    # load original RGB
    orig_bgr = cv2.imread(os.path.join(DATA_ROOT, fname), cv2.IMREAD_COLOR)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # overlay filename
    overlay_name = f"{base}{OVERLAY_SUFFIX.format(true=t,pred=p)}"

    # 1) Grad-CAM
    path1 = os.path.join(GRADCAM_OVERLAY_DIR, overlay_name)
    ov1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    if ov1 is None:
        raise FileNotFoundError(f"GradCAM overlay not found: {path1}")
    ov1_rgb = cv2.cvtColor(ov1, cv2.COLOR_BGR2RGB)

    # 2) Grad-CAM++
    path2 = os.path.join(GRADCAMPP_OVERLAY_DIR, overlay_name)
    ov2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    if ov2 is None:
        raise FileNotFoundError(f"GradCAM++ overlay not found: {path2}")
    ov2_rgb = cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB)

    # 3) SHAP-Gradient Explainer
    path3 = os.path.join(SHAP_GRAD_OVERLAY_DIR, overlay_name)
    ov3 = cv2.imread(path3, cv2.IMREAD_COLOR)
    if ov3 is None:
        raise FileNotFoundError(f"SHAP-Gradient overlay not found: {path3}")
    ov3_rgb = cv2.cvtColor(ov3, cv2.COLOR_BGR2RGB)

    # 4) SHAP-Deep Explainer
    path4 = os.path.join(SHAP_DEEP_OVERLAY_DIR, overlay_name)
    ov4 = cv2.imread(path4, cv2.IMREAD_COLOR)
    if ov4 is None:
        raise FileNotFoundError(f"SHAP-Deep overlay not found: {path4}")
    ov4_rgb = cv2.cvtColor(ov4, cv2.COLOR_BGR2RGB)

    # descriptive labels
    orig_label = LABEL_MAP[t]
    pred_label = LABEL_MAP[p]

    # Plotting each column
    titles = [
        f"Original {orig_label},\nPred {pred_label}",
        "Grad-CAM",
        "Grad-CAM++",
        "SHAP-Gradient Explainer",
        "SHAP-Deep Explainer",
    ]
    images = [orig_rgb, ov1_rgb, ov2_rgb, ov3_rgb, ov4_rgb]

    for j, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i,j]
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

plt.tight_layout()
out_path = os.path.join(PLOT_OUT_DIR, "confusion_examples_all5.png")
plt.savefig(out_path, dpi=150)
print(f"✅ Saved grid to {out_path}")
