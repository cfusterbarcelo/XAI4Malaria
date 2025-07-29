import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
CSV_PATH = r"D:\Results\XAI4Malaria\runs\2025-07-10_10-47-13_SPCNN\fold_1\test_predictions.csv"
DATA_ROOT = r"D:\Data\Malaria\MalariaSingleCellDataset\cell_images"

# Overlay directories for each XAI method
overlay_dirs = {
    'gradcam':       r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\gradcam\overlay_40%",
    'gradcam++':     r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\gradcam++\overlay_40%",
    'shap-gradient': r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\shap-grad\overlay",
    'shap-deep':     r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\shap-deep\overlay",
    'lime':          r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\lime\overlay"
}

PLOT_OUT_DIR = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\plots"
EVAL_DIR = os.path.join(PLOT_OUT_DIR, 'evalGoodness')
NUM_ROWS = 10       # Number of image pairs per figure
dpi = 300           # Figure resolution

# true/pred column names
TRUE_COL  = 'true_label'
PRED_COL  = 'predicted_label'
FNAME_COL = 'image_path'

# Human-readable labels
LABEL_MAP = {1: 'Uninfected', 0: 'Parasitized'}

os.makedirs(EVAL_DIR, exist_ok=True)

# Load predictions
df = pd.read_csv(CSV_PATH)

# Prepare pools
correct_uninf   = df[(df[TRUE_COL]==1) & (df[PRED_COL]==1)]
correct_para    = df[(df[TRUE_COL]==0) & (df[PRED_COL]==0)]
err_uninf2para  = df[(df[TRUE_COL]==1) & (df[PRED_COL]==0)]
err_para2uninf  = df[(df[TRUE_COL]==0) & (df[PRED_COL]==1)]

# Check availability
if len(correct_uninf) < 4 or len(correct_para) < 4 or len(err_uninf2para) < 1 or len(err_para2uninf) < 1:
    raise ValueError('Not enough samples to satisfy the 4/4/1/1 split')

# Sample indices once (same for all methods)
sampled = pd.concat([
    correct_uninf.sample(4, random_state=0),
    correct_para.sample(4, random_state=1),
    err_uninf2para.sample(1, random_state=2),
    err_para2uninf.sample(1, random_state=3),
]).reset_index(drop=True)

# Tags for each row
tags = (
    ['Correct: \nUninfected'] * 4 +
    ['Correct: \nParasitized'] * 4 +
    ['Error: \nUninfected predicted as Parasitized'] +
    ['Error: \nParasitized predicted as Uninfected']
)

# Loop over methods
for method, overlay_dir in overlay_dirs.items():
    fig, axes = plt.subplots(NUM_ROWS, 2, figsize=(6, NUM_ROWS * 2.5), squeeze=False)
    # Reduced title size and vertical position for better balance 
    fig.subplots_adjust(left=0.25, top=0.96)
    fig.suptitle(
        f'Explanation Goodness & Satisfaction — {method.upper()}',
        fontsize=14,
        y=0.98
    )

    records = []
    for i, row in sampled.iterrows():
        t = row[TRUE_COL]; p = row[PRED_COL]
        fname = row[FNAME_COL]
        base, _ = os.path.splitext(os.path.basename(fname))

        # Load original and explanation
        orig = cv2.imread(os.path.join(DATA_ROOT, fname))
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        overlay_name = f"{base}_true{t}_pred{p}.png"
        ov = cv2.imread(os.path.join(overlay_dir, overlay_name))
        if ov is None:
            raise FileNotFoundError(f"Missing overlay for {method}: {overlay_name}")
        ov_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)

        # Plot row i
        ax_orig = axes[i, 0]
        ax_exp  = axes[i, 1]
        ax_orig.imshow(orig_rgb); ax_orig.set_title('Original', fontsize=10); ax_orig.axis('off')
        ax_exp.imshow(ov_rgb);   ax_exp.set_title('Explanation', fontsize=10); ax_exp.axis('off')

        # Row label
        ax_orig.text(
            -0.2, 0.5, tags[i],
            transform=ax_orig.transAxes,
            fontsize=9, va='center', ha='right'
        )

        # Record mapping
        records.append({
            'method': method,
            'plot': f'{method}_evalGoodness.png',
            'original_image': fname,
            'explanation_image': overlay_name
        })

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(EVAL_DIR, f'{method}_evalGoodness.png')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # Save mapping CSV per method
    pd.DataFrame(records).to_csv(os.path.join(EVAL_DIR, f'{method}_mapping.csv'), index=False)

print(f'Done: generated evaluation figures in {EVAL_DIR}')
