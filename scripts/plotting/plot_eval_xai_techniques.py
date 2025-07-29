import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
CSV_PATH = r"D:\Results\XAI4Malaria\runs\2025-07-10_10-47-13_SPCNN\fold_1\test_predictions.csv"
DATA_ROOT = r"D:\Data\Malaria\MalariaSingleCellDataset\cell_images"

# Overlay directories for each XAI method
GRADCAM_OVERLAY_DIR   = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\gradcam\overlay_40%"
GRADCAMPP_OVERLAY_DIR = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\gradcam++\overlay_40%"
SHAP_GRAD_OVERLAY_DIR = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\shap-grad\overlay"
SHAP_DEEP_OVERLAY_DIR = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\shap-deep\overlay"
LIME_OVERLAY_DIR      = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\lime\overlay"

PLOT_OUT_DIR = r"D:\Results\XAI4Malaria\explainability-runs\2025-07-10_10-47-13_SPCNN\fold_1\plots"
EVAL_DIR_NAME = "evalXAI"
NUM_PLOTS = 30          # <-- Set the number of figures to generate
ACCURACY = 0.94         # Desired classification accuracy (e.g. 0.94 = 94% correct)
DPI = 300               # Figure resolution for publication
RANDOM_SEED = 42
# ----------------------------------------------

# Column names in CSV
TRUE_COL  = "true_label"
PRED_COL  = "predicted_label"
FNAME_COL = "image_path"
OVERLAY_SUFFIX = "_{base}_true{true}_pred{pred}.png"  # will format manually

# Human-readable labels
LABEL_MAP = {1: "Uninfected", 0: "Parasitized"}

# Ensure reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create evaluation directory
eval_dir = os.path.join(PLOT_OUT_DIR, EVAL_DIR_NAME)
os.makedirs(eval_dir, exist_ok=True)

# Load predictions
df = pd.read_csv(CSV_PATH)

# Split correct vs. misclassified
df_correct = df[df[TRUE_COL] == df[PRED_COL]]
df_incorrect = df[df[TRUE_COL] != df[PRED_COL]]

# Determine number of correct / incorrect samples
num_correct   = int(round(NUM_PLOTS * ACCURACY))
num_incorrect = NUM_PLOTS - num_correct

# Sample without replacement
if num_correct > len(df_correct) or num_incorrect > len(df_incorrect):
    raise ValueError("Not enough samples to satisfy requested accuracy and N plots.")
sample_correct   = df_correct.sample(num_correct, random_state=RANDOM_SEED)
sample_incorrect = df_incorrect.sample(num_incorrect, random_state=RANDOM_SEED)

dsamp = pd.concat([sample_correct, sample_incorrect])
# Shuffle question order
dsamp = dsamp.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Prepare mapping records
records = []

# Directories and method labels
method_dirs = {
    'gradcam': GRADCAM_OVERLAY_DIR,
    'gradcam++': GRADCAMPP_OVERLAY_DIR,
    'shap-gradient': SHAP_GRAD_OVERLAY_DIR,
    'shap-deep': SHAP_DEEP_OVERLAY_DIR,
    'lime': LIME_OVERLAY_DIR
}

# Loop over samples
for idx, row in dsamp.iterrows():
    t = row[TRUE_COL]
    p = row[PRED_COL]
    fname = row[FNAME_COL]
    base, _ = os.path.splitext(os.path.basename(fname))

    # Load original
    orig_bgr = cv2.imread(os.path.join(DATA_ROOT, fname), cv2.IMREAD_COLOR)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Randomize mapping of methods to labels A-E
    items = list(method_dirs.items())
    random.shuffle(items)
    mapping = { letter: method for letter, (method, _) in zip('ABCDE', items) }

    # Build plot
    fig, axes = plt.subplots(1, 6, figsize=(18, 3), squeeze=True)
    fig.suptitle(
        f"Actual class: {LABEL_MAP[t]}   │   Model prediction: {LABEL_MAP[p]}",
        fontsize=14,
        y=1.02
    )

    # Panel 1: original
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')

    # Panels A-E: explanations
    for j, letter in enumerate('ABCDE', start=1):
        method = mapping[letter]
        overlay_name = f"{base}_true{t}_pred{p}.png"
        overlay_path = os.path.join(method_dirs[method], overlay_name)
        ov_bgr = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
        if ov_bgr is None:
            raise FileNotFoundError(f"Overlay not found for {method}: {overlay_path}")
        ov_rgb = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)

        axes[j].imshow(ov_rgb)
        axes[j].set_title(f"Explanation {letter}", fontsize=12)
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Save figure
    out_name = f"{base}_true{t}_pred{p}_evalXAI.png"
    out_path = os.path.join(eval_dir, out_name)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # Record mapping
    record = {
        'plot_filename': out_name,
        'original_image': fname
    }
    for letter in 'ABCDE':
        record[f'explanation_{letter}'] = mapping[letter]
    records.append(record)

# Save mapping CSV
map_df = pd.DataFrame(records)
map_csv = os.path.join(eval_dir, 'mapping.csv')
map_df.to_csv(map_csv, index=False)

print(f"Generated {len(records)} plots in: {eval_dir}")
print(f"Mapping CSV saved to: {map_csv}")
