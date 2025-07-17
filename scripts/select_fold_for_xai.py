import numpy as np
import os

def select_representative_fold(fold_accuracies):
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    differences = [abs(acc - mean_acc) for acc in fold_accuracies]
    best_index = int(np.argmin(differences))
    return best_index, mean_acc, std_acc, differences

if __name__ == "__main__":
    # === Modify this to your actual run folder ===
    run_folder = "D:/Results/XAI4Malaria/runs/2025-07-10_10-47-13_SPCNN"
    output_file = os.path.join(run_folder, "representative_fold_analysis.txt")

    # === Replace with your actual fold accuracies ===
    fold_accuracies = [0.9439, 0.9414, 0.9332, 0.9462, 0.9477]

    # === Compute best fold
    best_fold, mean_acc, std_acc, differences = select_representative_fold(fold_accuracies)

    # === Compose analysis
    lines = []
    lines.append("Fold-wise Accuracy Analysis for XAI Selection\n")
    lines.append("Fold Accuracies:\n")
    for i, acc in enumerate(fold_accuracies):
        lines.append(f"  Fold {i}: {acc:.4f} (Delta from mean: {differences[i]:.4f})\n")
    lines.append(f"\nMean Accuracy: {mean_acc:.4f}")
    lines.append(f"\nStandard Deviation: {std_acc:.4f}")
    lines.append(f"\n\nSelected Fold: {best_fold} (closest to mean accuracy)\n")
    lines.append("Rationale: This fold best represents the model's average behavior and avoids bias from best/worst performance.\n")

    # === Save to file
    os.makedirs(run_folder, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Representative fold: {best_fold}")
    print(f"Saved analysis to: {output_file}")
