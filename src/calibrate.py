"""Per-class confidence threshold calibration using real data."""

import os
import json
import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.inference import get_eval_transform, predict_image


def calibrate(model, config):
    """Find optimal per-class confidence thresholds on real data."""
    model.eval()
    classes = config["data"]["classes"]
    output_dir = os.path.dirname(config["output"]["checkpoint_dir"])

    probs, targets = collect_predictions(model, config)
    thresholds = find_best_thresholds(probs, targets, classes)

    preds = probs.argmax(axis=1)
    print_summary(thresholds, targets, preds, classes)
    save_results(thresholds, targets, preds, classes, output_dir)


def collect_predictions(model, config):
    """Run model on all real data images, return softmax probs and labels."""
    classes = config["data"]["classes"]
    transform = get_eval_transform(config["data"]["image_size"])
    all_probs, all_targets = [], []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join("data/real_data", class_name)
        for fname in sorted(os.listdir(class_dir)):
            if not fname.endswith(".png") or "(" in fname:
                continue
            probs = predict_image(model, os.path.join(class_dir, fname), transform)
            all_probs.append(probs)
            all_targets.append(class_idx)

    return np.array(all_probs), np.array(all_targets)


def find_best_thresholds(probs, targets, classes):
    """Use sklearn's precision_recall_curve to find the threshold maximizing F1 per class."""
    thresholds = {}
    for class_idx, class_name in enumerate(classes):
        binary_targets = (targets == class_idx).astype(int)
        class_probs = probs[:, class_idx]

        precision, recall, thresh = precision_recall_curve(binary_targets, class_probs)
        # precision_recall_curve returns arrays where len(thresh) == len(precision) - 1
        precision = precision[:-1]
        recall = recall[:-1]

        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
        best_idx = f1.argmax()
        thresholds[class_name] = round(float(thresh[best_idx]), 3)

    return thresholds


def print_summary(thresholds, targets, preds, classes):
    """Print calibration results to console."""
    print("\n" + "=" * 50)
    print("THRESHOLD CALIBRATION (real data)")
    print("=" * 50)
    for cls, t in thresholds.items():
        print(f"  {cls}: {t}")
    print(f"\n{classification_report(targets, preds, target_names=classes)}")
    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))


def save_results(thresholds, targets, preds, classes, output_dir):
    """Save thresholds JSON, text report, and confusion matrix plot."""
    os.makedirs(output_dir, exist_ok=True)

    # Thresholds JSON
    path = os.path.join(output_dir, "thresholds.json")
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nThresholds saved to {path}")

    # Text report
    report = classification_report(targets, preds, target_names=classes)
    cm = confusion_matrix(targets, preds)
    path = os.path.join(output_dir, "real_test_results.txt")
    with open(path, "w") as f:
        f.write("REAL DATA EVALUATION\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-class confidence thresholds:\n")
        for cls, t in thresholds.items():
            f.write(f"  {cls}: {t}\n")
        f.write(f"\n{report}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"Report saved to {path}")

    # Confusion matrix plot
    save_confusion_matrix_plot(cm, classes, output_dir)


def save_confusion_matrix_plot(cm, classes, output_dir):
    """Save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix (Real Data)")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes)
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes)
    mid = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > mid else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    path = os.path.join(output_dir, "real_confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {path}")
