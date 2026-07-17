import argparse
import json

import numpy as np
import tensorflow as tf

from config import AAMI_CLASSES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR
from dataset import load_dataset
from metrics import per_class_report


def plot_confusion_matrix(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    normalized = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(AAMI_CLASSES)), AAMI_CLASSES)
    ax.set_yticks(range(len(AAMI_CLASSES)), AAMI_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(AAMI_CLASSES)):
        for j in range(len(AAMI_CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if normalized[i, j] < 0.5 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_predictions(y_true, y_pred, name, save=True):
    report = per_class_report(y_true, y_pred)
    print(f"\n=== {name} on DS2 (held-out patients) ===")
    print(f"accuracy {report['accuracy']}  macro-F1(N,S,V,F) {report['macro_f1_nsvf']:.4f}  macro-F1(5) {report['macro_f1_5class']:.4f}")
    print(f"{'class':<6}{'support':>9}{'sens':>9}{'spec':>9}{'ppv':>9}{'f1':>9}")
    for cls, m in report["per_class"].items():
        print(f"{cls:<6}{m['support']:>9}{m['sensitivity']:>9}{m['specificity']:>9}{m['ppv']:>9}{m['f1']:>9}")

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{name}_ds2_report.json").write_text(json.dumps(report, indent=2))
        plot_confusion_matrix(report["confusion_matrix"], FIGURES_DIR / f"{name}_confusion.png", f"{name} — DS2 confusion")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the held-out DS2 patients.")
    parser.add_argument("--model", default=str(MODELS_DIR / "gru32.keras"))
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or __import__("pathlib").Path(args.model).stem
    test = load_dataset()["test"]
    model = tf.keras.models.load_model(args.model)
    y_pred = np.argmax(model.predict({"beat": test["x"], "rr": test["rr"]}, verbose=0), axis=1)
    evaluate_predictions(test["y"], y_pred, name)


if __name__ == "__main__":
    main()
