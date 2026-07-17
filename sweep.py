import argparse
import json
import shutil

import numpy as np
import tensorflow as tf

from config import MODELS_DIR, RESULTS_DIR
from dataset import load_dataset
from metrics import per_class_report
from train import train_model

HIDDEN_SIZES = [32, 16, 8]
SEEDS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser(description="Multi-seed architecture sweep with validation-based selection.")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    test = load_dataset()["test"]
    test_inputs = {"beat": test["x"], "rr": test["rr"]}
    y_test = test["y"]

    summary = {}
    for hidden in HIDDEN_SIZES:
        runs = []
        for seed in SEEDS:
            path = str(MODELS_DIR / f"gru{hidden}_s{seed}.keras")
            model, val_f1 = train_model(hidden, path, epochs=args.epochs, seed=seed, verbose=0)
            pred = np.argmax(model.predict(test_inputs, verbose=0), axis=1)
            test_f1 = per_class_report(y_test, pred)["macro_f1_nsvf"]
            runs.append({"seed": seed, "val_macro_f1": round(val_f1, 4), "test_macro_f1": round(test_f1, 4)})
            print(f"GRU-{hidden} seed {seed}: val {val_f1:.4f}  test {test_f1:.4f}")

        best = max(runs, key=lambda r: r["val_macro_f1"])
        shutil.copyfile(MODELS_DIR / f"gru{hidden}_s{best['seed']}.keras", MODELS_DIR / f"gru{hidden}.keras")
        test_scores = [r["test_macro_f1"] for r in runs]
        summary[f"gru{hidden}"] = {
            "runs": runs,
            "test_macro_f1_mean": round(float(np.mean(test_scores)), 4),
            "test_macro_f1_std": round(float(np.std(test_scores)), 4),
            "selected_seed": best["seed"],
            "selected_test_macro_f1": best["test_macro_f1"],
        }
        print(f"GRU-{hidden}: test macro-F1 {summary[f'gru{hidden}']['test_macro_f1_mean']} "
              f"+/- {summary[f'gru{hidden}']['test_macro_f1_std']}  (deploy seed {best['seed']})\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "seed_sweep.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RESULTS_DIR/'seed_sweep.json'}")


if __name__ == "__main__":
    main()
