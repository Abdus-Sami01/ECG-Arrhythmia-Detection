import json

import numpy as np
import tensorflow as tf

from baselines import build_cnn
from benchmark_edge import desktop_latency_ms
from config import MODELS_DIR, RESULTS_DIR
from dataset import load_dataset
from metrics import per_class_report
from quantize import convert_int8, representative_dataset, tflite_predict
from train import train_model

FILTERS = [8, 16, 32]
SEEDS = [0, 1, 2]


def main():
    data = load_dataset()
    train, test = data["train"], data["test"]
    x_test, rr_test, y_test = test["x"], test["rr"], test["y"]
    rep_gen = representative_dataset(train["x"], train["rr"])

    rows = []
    for filters in FILTERS:
        runs = []
        for seed in SEEDS:
            path = str(MODELS_DIR / f"cnn{filters}_s{seed}.keras")
            model, val_f1 = train_model(None, path, seed=seed, verbose=0, build_fn=lambda h, f=filters: build_cnn(f))
            pred = np.argmax(model.predict({"beat": x_test, "rr": rr_test}, verbose=0), axis=1)
            runs.append({"seed": seed, "val": round(val_f1, 4), "test": round(per_class_report(y_test, pred)["macro_f1_nsvf"], 4), "path": path})

        best = max(runs, key=lambda r: r["val"])
        model = tf.keras.models.load_model(best["path"])
        params = int(model.count_params())
        int8_bytes = convert_int8(model, rep_gen)
        (MODELS_DIR / f"cnn{filters}.tflite").write_bytes(int8_bytes)
        int8_report = per_class_report(y_test, tflite_predict(int8_bytes, x_test, rr_test))
        test_scores = [r["test"] for r in runs]

        rows.append({
            "model": f"CNN-{filters}",
            "params": params,
            "int8_size_kb": round(len(int8_bytes) / 1024, 1),
            "macro_f1_int8": round(int8_report["macro_f1_nsvf"], 4),
            "test_macro_f1_mean": round(float(np.mean(test_scores)), 4),
            "test_macro_f1_std": round(float(np.std(test_scores)), 4),
            "s_f1": int8_report["per_class"]["S"]["f1"],
            "v_f1": int8_report["per_class"]["V"]["f1"],
            "desktop_latency_ms": round(desktop_latency_ms(int8_bytes), 3),
        })
        print(f"CNN-{filters}: params {params}  int8 {rows[-1]['int8_size_kb']} KB  "
              f"macro-F1 {int8_report['macro_f1_nsvf']:.4f} (mean {rows[-1]['test_macro_f1_mean']}+/-{rows[-1]['test_macro_f1_std']})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "cnn_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {RESULTS_DIR/'cnn_sweep.json'}")


if __name__ == "__main__":
    main()
