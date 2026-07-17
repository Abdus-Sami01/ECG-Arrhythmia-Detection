import json

import numpy as np
import tensorflow as tf

from baselines import build_cnn, build_fc, build_lstm
from benchmark_edge import desktop_latency_ms
from config import MODELS_DIR, RESULTS_DIR
from dataset import load_dataset
from metrics import per_class_report
from model import build_gru
from quantize import convert_int8, representative_dataset, tflite_predict
from train import train_model

SEEDS = [0, 1, 2]

CONFIGS = [
    ("gru16", lambda h=None: build_gru(16), lambda: build_gru(16, unroll=True), "unroll-required"),
    ("lstm16", lambda h=None: build_lstm(16), lambda: build_lstm(16, unroll=True), "unroll-required"),
    ("cnn16", lambda h=None: build_cnn(16), None, "native-int8"),
    ("fc32", lambda h=None: build_fc(32), None, "native-int8"),
]


def best_of_seeds(name, build_fn):
    best = None
    for seed in SEEDS:
        path = str(MODELS_DIR / f"{name}_s{seed}.keras")
        model, val_f1 = train_model(None, path, seed=seed, verbose=0, build_fn=build_fn)
        if best is None or val_f1 > best[0]:
            best = (val_f1, path)
    return tf.keras.models.load_model(best[1])


def main():
    data = load_dataset()
    train, test = data["train"], data["test"]
    x_test, rr_test, y_test = test["x"], test["rr"], test["y"]
    rep_gen = representative_dataset(train["x"], train["rr"])

    rows = []
    for name, build_fn, unrolled_fn, deploy in CONFIGS:
        model = best_of_seeds(name, build_fn)
        params = int(model.count_params())
        float_pred = np.argmax(model.predict({"beat": x_test, "rr": rr_test}, verbose=0), axis=1)
        float_f1 = per_class_report(y_test, float_pred)["macro_f1_nsvf"]

        convertible = model
        if unrolled_fn is not None:
            convertible = unrolled_fn()
            convertible.set_weights(model.get_weights())
        int8_bytes = convert_int8(convertible, rep_gen)
        (MODELS_DIR / f"{name}_int8.tflite").write_bytes(int8_bytes)
        int8_f1 = per_class_report(y_test, tflite_predict(int8_bytes, x_test, rr_test))["macro_f1_nsvf"]

        rows.append({
            "model": name,
            "params": params,
            "int8_size_kb": round(len(int8_bytes) / 1024, 1),
            "macro_f1_float": round(float_f1, 4),
            "macro_f1_int8": round(int8_f1, 4),
            "desktop_latency_ms": round(desktop_latency_ms(int8_bytes), 3),
            "tflite_micro_path": deploy,
        })
        print(f"{name:8s} params {params:5d}  int8 {rows[-1]['int8_size_kb']:6.1f} KB  "
              f"macro-F1 float {float_f1:.4f} int8 {int8_f1:.4f}  [{deploy}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline_comparison.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {RESULTS_DIR/'baseline_comparison.json'}")


if __name__ == "__main__":
    main()
