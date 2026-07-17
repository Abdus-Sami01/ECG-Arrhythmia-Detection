import csv
import json

import numpy as np
import tensorflow as tf

from benchmark_edge import (
    CORTEX_M4_CLOCK_HZ,
    MACS_PER_CYCLE_INT8,
    analytic_macs,
    desktop_latency_ms,
    tensor_arena_bytes,
)
from config import MODELS_DIR, RESULTS_DIR
from dataset import load_dataset
from metrics import per_class_report
from quantize import convert_int8, representative_dataset, tflite_predict, unrolled_twin

HIDDEN_SIZES = [32, 16, 8]


def main():
    data = load_dataset()
    train, test = data["train"], data["test"]
    x_test, rr_test, y_test = test["x"], test["rr"], test["y"]
    rep_gen = representative_dataset(train["x"], train["rr"])

    rows = []
    for hidden in HIDDEN_SIZES:
        keras_model = tf.keras.models.load_model(MODELS_DIR / f"gru{hidden}.keras")
        params = int(keras_model.count_params())

        float_pred = np.argmax(keras_model.predict({"beat": x_test, "rr": rr_test}, verbose=0), axis=1)
        float_f1 = per_class_report(y_test, float_pred)["macro_f1_nsvf"]

        twin = unrolled_twin(keras_model)
        int8_bytes = convert_int8(twin, rep_gen)
        (MODELS_DIR / f"gru{hidden}_int8.tflite").write_bytes(int8_bytes)
        int8_pred = tflite_predict(int8_bytes, x_test, rr_test)
        int8_report = per_class_report(y_test, int8_pred)

        macs = analytic_macs(twin)
        est_latency = {c: round(macs / MACS_PER_CYCLE_INT8 / hz * 1000, 2) for c, hz in CORTEX_M4_CLOCK_HZ.items()}

        rows.append({
            "model": f"GRU-{hidden}",
            "params": params,
            "float32_size_kb": round(params * 4 / 1024, 1),
            "int8_size_kb": round(len(int8_bytes) / 1024, 1),
            "tensor_arena_kb": round((tensor_arena_bytes(int8_bytes) or 0) / 1024, 1),
            "macro_f1_float": round(float_f1, 4),
            "macro_f1_int8": round(int8_report["macro_f1_nsvf"], 4),
            "int8_per_class_f1": {c: m["f1"] for c, m in int8_report["per_class"].items()},
            "macs_per_inference": int(macs),
            "desktop_latency_ms": round(desktop_latency_ms(int8_bytes), 3),
            "est_cortex_m4_latency_ms": est_latency,
        })
        print(f"GRU-{hidden}: params {params}  int8 {rows[-1]['int8_size_kb']} KB  "
              f"macro-F1 float {float_f1:.4f} -> int8 {int8_report['macro_f1_nsvf']:.4f}  "
              f"est M4 latency {est_latency['80MHz']}-{est_latency['64MHz']} ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "headline_table.json").write_text(json.dumps(rows, indent=2))
    with open(RESULTS_DIR / "headline_table.csv", "w", newline="") as handle:
        fields = ["model", "params", "float32_size_kb", "int8_size_kb", "tensor_arena_kb",
                  "macro_f1_float", "macro_f1_int8", "macs_per_inference", "desktop_latency_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {RESULTS_DIR/'headline_table.csv'}")


if __name__ == "__main__":
    main()
