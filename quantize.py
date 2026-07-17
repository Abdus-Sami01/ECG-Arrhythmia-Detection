import argparse
import json

import numpy as np
import tensorflow as tf

from config import MODELS_DIR, RESULTS_DIR
from dataset import load_dataset
from evaluate import evaluate_predictions


def representative_dataset(x_train, rr_train, n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(x_train.shape[0], size=min(n, x_train.shape[0]), replace=False)

    def generator():
        for i in idx:
            yield [x_train[i : i + 1].astype(np.float32), rr_train[i : i + 1].astype(np.float32)]

    return generator


def convert_int8(model, rep_gen):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def convert_dynamic(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def _feed(detail, values):
    scale, zero = detail["quantization"]
    if detail["dtype"] == np.int8 and scale > 0:
        values = np.clip(np.round(values / scale + zero), -128, 127)
    return values.astype(detail["dtype"])


def tflite_predict(tflite_bytes, x, rr):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    beat_in = next(d for d in inputs if len(d["shape"]) == 3)
    rr_in = next(d for d in inputs if len(d["shape"]) == 2)
    out = interpreter.get_output_details()[0]
    preds = np.empty(x.shape[0], dtype=np.int64)
    for i in range(x.shape[0]):
        interpreter.set_tensor(beat_in["index"], _feed(beat_in, x[i : i + 1]))
        interpreter.set_tensor(rr_in["index"], _feed(rr_in, rr[i : i + 1]))
        interpreter.invoke()
        preds[i] = int(np.argmax(interpreter.get_tensor(out["index"])[0]))
    return preds


def main():
    parser = argparse.ArgumentParser(description="Quantize a trained model to TFLite int8 and measure the accuracy cost on DS2.")
    parser.add_argument("--model", default=str(MODELS_DIR / "gru32.keras"))
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or __import__("pathlib").Path(args.model).stem
    data = load_dataset()
    train, test = data["train"], data["test"]
    x_test, rr_test, y_test = test["x"], test["rr"], test["y"]
    model = tf.keras.models.load_model(args.model)

    result = {"model": name}
    rep_gen = representative_dataset(train["x"], train["rr"])

    try:
        int8_bytes = convert_int8(model, rep_gen)
        int8_path = MODELS_DIR / f"{name}_int8.tflite"
        int8_path.write_bytes(int8_bytes)
        report = evaluate_predictions(y_test, tflite_predict(int8_bytes, x_test, rr_test), f"{name}_int8", save=True)
        result["int8"] = {
            "path": str(int8_path),
            "size_bytes": len(int8_bytes),
            "macro_f1_nsvf": report["macro_f1_nsvf"],
            "macro_f1_5class": report["macro_f1_5class"],
        }
        print(f"\nint8 tflite: {len(int8_bytes)/1024:.1f} KB  macro-F1(NSVF) {report['macro_f1_nsvf']:.4f}")
    except Exception as exc:
        result["int8_error"] = f"{type(exc).__name__}: {exc}"
        print(f"\nfull-int8 conversion failed: {type(exc).__name__}: {exc}")

    dyn_bytes = convert_dynamic(model)
    dyn_path = MODELS_DIR / f"{name}_dynamic.tflite"
    dyn_path.write_bytes(dyn_bytes)
    dyn_report = evaluate_predictions(y_test, tflite_predict(dyn_bytes, x_test, rr_test), f"{name}_dynamic", save=False)
    result["dynamic_range"] = {
        "path": str(dyn_path),
        "size_bytes": len(dyn_bytes),
        "macro_f1_nsvf": dyn_report["macro_f1_nsvf"],
    }
    print(f"dynamic-range tflite: {len(dyn_bytes)/1024:.1f} KB  macro-F1(NSVF) {dyn_report['macro_f1_nsvf']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}_quantization.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
