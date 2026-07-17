import argparse
import json
import time

import numpy as np
import tensorflow as tf

from config import MODELS_DIR, RESULTS_DIR

CORTEX_M4_CLOCK_HZ = {"64MHz": 64e6, "80MHz": 80e6}
MACS_PER_CYCLE_INT8 = 1.0


def gru_macs(input_dim, hidden, timesteps):
    return timesteps * 3 * hidden * (input_dim + hidden)


def dense_macs(input_dim, units):
    return input_dim * units


def analytic_macs(model):
    total = 0
    for layer in model.layers:
        cls = layer.__class__.__name__
        if cls == "GRU":
            total += gru_macs(layer.input.shape[-1], layer.units, layer.input.shape[1])
        elif cls == "Dense":
            total += dense_macs(layer.input.shape[-1], layer.units)
    return total


def tensor_arena_bytes(tflite_bytes):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    try:
        return int(interpreter._interpreter.ArenaUsedBytes())
    except Exception:
        return None


def desktop_latency_ms(tflite_bytes, runs=200):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    sample = np.zeros(inp["shape"], dtype=inp["dtype"])
    for _ in range(20):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
    start = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
        interpreter.get_tensor(out["index"])
    return (time.perf_counter() - start) / runs * 1000


def main():
    parser = argparse.ArgumentParser(description="Report edge footprint and simulated Cortex-M4 latency for a TFLite model.")
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--keras", required=True, help="Keras model for analytic MAC counting.")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or __import__("pathlib").Path(args.tflite).stem
    tflite_bytes = open(args.tflite, "rb").read()
    model = tf.keras.models.load_model(args.keras)

    macs = analytic_macs(model)
    est_latency = {
        clock: round(macs / MACS_PER_CYCLE_INT8 / hz * 1000, 2) for clock, hz in CORTEX_M4_CLOCK_HZ.items()
    }

    result = {
        "model": name,
        "flash_size_kb": round(len(tflite_bytes) / 1024, 2),
        "tensor_arena_bytes": tensor_arena_bytes(tflite_bytes),
        "macs_per_inference": int(macs),
        "desktop_latency_ms": round(desktop_latency_ms(tflite_bytes), 3),
        "estimated_cortex_m4_latency_ms": est_latency,
        "estimate_assumptions": f"{MACS_PER_CYCLE_INT8} MAC/cycle (CMSIS-NN int8), simulated not measured on hardware",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}_edge.json").write_text(json.dumps(result, indent=2))

    print(f"{name}: flash {result['flash_size_kb']} KB  arena {result['tensor_arena_bytes']} B  "
          f"{result['macs_per_inference']} MAC/inf")
    print(f"  desktop latency {result['desktop_latency_ms']} ms (x86, reference only)")
    print(f"  estimated Cortex-M4 latency {est_latency} ms — {result['estimate_assumptions']}")


if __name__ == "__main__":
    main()
