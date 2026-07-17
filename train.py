import argparse
import json

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import AAMI_CLASSES, MODELS_DIR, RESULTS_DIR
from dataset import load_dataset
from metrics import macro_f1
from model import build_gru

NSVF_LABELS = [AAMI_CLASSES.index(c) for c in ("N", "S", "V", "F")]


class MacroF1(tf.keras.callbacks.Callback):
    def __init__(self, inputs, y_val):
        super().__init__()
        self.inputs = inputs
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        y_pred = np.argmax(self.model.predict(self.inputs, verbose=0), axis=1)
        logs["val_macro_f1"] = macro_f1(self.y_val, y_pred, labels=NSVF_LABELS)


def class_weights(y, cap=50.0):
    classes = np.arange(len(AAMI_CLASSES))
    present = np.unique(y)
    weights = compute_class_weight("balanced", classes=present, y=y)
    lookup = {int(c): min(float(w), cap) for c, w in zip(present, weights)}
    return {int(c): lookup.get(int(c), 0.0) for c in classes}


def main():
    parser = argparse.ArgumentParser(description="Train a full-precision GRU on the patient-independent split.")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    model_path = args.model_path or str(MODELS_DIR / f"gru{args.hidden_size}.keras")
    _, best = train_model(args.hidden_size, model_path, args.epochs, args.batch_size, args.patience, args.seed)
    print(f"\nbest val macro-F1 (N,S,V,F): {best:.4f}")
    print(f"saved model to {model_path}")


def train_model(hidden_size, model_path, epochs=60, batch_size=128, patience=12, seed=42, verbose=2, build_fn=None):
    tf.keras.utils.set_random_seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    train, val = data["train"], data["val"]
    train_inputs = {"beat": train["x"], "rr": train["rr"]}
    val_inputs = {"beat": val["x"], "rr": val["rr"]}
    y_train, y_val = train["y"], val["y"]

    model = (build_fn or build_gru)(hidden_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        MacroF1(val_inputs, y_val),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_macro_f1", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights(y_train),
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, float(np.max(history.history["val_macro_f1"]))


if __name__ == "__main__":
    main()
