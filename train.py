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
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        y_pred = np.argmax(self.model.predict(self.x_val, verbose=0), axis=1)
        logs["val_macro_f1"] = macro_f1(self.y_val, y_pred, labels=NSVF_LABELS)


def class_weights(y):
    classes = np.arange(len(AAMI_CLASSES))
    present = np.unique(y)
    weights = compute_class_weight("balanced", classes=present, y=y)
    lookup = {int(c): float(w) for c, w in zip(present, weights)}
    return {int(c): lookup.get(int(c), 0.0) for c in classes}


def main():
    parser = argparse.ArgumentParser(description="Train a full-precision GRU on the patient-independent split.")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]

    model = build_gru(hidden_size=args.hidden_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model_path = MODELS_DIR / f"gru{args.hidden_size}.keras"
    callbacks = [
        MacroF1(x_val, y_val),
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_macro_f1", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights(y_train),
        callbacks=callbacks,
        verbose=2,
    )

    best = float(np.max(history.history["val_macro_f1"]))
    history_path = RESULTS_DIR / f"gru{args.hidden_size}_history.json"
    history_path.write_text(json.dumps({k: [float(v) for v in vs] for k, vs in history.history.items()}, indent=2))
    print(f"\nbest val macro-F1 (N,S,V,F): {best:.4f}")
    print(f"saved model to {model_path}")


if __name__ == "__main__":
    main()
