from tensorflow import keras
from tensorflow.keras import layers

from config import AAMI_CLASSES, WINDOW_LENGTH


def build_gru(hidden_size=32, dense_units=16, dropout=0.3, unroll=False):
    return keras.Sequential(
        [
            keras.Input(shape=(WINDOW_LENGTH, 1)),
            layers.GRU(hidden_size, unroll=unroll),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(len(AAMI_CLASSES), activation="softmax"),
        ],
        name=f"gru{hidden_size}",
    )
