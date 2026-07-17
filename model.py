from tensorflow import keras
from tensorflow.keras import layers

from config import AAMI_CLASSES, RR_FEATURES, WINDOW_LENGTH


def build_gru(hidden_size=32, dense_units=16, dropout=0.3, unroll=False):
    beat_input = keras.Input(shape=(WINDOW_LENGTH, 1), name="beat")
    rr_input = keras.Input(shape=(RR_FEATURES,), name="rr")

    morphology = layers.GRU(hidden_size, unroll=unroll)(beat_input)
    merged = layers.Concatenate()([morphology, rr_input])
    dense = layers.Dense(dense_units, activation="relu")(merged)
    dense = layers.Dropout(dropout)(dense)
    output = layers.Dense(len(AAMI_CLASSES), activation="softmax")(dense)

    return keras.Model(inputs=[beat_input, rr_input], outputs=output, name=f"gru{hidden_size}")
