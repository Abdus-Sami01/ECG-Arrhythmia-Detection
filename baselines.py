from tensorflow import keras
from tensorflow.keras import layers

from config import AAMI_CLASSES, GRU_TIMESTEPS, RR_FEATURES, WINDOW_LENGTH

SEGMENT_LENGTH = WINDOW_LENGTH // GRU_TIMESTEPS
N_CLASSES = len(AAMI_CLASSES)


def _head(features, rr_input, dense_units, dropout):
    merged = layers.Concatenate()([features, rr_input])
    dense = layers.Dense(dense_units, activation="relu")(merged)
    dense = layers.Dropout(dropout)(dense)
    return layers.Dense(N_CLASSES, activation="softmax")(dense)


def build_cnn(filters=16, dense_units=16, dropout=0.3):
    beat_input = keras.Input(shape=(WINDOW_LENGTH, 1), name="beat")
    rr_input = keras.Input(shape=(RR_FEATURES,), name="rr")
    x = layers.Conv1D(filters, 7, strides=2, activation="relu")(beat_input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(filters, 5, strides=2, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model([beat_input, rr_input], _head(x, rr_input, dense_units, dropout), name=f"cnn{filters}")


def build_lstm(hidden_size=16, dense_units=16, dropout=0.3, unroll=False):
    beat_input = keras.Input(shape=(WINDOW_LENGTH, 1), name="beat")
    rr_input = keras.Input(shape=(RR_FEATURES,), name="rr")
    sequence = layers.Reshape((GRU_TIMESTEPS, SEGMENT_LENGTH))(beat_input)
    x = layers.LSTM(hidden_size, unroll=unroll)(sequence)
    return keras.Model([beat_input, rr_input], _head(x, rr_input, dense_units, dropout), name=f"lstm{hidden_size}")


def build_fc(units=32, dense_units=16, dropout=0.3):
    beat_input = keras.Input(shape=(WINDOW_LENGTH, 1), name="beat")
    rr_input = keras.Input(shape=(RR_FEATURES,), name="rr")
    x = layers.Flatten()(beat_input)
    x = layers.Dense(units, activation="relu")(x)
    return keras.Model([beat_input, rr_input], _head(x, rr_input, dense_units, dropout), name=f"fc{units}")
