"""Recurrent models (GRU main / ablation, optional LSTM) using train.py hyper-parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense, Input
from tensorflow.keras.models import Sequential

from config import NUM_KEYS
from train import BATCH_SIZE, EPOCHS, GRU_UNITS

PATIENCE = 10


def build_rnn(cell: str = "gru", units: int = GRU_UNITS) -> tf.keras.Model:
    """Input(11,3) -> GRU/LSTM(units) -> Dense(1, sigmoid), BCE + Adam."""
    layer = GRU(units) if cell == "gru" else LSTM(units)
    model = Sequential(
        [Input(shape=(NUM_KEYS, 3)), layer, Dense(1, activation="sigmoid")],
        name=f"keystroke_{cell}",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights, same formula as train.py."""
    counts = np.bincount(y.astype(int), minlength=2)
    return {0: len(y) / (2 * counts[0]), 1: len(y) / (2 * counts[1])}


def train_rnn(
    cell: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: Path,
    epochs: int = EPOCHS,
    seed: int = 42,
    patience: int = PATIENCE,
) -> tuple[tf.keras.Model, dict]:
    """Train with EarlyStopping(val_loss, patience, restore_best_weights); save model."""
    tf.keras.utils.set_random_seed(seed)
    model = build_rnn(cell)
    cw = class_weights(y_train)
    print(f"[{cell}] X_train={X_train.shape}  X_val={X_val.shape}  class_weight={cw}")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                 restore_best_weights=True, verbose=1)],
        verbose=2,
    )
    h = history.history
    best = int(np.argmin(h["val_loss"])) + 1
    print(f"[{cell}] trained {len(h['loss'])} epochs; best val_loss={min(h['val_loss']):.4f} "
          f"at epoch {best}")
    model.save(model_path)
    print(f"[{cell}] saved model -> {model_path}")
    h["best_epoch"] = best
    return model, h
