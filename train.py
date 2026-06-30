"""
สคริปต์เทรนโมเดล GRU สำหรับ Keystroke Dynamics (Binary Classification: Owner vs Impostor)
"""

import sys

import numpy as np
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras.models import Sequential

from config import MODEL_PATH, PROCESSED_DATA_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EPOCHS = 25
BATCH_SIZE = 128
GRU_UNITS = 64

TRAIN_FILES = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]


def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val arrays after verifying all required files exist."""
    missing = [name for name in TRAIN_FILES if not (PROCESSED_DATA_PATH / name).is_file()]
    if missing:
        print("ERROR: Missing required training data files:")
        for name in missing:
            print(f"  - {PROCESSED_DATA_PATH / name}")
        print("Please run split_and_save.py first.")
        sys.exit(1)

    X_train = np.load(PROCESSED_DATA_PATH / "X_train.npy")
    y_train = np.load(PROCESSED_DATA_PATH / "y_train.npy")
    X_val = np.load(PROCESSED_DATA_PATH / "X_val.npy")
    y_val = np.load(PROCESSED_DATA_PATH / "y_val.npy")
    return X_train, y_train, X_val, y_val


def main() -> None:
    # -------------------------------------------------------------------------
    # ขั้นที่ 1: โหลดข้อมูล Train / Val จาก split_and_save.py
    # -------------------------------------------------------------------------
    X_train, y_train, X_val, y_val = load_training_data()
    print(f"[1] โหลดข้อมูลแล้ว: X_train.shape={X_train.shape}, X_val.shape={X_val.shape}")
    print(
        f"    Train — Owner (1): {int(y_train.sum())}, "
        f"Impostor (0): {int(len(y_train) - y_train.sum())}"
    )
    print(
        f"    Val   — Owner (1): {int(y_val.sum())}, "
        f"Impostor (0): {int(len(y_val) - y_val.sum())}"
    )

    # -------------------------------------------------------------------------
    # ขั้นที่ 2: คำนวณ class_weight (แก้ Data Imbalance) — pure numpy
    # -------------------------------------------------------------------------
    total_samples = len(y_train)
    counts = np.bincount(y_train.astype(int), minlength=2)
    class_weight_dict = {
        0: total_samples / (2 * counts[0]),
        1: total_samples / (2 * counts[1]),
    }
    print(
        f"Class weights applied: "
        f"{{0: {class_weight_dict[0]:.4f}}}, {{1: {class_weight_dict[1]:.4f}}}"
    )

    # -------------------------------------------------------------------------
    # ขั้นที่ 3: สร้างโมเดล GRU + Dense (sigmoid)
    # -------------------------------------------------------------------------
    model = Sequential(
        [
            Input(shape=(11, 3)),
            GRU(GRU_UNITS),
            Dense(1, activation="sigmoid"),
        ],
        name="keystroke_gru",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    print("[3] สร้างและ compile โมเดล GRU แล้ว")
    model.summary()

    # -------------------------------------------------------------------------
    # ขั้นที่ 4: เทรนโมเดล
    # -------------------------------------------------------------------------
    fit_kwargs: dict = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "class_weight": class_weight_dict,
        "verbose": 1,
    }
    if len(X_val) > 0:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        print("WARNING: Validation set is empty — training without validation_data.")

    history = model.fit(X_train, y_train, **fit_kwargs)
    final_train_loss = history.history["loss"][-1]
    if len(X_val) > 0 and "val_loss" in history.history:
        final_val_loss = history.history["val_loss"][-1]
        print(
            f"[4] เทรนเสร็จ {EPOCHS} epochs — "
            f"loss(train)={final_train_loss:.4f}, loss(val)={final_val_loss:.4f}"
        )
    else:
        print(
            f"[4] เทรนเสร็จ {EPOCHS} epochs — "
            f"loss(train)={final_train_loss:.4f} (no validation loss)"
        )

    # -------------------------------------------------------------------------
    # ขั้นที่ 5: บันทึกโมเดล
    # -------------------------------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[5] บันทึกโมเดลแล้ว → {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
