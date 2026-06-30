"""
สคริปต์ประเมินโมเดล GRU บน Test Set — คำนวณ FAR/FRR และ Estimated EER
"""

import sys

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

from config import MODEL_PATH, PROCESSED_DATA_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    required_files = [
        PROCESSED_DATA_PATH / "X_test.npy",
        PROCESSED_DATA_PATH / "y_test.npy",
        MODEL_PATH,
    ]
    for path in required_files:
        if not path.is_file():
            print(f"ERROR: ไม่พบไฟล์ {path}")
            print("กรุณารัน split_and_save.py และ train.py ก่อน")
            sys.exit(1)

    X_test = np.load(PROCESSED_DATA_PATH / "X_test.npy")
    y_test = np.load(PROCESSED_DATA_PATH / "y_test.npy")
    model = load_model(MODEL_PATH)

    y_pred_prob = model.predict(X_test, verbose=0).ravel()

    n_impostor = int((y_test == 0).sum())
    n_owner = int((y_test == 1).sum())

    if n_impostor == 0 or n_owner == 0:
        print("ERROR: Test set must contain both Owner and Impostor samples. EER is invalid.")
        sys.exit(1)

    thresholds = np.arange(0.0, 1.01, 0.01)
    far_list, frr_list = [], []

    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        far = ((y_pred == 1) & (y_test == 0)).sum() / n_impostor
        frr = ((y_pred == 0) & (y_test == 1)).sum() / n_owner
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    eer_idx = int(np.argmin(np.abs(far_arr - frr_arr)))
    eer_threshold = float(thresholds[eer_idx])
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2)
    far_at_eer = float(far_arr[eer_idx])
    frr_at_eer = float(frr_arr[eer_idx])

    print(f"{'Threshold':>10}  {'FAR':>8}  {'FRR':>8}")
    print("-" * 32)
    for t, far, frr in zip(thresholds, far_list, frr_list):
        print(f"{t:10.2f}  {far:8.4f}  {frr:8.4f}")

    print()
    print(
        f"Estimated EER: {eer:.4f}  |  Threshold: {eer_threshold:.2f}  |  "
        f"FAR: {far_at_eer:.4f}  |  FRR: {frr_at_eer:.4f}"
    )

    y_pred_eer = (y_pred_prob >= eer_threshold).astype(int)
    print()
    print(classification_report(y_test, y_pred_eer, target_names=["Impostor", "Owner"]))


if __name__ == "__main__":
    main()
