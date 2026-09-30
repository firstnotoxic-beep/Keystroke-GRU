"""Normalization ablation for the baselines (SVM, Manhattan, LSTM): ON vs OFF, seed 42.

Uses the same seed-42 split / augmentation as run_experiment.py. LSTM gets a longer
budget (default 150 epochs, patience 15) so it is not cut off mid-improvement.
Does not overwrite results/final_results_summary.csv.

Usage:
    python run_baselines_norm_ablation.py
    python run_baselines_norm_ablation.py --lstm-epochs 150 --patience 15
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

import pandas as pd  # noqa: E402

from config import RAW_DATA_PATH  # noqa: E402
from experiment.common import (  # noqa: E402
    LABEL_COL,
    MODELS_DIR,
    RESULTS_DIR,
    banner,
    ensure_dirs,
    load_real_data,
    prepare_features,
    reshape_sequences,
    set_global_seed,
    split_real,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """CLI flags."""
    p = argparse.ArgumentParser(description="Baseline normalization ablation (ON vs OFF)")
    p.add_argument("--data", default=str(RAW_DATA_PATH))
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--lstm-epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--no-lstm", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run SVM / Manhattan / LSTM with normalization ON and OFF; save one comparison table."""
    args = parse_args()
    t0 = time.time()
    ensure_dirs()
    set_global_seed()

    from experiment.augment import augment_train
    from experiment.baselines import manhattan_distance, manhattan_template, svm_scores, train_svm
    from experiment.metrics import report
    from experiment.models import train_rnn

    df = load_real_data(args.data)
    train_real, val_real, test_real = split_real(df)
    train_aug = augment_train(train_real, args.target)
    y_tr = train_aug[LABEL_COL].to_numpy(dtype=int)
    y_va = val_real[LABEL_COL].to_numpy(dtype=int)
    y_te = test_real[LABEL_COL].to_numpy(dtype=int)
    owner_train = train_real[train_real[LABEL_COL] == 1]
    print(f"[data] train_aug={len(train_aug)} val={len(val_real)} test={len(test_real)} "
          f"(owner={int(y_te.sum())}, impostor={int((y_te == 0).sum())})")

    rows: list[dict] = []

    def add(model: str, norm: bool, n_train: int, m: dict, extra: str = "", best_epoch=None):
        rows.append({"model_name": model, "normalization": "ON" if norm else "OFF",
                     "n_train_sequences": n_train,
                     "EER(%)": round(m["eer"] * 100, 2), "FAR(%)": round(m["far"] * 100, 2),
                     "FRR(%)": round(m["frr"] * 100, 2), "AUC": round(m["auc"], 4),
                     "optimal_threshold": round(m["threshold"], 6),
                     "best_epoch": best_epoch, "notes": extra})

    for norm in (True, False):
        tag = "on" if norm else "off"
        banner(f"Normalization {tag.upper()}")

        try:
            clf, scaler, info = train_svm(prepare_features(train_aug, norm), y_tr,
                                          prepare_features(val_real, norm), y_va)
            pd.DataFrame(info["grid"]).to_csv(RESULTS_DIR / f"svm_grid_search_norm_{tag}.csv",
                                              index=False)
            m = report(f"SVM norm {tag}", y_te,
                       svm_scores(clf, scaler, prepare_features(test_real, norm)))
            add("SVM", norm, len(train_aug), m, f"C={info['C']}, gamma={info['gamma']}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR [SVM norm {tag}]: {type(exc).__name__}: {exc} - skipping")

        try:
            template = manhattan_template(prepare_features(owner_train, norm))
            dist = manhattan_distance(template, prepare_features(test_real, norm))
            m = report(f"Manhattan norm {tag}", y_te, -dist, " (negated distance)")
            m["threshold"] = -m["threshold"]
            add("Manhattan Distance", norm, len(owner_train), m, "accept if L1 distance <= threshold")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR [Manhattan norm {tag}]: {type(exc).__name__}: {exc} - skipping")

        if args.no_lstm:
            continue
        try:
            set_global_seed()

            def seq(frame: pd.DataFrame):
                return reshape_sequences(prepare_features(frame, norm))

            model, hist = train_rnn("lstm", seq(train_aug), y_tr, seq(val_real), y_va,
                                    MODELS_DIR / f"lstm_norm_{tag}_ep{args.lstm_epochs}.keras",
                                    args.lstm_epochs, patience=args.patience)
            m = report(f"LSTM norm {tag}", y_te, model.predict(seq(test_real), verbose=0).ravel())
            add("LSTM", norm, len(train_aug), m,
                f"{len(hist['loss'])}/{args.lstm_epochs} epochs run", hist["best_epoch"])
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR [LSTM norm {tag}]: {type(exc).__name__}: {exc} - skipping")

    banner("Baseline normalization ablation (seed 42, real-only test set)")
    out = pd.DataFrame(rows).sort_values(["model_name", "normalization"], ascending=[True, False])
    print(out.to_string(index=False))
    p = RESULTS_DIR / "baselines_norm_on_vs_off.csv"
    out.to_csv(p, index=False)
    print(f"\nSaved: {p}\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
