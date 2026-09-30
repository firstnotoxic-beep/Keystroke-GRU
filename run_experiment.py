"""End-to-end reproducible experiment: split -> augment -> sanity check -> models -> summary.

Usage:
    python run_experiment.py                  # full run (GRU main, GRU ablation, SVM, Manhattan, LSTM)
    python run_experiment.py --no-lstm        # skip the optional LSTM
    python run_experiment.py --reshape existing   # use split_and_save.reshape_to_3d layout
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FEAT_COLS, RAW_DATA_PATH  # noqa: E402
from experiment.common import (  # noqa: E402
    FIGURES_DIR,
    LABEL_COL,
    MODELS_DIR,
    RESULTS_DIR,
    SANITY_DIR,
    SPLITS_DIR,
    SUBJECT_COL,
    banner,
    check_normalization_matches_preprocess,
    compare_with_existing_reshape,
    ensure_dirs,
    load_real_data,
    prepare_features,
    print_split_counts,
    reshape_sequences,
    set_global_seed,
    split_real,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """CLI flags for the experiment."""
    p = argparse.ArgumentParser(description="Keystroke-dynamics experiment pipeline")
    p.add_argument("--data", default=str(RAW_DATA_PATH), help="merged real dataset CSV")
    p.add_argument("--target", type=int, default=2000, help="total train sequences after augmentation")
    p.add_argument("--epochs", type=int, default=None, help="override train.py EPOCHS (25)")
    p.add_argument("--no-lstm", action="store_true", help="skip OPTIONAL LSTM model")
    p.add_argument("--reshape", choices=["spec", "existing"], default="spec",
                   help="spec: [H_i,DD_i,UD_i] pad at end; existing: split_and_save.reshape_to_3d")
    p.add_argument("--baseline-norm", choices=["on", "off"], default="on",
                   help="apply intra-sequence ratio normalization to SVM / Manhattan features")
    return p.parse_args()


def main() -> None:
    """Run every step of the experiment and print a recap of output files."""
    args = parse_args()
    t0 = time.time()
    outputs: list = []
    warnings: list[str] = []

    banner("SETUP")
    set_global_seed()
    ensure_dirs()

    from experiment.augment import augment_train
    from experiment.baselines import (
        manhattan_distance, manhattan_template, svm_scores, train_svm,
    )
    from experiment.metrics import plot_det, plot_history, plot_roc, report
    from experiment.sanity_check import run_sanity_check
    from split_and_save import reshape_to_3d
    from train import EPOCHS

    epochs = args.epochs or EPOCHS
    reshape = reshape_sequences if args.reshape == "spec" else reshape_to_3d
    print(f"[config] epochs={epochs}, target={args.target}, reshape={args.reshape}, "
          f"LSTM={'OFF' if args.no_lstm else 'ON (optional)'}, baseline_norm={args.baseline_norm}")

    # ------------------------------------------------------------------ STEP 0
    banner("STEP 0 - Load real data & split 70/15/15 BEFORE augmentation")
    df = load_real_data(args.data)
    X_all_raw = df[FEAT_COLS].to_numpy(dtype=np.float64)
    if compare_with_existing_reshape(X_all_raw):
        warnings.append("Reshape padding differs from split_and_save.reshape_to_3d")
    check_normalization_matches_preprocess(X_all_raw)

    train_real, val_real, test_real = split_real(df)
    for name, part in (("train_real", train_real), ("val_real", val_real), ("test_real", test_real)):
        p = SPLITS_DIR / f"{name}.csv"
        part.to_csv(p, index=False)
        outputs.append(p)
    print("[split] Per-subject counts:")
    print_split_counts({"train": train_real, "val": val_real, "test": test_real})

    # ------------------------------------------------------------------ STEP 1
    banner("STEP 1 - Augment train_real only (gaussian / timewarp / mixup)")
    train_aug = augment_train(train_real, args.target)
    p = SPLITS_DIR / "train_augmented.csv"
    train_aug.to_csv(p, index=False)
    outputs.append(p)
    print(train_aug.groupby([SUBJECT_COL, "aug_method"]).size().unstack(fill_value=0).to_string())

    # ------------------------------------------------------------------ STEP 2
    banner("STEP 2 - Sanity check (real vs augmented) before training")
    sanity = run_sanity_check(train_real, train_aug, SANITY_DIR)
    outputs.extend(sanity["paths"].values())
    if sanity["n_mean_warn"] or sanity["n_ks_warn"]:
        warnings.append(f"Sanity: {sanity['n_mean_warn']} mean-shift, "
                        f"{sanity['n_ks_warn']}/{sanity['n_tests']} KS p<0.01")

    # ------------------------------------------------------------------ STEP 3
    banner("STEP 3 - Train & evaluate models on the SAME real-only test set")
    y_tr = train_aug[LABEL_COL].to_numpy(dtype=int)
    y_va = val_real[LABEL_COL].to_numpy(dtype=int)
    y_te = test_real[LABEL_COL].to_numpy(dtype=int)
    print(f"[data] test_real: owner={int(y_te.sum())}, impostor={int((y_te == 0).sum())}")

    def seq(frame: pd.DataFrame, normalize: bool) -> np.ndarray:
        return reshape(prepare_features(frame, normalize))

    results: dict[str, dict] = {}
    rows: list[dict] = []
    histories: dict[str, dict] = {}

    rnn_runs = [
        ("GRU (with correction)", "gru", True, "gru_main_norm_on"),
        ("GRU (ablation, no correction)", "gru", False, "gru_ablation_norm_off"),
    ]
    if not args.no_lstm:
        rnn_runs.append(("LSTM (optional)", "lstm", True, "lstm_optional_norm_on"))

    from experiment.models import train_rnn

    for display, cell, normalize, tag in rnn_runs:
        print(f"\n--- {display}: normalization {'ON' if normalize else 'OFF'} ---")
        try:
            set_global_seed()
            model_path = MODELS_DIR / f"{tag}.keras"
            model, hist = train_rnn(cell, seq(train_aug, normalize), y_tr,
                                    seq(val_real, normalize), y_va, model_path, epochs)
            outputs.append(model_path)
            outputs.extend(plot_history(hist, tag, FIGURES_DIR))
            scores = model.predict(seq(test_real, normalize), verbose=0).ravel()
            m = report(display, y_te, scores)
            m["best_epoch"] = hist["best_epoch"]
            results[display] = m
            histories[display] = hist
            rows.append({"model_name": display, "uses_sign_flip_correction": normalize,
                         "n_train_sequences": len(train_aug), **_pct(m)})
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR [{display}]: {type(exc).__name__}: {exc} - skipping")
            warnings.append(f"{display} failed: {exc}")

    base_norm = args.baseline_norm == "on"
    print(f"\n--- SVM (RBF): normalization {'ON' if base_norm else 'OFF'} ---")
    try:
        X_tr_f = prepare_features(train_aug, base_norm)
        clf, scaler, info = train_svm(X_tr_f, y_tr, prepare_features(val_real, base_norm), y_va)
        pd.DataFrame(info["grid"]).to_csv(RESULTS_DIR / "svm_grid_search.csv", index=False)
        outputs.append(RESULTS_DIR / "svm_grid_search.csv")
        m = report("SVM", y_te, svm_scores(clf, scaler, prepare_features(test_real, base_norm)),
                   f" (decision_function; C={info['C']}, gamma={info['gamma']})")
        results["SVM"] = m
        rows.append({"model_name": "SVM", "uses_sign_flip_correction": base_norm,
                     "n_train_sequences": len(train_aug), **_pct(m)})
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR [SVM]: {type(exc).__name__}: {exc} - skipping")
        warnings.append(f"SVM failed: {exc}")

    print(f"\n--- Manhattan distance: normalization {'ON' if base_norm else 'OFF'} ---")
    try:
        owner_train = train_real[train_real[LABEL_COL] == 1]
        template = manhattan_template(prepare_features(owner_train, base_norm))
        dist = manhattan_distance(template, prepare_features(test_real, base_norm))
        m = report("Manhattan Distance", y_te, -dist, " (negated distance)")
        m["threshold"] = -m["threshold"]
        print(f"[Manhattan Distance] accept as owner if L1 distance <= {m['threshold']:.6g}")
        results["Manhattan Distance"] = m
        rows.append({"model_name": "Manhattan Distance", "uses_sign_flip_correction": base_norm,
                     "n_train_sequences": len(owner_train), **_pct(m)})
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR [Manhattan]: {type(exc).__name__}: {exc} - skipping")
        warnings.append(f"Manhattan failed: {exc}")

    if results:
        for fn, name in ((plot_roc, "roc_comparison.png"), (plot_det, "det_comparison.png")):
            p = FIGURES_DIR / name
            fn(results, p)
            outputs.append(p)

    # ------------------------------------------------------------------ ablation
    abl = [k for k in ("GRU (with correction)", "GRU (ablation, no correction)") if k in results]
    if abl:
        banner("Ablation - GRU normalization ON (3a) vs OFF (3b)")
        abl_df = pd.DataFrame([{
            "model_name": k, "EER(%)": results[k]["eer"] * 100, "FAR(%)": results[k]["far"] * 100,
            "FRR(%)": results[k]["frr"] * 100, "AUC": results[k]["auc"],
            "threshold": results[k]["threshold"], "best_epoch": results[k]["best_epoch"],
        } for k in abl]).round(4)
        print(abl_df.to_string(index=False))
        p = RESULTS_DIR / "ablation_gru_norm_on_vs_off.csv"
        abl_df.to_csv(p, index=False)
        outputs.append(p)

    # ------------------------------------------------------------------ STEP 4
    banner("STEP 4 - Final results summary (real-only test set)")
    order = ["GRU (with correction)", "GRU (ablation, no correction)", "SVM",
             "Manhattan Distance", "LSTM (optional)"]
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["_o"] = summary["model_name"].map(order.index)
        summary = summary.sort_values("_o").drop(columns="_o").reset_index(drop=True)
        print(summary.to_string(index=False))
    p = RESULTS_DIR / "final_results_summary.csv"
    summary.to_csv(p, index=False)
    outputs.append(p)

    banner("RECAP - output files")
    for f in outputs:
        print(f"  {f}")
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")
    print(f"\nDone in {time.time() - t0:.1f}s")


def _pct(m: dict) -> dict:
    """Format metrics for the summary table (percent, 2 dp)."""
    return {"EER(%)": round(m["eer"] * 100, 2), "FAR(%)": round(m["far"] * 100, 2),
            "FRR(%)": round(m["frr"] * 100, 2), "optimal_threshold": round(m["threshold"], 6)}


if __name__ == "__main__":
    main()
