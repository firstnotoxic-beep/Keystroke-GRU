"""Multi-seed robustness check for the GRU ablation (normalization ON vs OFF).

For each seed: stratified 70/15/15 split -> augment train only -> train GRU with and
without intra_sequence_ratio_normalization on the same split -> evaluate on that seed's
real-only test set. SVM / Manhattan / LSTM are NOT re-run.

Usage:
    python run_ablation_multiseed.py
    python run_ablation_multiseed.py --seeds 42 7 123 --epochs 80 --patience 15
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
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

CONDITIONS = (("with_correction", True), ("without_correction", False))


def parse_args() -> argparse.Namespace:
    """CLI flags for the multi-seed ablation."""
    p = argparse.ArgumentParser(description="Multi-seed GRU ablation (normalization ON vs OFF)")
    p.add_argument("--data", default=str(RAW_DATA_PATH))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 2024, 99])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--target", type=int, default=2000)
    return p.parse_args()


def run_seed(df: pd.DataFrame, seed: int, args: argparse.Namespace) -> list[dict]:
    """Split, augment and train both GRU conditions for one seed; return per-condition rows."""
    from experiment.augment import augment_train
    from experiment.metrics import report
    from experiment.models import train_rnn

    set_global_seed(seed)
    train_real, val_real, test_real = split_real(df, seed=seed)
    train_aug = augment_train(train_real, args.target, seed=seed)
    y_tr = train_aug[LABEL_COL].to_numpy(dtype=int)
    y_va = val_real[LABEL_COL].to_numpy(dtype=int)
    y_te = test_real[LABEL_COL].to_numpy(dtype=int)
    print(f"[seed {seed}] train={len(train_aug)} val={len(val_real)} "
          f"test={len(test_real)} (owner={int(y_te.sum())}, impostor={int((y_te == 0).sum())})")

    out_dir = MODELS_DIR / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond, normalize in CONDITIONS:
        def seq(frame: pd.DataFrame) -> np.ndarray:
            return reshape_sequences(prepare_features(frame, normalize))

        model, hist = train_rnn("gru", seq(train_aug), y_tr, seq(val_real), y_va,
                                out_dir / f"gru_{cond}_seed{seed}.keras", args.epochs,
                                seed=seed, patience=args.patience)
        m = report(f"seed {seed} {cond}", y_te, model.predict(seq(test_real), verbose=0).ravel())
        rows.append({"seed": seed, "condition": cond,
                     "EER": round(m["eer"] * 100, 2), "FAR": round(m["far"] * 100, 2),
                     "FRR": round(m["frr"] * 100, 2), "threshold": round(m["threshold"], 6),
                     "best_epoch": hist["best_epoch"], "epochs_run": len(hist["loss"])})
    return rows


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD per condition plus paired EER difference (without - with) per seed."""
    wide = per_seed.pivot(index="seed", columns="condition", values="EER").dropna()
    diff = wide["without_correction"] - wide["with_correction"]
    n_fav = int((diff > 0).sum())
    print("\nPaired EER difference per seed (without - with; positive = correction helps):")
    for s, d in diff.items():
        print(f"  seed {s}: {d:+.2f} pp")
    print(f"Correction reduced EER in {n_fav}/{len(diff)} seeds; "
          f"mean diff = {diff.mean():+.2f} ± {diff.std(ddof=1):.2f} pp")
    if len(diff) >= 2:
        from scipy.stats import ttest_rel
        t = ttest_rel(wide["without_correction"], wide["with_correction"])
        print(f"Paired t-test (n={len(diff)}): t={t.statistic:.3f}, p={t.pvalue:.4f}")

    rows = []
    for cond, _ in CONDITIONS:
        g = per_seed[per_seed["condition"] == cond]
        rows.append({
            "condition": cond,
            "mean_EER": round(g["EER"].mean(), 2), "sd_EER": round(g["EER"].std(ddof=1), 2),
            "mean_FAR": round(g["FAR"].mean(), 2), "sd_FAR": round(g["FAR"].std(ddof=1), 2),
            "mean_FRR": round(g["FRR"].mean(), 2), "sd_FRR": round(g["FRR"].std(ddof=1), 2),
            "n_seeds_favoring_correction": n_fav,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Loop over seeds, save per-seed and aggregate tables."""
    args = parse_args()
    t0 = time.time()
    ensure_dirs()
    print(f"[config] seeds={args.seeds} epochs={args.epochs} patience={args.patience} "
          f"target={args.target}")
    df = load_real_data(args.data)

    rows: list[dict] = []
    for seed in args.seeds:
        banner(f"SEED {seed}")
        try:
            seed_rows = run_seed(df, seed, args)
            rows.extend(seed_rows)
            e = {r["condition"]: r for r in seed_rows}
            print(f">>> Seed {seed}: with_correction EER={e['with_correction']['EER']:.2f}% "
                  f"(best ep {e['with_correction']['best_epoch']}), "
                  f"without_correction EER={e['without_correction']['EER']:.2f}% "
                  f"(best ep {e['without_correction']['best_epoch']})")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR [seed {seed}]: {type(exc).__name__}: {exc} - skipping")

    if not rows:
        print("No seeds completed.")
        return

    banner("AGGREGATE")
    per_seed = pd.DataFrame(rows)
    p1 = RESULTS_DIR / "ablation_multiseed_results.csv"
    per_seed.to_csv(p1, index=False)
    print(per_seed.to_string(index=False))

    summary = summarize(per_seed)
    p2 = RESULTS_DIR / "ablation_multiseed_summary.csv"
    summary.to_csv(p2, index=False)
    print("\n" + summary.to_string(index=False))

    print(f"\nSaved:\n  {p1}\n  {p2}\n  {MODELS_DIR / 'multiseed'}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
