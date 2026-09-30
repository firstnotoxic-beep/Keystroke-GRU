"""Sanity checks comparing real vs augmented distributions before any training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config import FEAT_COLS
from experiment.common import SUBJECT_COL

MEAN_SHIFT_LIMIT = 20.0  # percent
KS_P_LIMIT = 0.01
EPS = 1e-12


def moment_table(train_real: pd.DataFrame, train_aug: pd.DataFrame) -> pd.DataFrame:
    """Per subject, per feature: mean/std of real vs (real + synthetic) and % difference."""
    rows = []
    for subj in sorted(train_real[SUBJECT_COL].unique()):
        r = train_real.loc[train_real[SUBJECT_COL] == subj, FEAT_COLS]
        a = train_aug.loc[train_aug[SUBJECT_COL] == subj, FEAT_COLS]
        for f in FEAT_COLS:
            rm, rs, am, as_ = r[f].mean(), r[f].std(), a[f].mean(), a[f].std()
            rows.append({
                "subject": subj, "feature": f,
                "real_mean": rm, "aug_mean": am,
                "mean_diff_pct": (am - rm) / (abs(rm) + EPS) * 100,
                "real_std": rs, "aug_std": as_,
                "std_diff_pct": (as_ - rs) / (abs(rs) + EPS) * 100,
            })
    return pd.DataFrame(rows)


def ks_table(train_real: pd.DataFrame, train_aug: pd.DataFrame) -> pd.DataFrame:
    """Per subject, per feature: two-sample KS test, real vs synthetic-only samples."""
    rows = []
    synth = train_aug[train_aug["is_synthetic"]]
    for subj in sorted(train_real[SUBJECT_COL].unique()):
        r = train_real.loc[train_real[SUBJECT_COL] == subj, FEAT_COLS]
        s = synth.loc[synth[SUBJECT_COL] == subj, FEAT_COLS]
        for f in FEAT_COLS:
            if len(s) < 2:
                stat, p = np.nan, np.nan
            else:
                stat, p = ks_2samp(r[f], s[f])
            rows.append({"subject": subj, "feature": f, "ks_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def run_sanity_check(train_real: pd.DataFrame, train_aug: pd.DataFrame, out_dir: Path) -> dict:
    """Print % diff and KS tables, emit WARNINGs, save CSVs. Returns warning counts + paths."""
    moments = moment_table(train_real, train_aug)
    ks = ks_table(train_real, train_aug)

    pd.set_option("display.width", 200)
    print("\n[sanity] Mean % diff (real+synthetic vs real) - rows=feature, cols=subject")
    print(moments.pivot(index="feature", columns="subject", values="mean_diff_pct")
          .reindex(FEAT_COLS).round(2).to_string())
    print("\n[sanity] Std % diff (real+synthetic vs real)")
    print(moments.pivot(index="feature", columns="subject", values="std_diff_pct")
          .reindex(FEAT_COLS).round(2).to_string())
    print("\n[sanity] KS test p-values (real vs synthetic)")
    print(ks.pivot(index="feature", columns="subject", values="p_value")
          .reindex(FEAT_COLS).map(lambda v: f"{v:.3g}").to_string())

    mean_warn = moments[moments["mean_diff_pct"].abs() > MEAN_SHIFT_LIMIT]
    ks_warn = ks[ks["p_value"] < KS_P_LIMIT]
    for _, r in mean_warn.iterrows():
        print(f"WARNING [sanity]: mean shift {r.mean_diff_pct:+.1f}% > {MEAN_SHIFT_LIMIT}% "
              f"for {r.subject}/{r.feature}")
    shown = 15
    for _, r in ks_warn.head(shown).iterrows():
        print(f"WARNING [sanity]: KS p={r.p_value:.2e} < {KS_P_LIMIT} for {r.subject}/{r.feature}")
    if len(ks_warn) > shown:
        print(f"WARNING [sanity]: ... {len(ks_warn) - shown} more KS warnings (see CSV)")
    print(f"[sanity] Summary: {len(mean_warn)} mean-shift warnings, "
          f"{len(ks_warn)}/{len(ks)} KS warnings (p<{KS_P_LIMIT})")
    if len(ks_warn):
        print("[sanity] KS per subject:",
              ks_warn.groupby("subject").size().to_dict())

    paths = {
        "moments": out_dir / "sanity_mean_std_diff.csv",
        "ks": out_dir / "sanity_ks_test.csv",
    }
    moments.to_csv(paths["moments"], index=False)
    ks.to_csv(paths["ks"], index=False)
    return {"n_mean_warn": len(mean_warn), "n_ks_warn": len(ks_warn),
            "n_tests": len(ks), "paths": paths}
