"""Validation-only ablation: does one feature earn its place?

Refits ridge and hgb on *identical* rows with and without the named
feature(s), so the delta is the feature's alone. Comparing metrics.json
before and after a feature commit confounds the feature with any change in
which rows survive the lag warm-up (adding cnt_roll_28 also dropped 42
training rows). The test split is discarded unscored: this module is for
iterating, and iterating on test is tuning on test.

Run: python -m src.ablation <feature> [<feature> ...]
"""
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import PROCESSED, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split

ABLATED_MODELS = ("ridge", "hgb")
NOISE_R2 = 0.01  # the validation set is 182 days; smaller deltas are noise


def ablate(drop: list[str], df: pd.DataFrame | None = None) -> dict:
    df = pd.read_parquet(PROCESSED) if df is None else df
    cols = feature_columns(df)
    unknown = sorted(set(drop) - set(cols))
    if unknown:
        raise SystemExit(f"not features: {unknown}; features are: {', '.join(cols)}")
    train, valid, _ = temporal_split(df)
    variants = {"with": cols, "without": [c for c in cols if c not in drop]}
    out = {}
    for name in ABLATED_MODELS:
        for label, use in variants.items():
            pred = clone(MODELS[name]).fit(train[use], train[TARGET]).predict(valid[use])
            out[(name, label)] = (
                round(float(r2_score(valid[TARGET], pred)), 4),
                round(float(mean_absolute_error(valid[TARGET], pred)), 1),
            )
    return {"n_train": len(train), "n_valid": len(valid), "scores": out}


if __name__ == "__main__":
    drop = sys.argv[1:]
    if not drop:
        raise SystemExit("usage: python -m src.ablation <feature> [<feature> ...]")
    res = ablate(drop)
    s = res["scores"]
    print(f"valid only, train={res['n_train']} rows, valid={res['n_valid']} rows, same rows both ways")
    print(f"{'model':<6} {'R2 with':>8} {'R2 w/o':>8} {'dR2':>8} {'MAE with':>9} {'MAE w/o':>8}")
    deltas = {}
    for name in ABLATED_MODELS:
        (r_w, m_w), (r_wo, m_wo) = s[(name, "with")], s[(name, "without")]
        deltas[name] = round(r_w - r_wo, 4)
        print(f"{name:<6} {r_w:>8.4f} {r_wo:>8.4f} {deltas[name]:>+8.4f} {m_w:>9.1f} {m_wo:>8.1f}")
    verdict = "keep" if deltas["ridge"] >= NOISE_R2 else "within noise"
    print(f"OK ablation: {'+'.join(drop)} ridge dR2 {deltas['ridge']:+.4f} ({verdict}), "
          f"hgb dR2 {deltas['hgb']:+.4f}")
