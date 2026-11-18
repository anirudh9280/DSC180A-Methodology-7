"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EX4_METRICS, METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def fit_and_score() -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)

    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }
    for name, model in MODELS.items():
        model.fit(train[cols], train[TARGET])
        results[name] = {
            split: {
                "r2": round(float(r2_score(part[TARGET], model.predict(part[cols]))), 4),
                "mae": round(float(mean_absolute_error(part[TARGET], model.predict(part[cols]))), 1),
            }
            for split, part in [("valid", valid), ("test", test)]
        }
    return results


# --- Exercise 4: target transformations, behind a flag ---------------------
# `make train` never uses these; reports/metrics.json stays untransformed.
# Each entry is (forward, inverse), both allowed to read X so a transform can
# be relative to a lagged level. Every column they read is a past-only
# feature, so the inverse is computable at prediction time.
TARGET_TRANSFORMS = {
    "log1p": (lambda y, X: np.log1p(y), lambda z, X: np.expm1(z)),
    # Ratio to the trailing 28-day level: the tree learns a scale-free
    # "today vs. the last four weeks" and the level carries the growth.
    "ratio_roll28": (lambda y, X: y / X["cnt_roll_28"], lambda z, X: z * X["cnt_roll_28"]),
    # Same idea, additive and against the last week.
    "diff_roll7": (lambda y, X: y - X["cnt_roll_7"], lambda z, X: z + X["cnt_roll_7"]),
}


class TransformedTarget(BaseEstimator, RegressorMixin):
    """Fit `model` on forward(y, X); predict inverse(model(X), X)."""

    def __init__(self, model, transform: str):
        self.model = model
        self.transform = transform

    def fit(self, X, y):
        fwd, _ = TARGET_TRANSFORMS[self.transform]
        self.model_ = clone(self.model).fit(X, fwd(y, X))
        return self

    def predict(self, X):
        _, inv = TARGET_TRANSFORMS[self.transform]
        return np.asarray(inv(self.model_.predict(X), X), dtype=float)


def score_transform(name: str, final: bool = False) -> dict:
    """Validation-only unless final=True. Iterate without --final; pass it
    once, after the transform is chosen, and record that you did."""
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)
    splits = [("valid", valid)] + ([("test", test)] if final else [])
    # "level_only" is the transform's own baseline: a constant (the training
    # mean of the transformed target) pushed through the inverse, i.e. the
    # trailing level with no model at all. A model on a transformed target
    # has to beat this, or the transform is doing the work, not the model.
    level_only = TransformedTarget(DummyRegressor(strategy="mean"), name)
    out = {}
    for model_name, base in (("level_only", None), ("ridge", "ridge"), ("hgb", "hgb")):
        est = level_only if base is None else TransformedTarget(MODELS[base], name)
        model = est.fit(train[cols], train[TARGET])
        out[model_name] = {
            split: {
                "r2": round(float(r2_score(part[TARGET], model.predict(part[cols]))), 4),
                "mae": round(float(mean_absolute_error(part[TARGET], model.predict(part[cols]))), 1),
            }
            for split, part in splits
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-transform", choices=sorted(TARGET_TRANSFORMS),
                    help="Exercise 4 only: score ridge/hgb on a transformed target "
                         f"and write {EX4_METRICS.name}; metrics.json is untouched")
    ap.add_argument("--final", action="store_true",
                    help="with --target-transform: also score the test split (do this once)")
    ns = ap.parse_args()
    if ns.target_transform:
        res = json.loads(EX4_METRICS.read_text()) if EX4_METRICS.exists() else {}
        key = ns.target_transform + ("" if not ns.final else "__final")
        res[key] = score_transform(ns.target_transform, final=ns.final)
        EX4_METRICS.write_text(json.dumps(res, indent=2))
        print(json.dumps({key: res[key]}, indent=2))
        raise SystemExit(0)
    if ns.final:
        ap.error("--final only applies with --target-transform")
    res = fit_and_score()
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
