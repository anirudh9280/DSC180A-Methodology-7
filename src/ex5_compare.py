"""Exercise 5: where do ridge and persistence differ?

R^2 and MAE disagree on which is better on the test split. R^2 squares
errors, MAE does not, so the disagreement is about the *shape* of the
errors: a few large misses vs. many small ones. This prints that shape.
Run only after the feature set and models are frozen (it scores test).

Run: python -m src.ex5_compare
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, TARGET
from src.features import feature_columns
from src.train import MODELS, temporal_split


def error_profile(y: pd.Series, pred: np.ndarray, dates: pd.Series) -> dict:
    signed = pred - y.to_numpy()
    err = np.abs(signed)
    worst = int(np.argmax(err))
    return {
        "bias": round(float(signed.mean()), 1),  # mean(pred - actual); < 0 = under-forecast
        "mae": round(float(err.mean()), 1),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 1),
        "median_ae": round(float(np.median(err)), 1),
        "p90_ae": round(float(np.percentile(err, 90)), 1),
        "max_ae": round(float(err.max()), 1),
        "max_ae_date": str(dates.iloc[worst].date()),
        "days_over_2000": int((err > 2000).sum()),
    }


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)
    for split, part in (("valid", valid), ("test", test)):
        preds = {}
        for name in ("baseline_persistence", "ridge"):
            preds[name] = MODELS[name].fit(train[cols], train[TARGET]).predict(part[cols])
        print(f"[{split}] n={len(part)}")
        for name, p in preds.items():
            print(f"  {name:<21} {error_profile(part[TARGET], p, part.dteday)}")
        y = part[TARGET].to_numpy()
        ridge_closer = np.abs(y - preds["ridge"]) < np.abs(y - preds["baseline_persistence"])
        print(f"  ridge closer on {int(ridge_closer.sum())}/{len(part)} days "
              f"({ridge_closer.mean():.0%})")
    print("OK ex5_compare")
