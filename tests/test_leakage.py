"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import DERIVED_FEATURES, FORBIDDEN, RAW_FEATURES, TARGET
from src.features import build_features, encode_weather, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_feature_set_is_exactly_the_declared_set(feats):
    """test_no_forbidden_features checks names against FORBIDDEN, so a leaked
    column under a new name (`out["reg"] = out.registered`, or `casual`
    renamed in the CSV) sails through it. This compares what the model
    actually receives against the independent list in src/config.py."""
    actual = set(feature_columns(feats))
    declared = RAW_FEATURES | DERIVED_FEATURES
    assert actual == declared, (
        f"undeclared features {sorted(actual - declared)}; "
        f"declared but missing {sorted(declared - actual)}"
    )


def test_no_feature_set_reconstructs_the_target(feats):
    """Value-based, not name-based: if any linear combination of the model's
    inputs reproduces cnt exactly (casual + registered under any names), the
    least-squares residual collapses to ~0. Honest features leave residuals
    of hundreds of rides."""
    cols = feature_columns(feats)
    X = np.column_stack([feats[cols].to_numpy(dtype=float), np.ones(len(feats))])
    y = feats[TARGET].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    max_resid = float(np.abs(X @ coef - y).max())
    assert max_resid > 1.0, f"features reconstruct {TARGET} exactly (max residual {max_resid:.2e})"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_roll_28_is_mean_of_28_prior_calendar_days(feats):
    """cnt_roll_28 on 2011-06-15 must equal the mean of raw cnt over the 28
    calendar days strictly before it (2011-05-18 .. 2011-06-14). Expected is
    computed from load_raw(), not from feats, so it cannot share a bug with
    build_features. Unlike the roll_7 test there is no `if` guard: a gap in
    the window fails loudly instead of passing vacuously."""
    raw = load_raw().set_index("dteday")[TARGET]
    date = pd.Timestamp("2011-06-15")
    window = pd.date_range(date - pd.Timedelta(days=28), date - pd.Timedelta(days=1), freq="D")
    assert len(window) == 28
    assert window.isin(raw.index).all(), "window has a gap in the raw data; pick another date"
    expected = raw[window].sum() / 28
    assert feats.set_index("dteday").cnt_roll_28[date] == pytest.approx(expected)


def test_roll_28_drops_windows_containing_the_gap(feats):
    """2011-03-10 is absent from the raw data, so every date whose 28-day
    window contains it (2011-03-11 .. 2011-04-07) has an incomplete history
    and must be dropped -- not given a 27-day mean by a min_periods setting."""
    raw = load_raw()
    gap = pd.Timestamp("2011-03-10")
    assert gap not in set(raw.dteday), "gap day is present; this test is moot"
    affected = pd.date_range(gap + pd.Timedelta(days=1), gap + pd.Timedelta(days=28), freq="D")
    assert affected[-1] == pd.Timestamp("2011-04-07")
    leaked = affected[affected.isin(feats.dteday)]
    assert leaked.empty, f"rows kept despite incomplete 28-day window: {list(leaked.date)}"
    # The first date after the affected run must survive, or the test is too broad.
    assert pd.Timestamp("2011-04-08") in set(feats.dteday)


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique


def test_encode_weather_rejects_unseen_labels():
    """The raw data only holds GOOD/MISTY/BAD, so no data-driven test can see
    this guard. Without it `.map()` turns a new label into NaN and the later
    `dropna()` deletes those days silently -- feed it one directly."""
    s = pd.Series(["GOOD", "STORMY", "BAD"])
    try:
        out = encode_weather(s)
    except ValueError as e:
        assert "STORMY" in str(e)
    else:
        pytest.fail(f"unseen label 'STORMY' accepted silently; got {out.tolist()}")
