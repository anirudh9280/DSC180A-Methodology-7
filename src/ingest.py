"""Stage 1: fetch the raw dataset and assert its shape.

Run: python -m src.ingest
"""
import hashlib
import sys
import urllib.request

import pandas as pd

from src.config import DATA_URL, RAW, RAW_SHA256

EXPECTED_MIN_ROWS = 700
REQUIRED_COLS = {"dteday", "cnt", "casual", "registered", "temp", "hum", "windspeed"}


def download(force: bool = False) -> None:
    if RAW.exists() and not force:
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, RAW)


def raw_sha256() -> str:
    return hashlib.sha256(RAW.read_bytes()).hexdigest()


def verify_raw() -> None:
    """The raw file is immutable. Any byte change -- an edited value, an
    appended column, a re-download of a changed mirror -- stops the pipeline
    here instead of shifting every metric downstream without a word."""
    got = raw_sha256()
    if got != RAW_SHA256:
        raise SystemExit(
            f"FAIL ingest: {RAW.name} sha256 {got[:16]}... != pinned {RAW_SHA256[:16]}...\n"
            f"  The raw data changed. Restore it: git checkout -- {RAW.relative_to(RAW.parents[2])}"
        )


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["dteday"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"raw data is missing required columns: {sorted(missing)}")
    if len(df) < EXPECTED_MIN_ROWS:
        raise ValueError(f"raw data has only {len(df)} rows; expected >= {EXPECTED_MIN_ROWS}")
    return df


if __name__ == "__main__":
    download(force="--force" in sys.argv)
    verify_raw()
    df = load_raw()
    print(f"OK ingest: {len(df)} rows, {df.dteday.min().date()} -> {df.dteday.max().date()}, "
          f"sha256 {RAW_SHA256[:16]}")
