"""Load the two data legs used in this project.

1. Social Signal firm-day indices (main attention signal, 2012-2021).
2. CRSP daily returns (used only to build forward labels).

Point-in-time convention
------------------------
Every public function takes ``as_of`` and returns rows with ``date <= as_of``.
The attention index for day t is treated as known after the close of t, so
labels start at t+1.  Nothing here looks forward; the two functions that do
look forward (``add_forward_labels``, ``add_forward_drawdown``) are label
builders and are never allowed to feed a feature.

Why we exponentiate the index
-----------------------------
``zee_attn_pc`` is a standardised first principal component: its median is
about -0.21 and it takes negative values.  A rule such as "3x the trailing
60-day mean" is undefined on a series that can be negative (3 x a negative
mean is negative, so every ordinary day would trigger).  We therefore work on
the positive intensity scale ``A = exp(z)``, which is a monotone transform of
the published index.  A ratio of 3 on that scale means the index sits
``ln(3) ~= 1.10`` above the log of its trailing mean; nothing is lost, the
threshold just becomes well defined.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

SOCIAL_PATH = RAW / "social_signal_index.dta"
CRSP_PATH = RAW / "crsp_daily.parquet"

AS_OF_DEFAULT = "2021-12-31"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def as_of_ts(as_of) -> pd.Timestamp:
    """Normalise anything date-like to a tz-naive midnight Timestamp."""
    ts = pd.Timestamp(as_of)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


# --------------------------------------------------------------------------
# Social Signal
# --------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _read_social(path: str) -> pd.DataFrame:
    df = pd.read_stata(path, convert_dates=True)
    df = df.rename(columns={"zee_attn_pc": "attention", "zee_sent_pc": "sentiment"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["permno"] = df["permno"].astype("int32")
    return df.sort_values(["permno", "date"]).reset_index(drop=True)


def load_social_signal(as_of=AS_OF_DEFAULT, path=SOCIAL_PATH) -> pd.DataFrame:
    """Firm-day attention and sentiment, truncated at ``as_of``.

    Returns columns: permno, date, attention, sentiment.
    Only firm-days with at least 10 social messages are present in the source
    file; a missing row therefore means "little or no discussion", not a gap
    to be filled with an average.
    """
    ts = as_of_ts(as_of)
    df = _read_social(str(path))
    out = df.loc[df["date"] <= ts].copy()
    if len(out):
        assert out["date"].max() <= ts, "as_of filter failed"
    return out


def trading_calendar(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Trading days implied by the data (union of observed dates)."""
    return pd.DatetimeIndex(np.sort(df["date"].unique()))


def wide_attention(df: pd.DataFrame, as_of=AS_OF_DEFAULT, calendar=None) -> pd.DataFrame:
    """Date x permno matrix of the attention index.

    The grid is the full trading calendar, so a NaN cell means "no index that
    day" (fewer than 10 messages) and is interpreted later as quiet, not as an
    unknown value.
    """
    ts = as_of_ts(as_of)
    cal = pd.DatetimeIndex(calendar) if calendar is not None else trading_calendar(df)
    cal = cal[cal <= ts]
    wide = df.pivot(index="date", columns="permno", values="attention")
    wide = wide.reindex(index=cal)
    wide.columns.name = "permno"
    return wide.astype("float32")


def wide_sentiment(df: pd.DataFrame, as_of=AS_OF_DEFAULT, calendar=None) -> pd.DataFrame:
    """Same grid as ``wide_attention`` but for the sentiment index."""
    ts = as_of_ts(as_of)
    cal = pd.DatetimeIndex(calendar) if calendar is not None else trading_calendar(df)
    cal = cal[cal <= ts]
    wide = df.pivot(index="date", columns="permno", values="sentiment")
    wide = wide.reindex(index=cal)
    wide.columns.name = "permno"
    return wide.astype("float32")


def coverage_table(df: pd.DataFrame, as_of=AS_OF_DEFAULT) -> pd.DataFrame:
    """Per-firm coverage: how many days the index exists, and how dense it is.

    ``density`` is observed days divided by trading days between the firm's
    first and last observation, so it measures sparsity rather than age.
    """
    ts = as_of_ts(as_of)
    d = df.loc[df["date"] <= ts]
    cal = trading_calendar(d)
    pos = pd.Series(np.arange(len(cal)), index=cal)

    g = d.groupby("permno")["date"]
    out = pd.DataFrame(
        {
            "n_obs": g.size(),
            "first": g.min(),
            "last": g.max(),
        }
    )
    span = pos.reindex(out["last"]).to_numpy() - pos.reindex(out["first"]).to_numpy() + 1
    out["span_days"] = span
    out["density"] = (out["n_obs"] / out["span_days"]).round(3)
    return out.sort_values("n_obs", ascending=False)


# --------------------------------------------------------------------------
# CRSP labels
# --------------------------------------------------------------------------
def load_crsp(as_of=AS_OF_DEFAULT, path=CRSP_PATH) -> pd.DataFrame:
    """Daily CRSP returns, truncated at ``as_of``.

    Expected columns after loading: permno, date, ret (and optionally prc,
    shrout, vol, exchcd, hsiccd, dlret).  The file is not redistributed with
    the code; export it from WRDS and drop it in ``data/raw/``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Export daily CRSP (permno, date, ret, prc, vol) "
            "from WRDS to data/raw/ as parquet or csv."
        )
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in {".csv", ".txt"}:
        df = pd.read_csv(p)
    elif p.suffix == ".dta":
        df = pd.read_stata(p, convert_dates=True)
    else:
        raise ValueError(f"unsupported file type: {p.suffix}")

    df = df.rename(columns={c: c.lower() for c in df.columns})
    missing = {"permno", "date", "ret"} - set(df.columns)
    if missing:
        raise ValueError(f"CRSP file is missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["permno"] = df["permno"].astype("int32")
    ts = as_of_ts(as_of)
    return df.loc[df["date"] <= ts].sort_values(["permno", "date"]).reset_index(drop=True)


def add_forward_returns(px: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Compounded return over t+1 .. t+horizon.

    Uses cumulative log returns so a single shift does the work.  The last
    ``horizon`` rows of each firm are NaN by construction, which is the
    correct behaviour: the outcome is not yet observed.
    """
    df = px.sort_values(["permno", "date"]).copy()
    lr = np.log1p(df["ret"].astype("float64"))
    cum = lr.groupby(df["permno"]).cumsum()
    fwd_cum = cum.groupby(df["permno"]).shift(-horizon)
    df[f"r{horizon}"] = np.expm1(fwd_cum - cum)
    return df


def add_forward_drawdown(px: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Largest peak-to-trough loss inside t+1 .. t+horizon (<= 0).

    The running peak starts at t+1, i.e. the day-t price is not part of the
    peak, matching the proposal's definition.
    """
    df = px.sort_values(["permno", "date"]).copy()
    out = np.full(len(df), np.nan)

    for _, idx in df.groupby("permno").indices.items():
        idx = np.sort(idx)
        p = (1.0 + df["ret"].to_numpy()[idx]).cumprod()
        n = len(p)
        if n <= horizon:
            continue
        path = np.column_stack([p[k : k + n - horizon] for k in range(1, horizon + 1)])
        peak = np.maximum.accumulate(path, axis=1)
        dd = (path / peak - 1.0).min(axis=1)
        out[idx[: n - horizon]] = dd

    df[f"mdd{horizon}"] = out
    return df


def add_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Attach a within-firm price index (base 1) for drawdown calculations."""
    df = px.sort_values(["permno", "date"]).copy()
    df["pxidx"] = (1.0 + df["ret"].fillna(0.0)).groupby(df["permno"]).cumprod()
    return df


# --------------------------------------------------------------------------
# smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    as_of = AS_OF_DEFAULT
    df = load_social_signal(as_of)
    Z = wide_attention(df, as_of)

    print(f"as_of            : {as_of}")
    print(f"rows             : {len(df):,}")
    print(f"firms            : {df['permno'].nunique():,}")
    print(f"dates            : {df['date'].nunique():,}")
    print(f"date range       : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"wide matrix      : {Z.shape[0]} dates x {Z.shape[1]} firms")
    print(f"observed cells   : {Z.notna().to_numpy().sum():,} "
          f"({Z.notna().to_numpy().mean():.1%} of the grid)")

    cov = coverage_table(df, as_of)
    print("\ncoverage per firm (days with an index):")
    print(cov["n_obs"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_string())
    print("\nindex quantiles:")
    print(df["attention"].describe(percentiles=[0.01, 0.5, 0.95, 0.99]).round(3).to_string())
