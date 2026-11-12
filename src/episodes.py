"""Attention episodes: segmentation, shape hints, preview plots.

The rule is deliberately fixed, not tuned.  In words:

    start   attention is at least ``HI`` times its own trailing mean
    end     it has been at most ``LO`` times the baseline frozen at the start
            for ``QUIET_DAYS`` consecutive trading days (a day with no index
            counts as quiet, because the index only exists when a firm has
            >= 10 messages)

The baseline is frozen at episode onset rather than recomputed every day.  A
rolling baseline would be pushed up by the burst itself, so a plateau would be
declared finished after roughly five weeks even while attention stayed high,
and the plateau class could never be observed.

Because the published index is a z-scored principal component, the ratio is
computed on ``exp(z)`` -- see the note at the top of ``load_data.py``.

Shapes are labelled by hand in three classes (single peak / double peak /
plateau).  ``shape_auto`` below is only a suggestion to speed that labelling
up; the human label in the ``shape`` column wins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .load_data import (
    AS_OF_DEFAULT,
    INTERIM,
    as_of_ts,
    load_social_signal,
    wide_attention,
)

HI = 3.0            # start: attention >= 3x its trailing mean
LO = 1.5            # end:   attention falls back to <= 1.5x the mean
QUIET_DAYS = 5      # ... and stays there this many trading days
WINDOW = 60         # trailing window used for the mean, in trading days
MIN_OBS = 10        # need at least this many observed days in the window
MIN_DAYS = 3        # an episode must contain this many observed days
MAX_DAYS = 60       # longer stretches are a level shift, not a burst shape
SHAPES = ("single", "double", "plateau")


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def segment(
    Z: pd.DataFrame,
    as_of=AS_OF_DEFAULT,
    hi: float = HI,
    lo: float = LO,
    quiet_days: int = QUIET_DAYS,
    window: int = WINDOW,
    min_obs: int = MIN_OBS,
    min_days: int = MIN_DAYS,
    max_days: int = MAX_DAYS,
) -> pd.DataFrame:
    """Cut each firm's attention series into episodes.

    Parameters
    ----------
    Z : date x permno matrix from ``wide_attention`` (NaN = no index that day).
    as_of : hard right edge; rows after it are dropped before anything else.

    Returns
    -------
    One row per episode with start/end dates, peak, intensity and a normalised
    attention path.  ``status == 'open'`` means the episode had not finished
    by ``as_of`` and must not be used for shape labels.
    """
    if as_of is not None:
        Z = Z.loc[Z.index <= as_of_ts(as_of)]

    dates = Z.index
    # work on the positive intensity scale: exp(z) is monotone in z
    A = np.exp(Z.astype("float64"))
    base = A.rolling(window, min_periods=min_obs).mean().shift(1)

    rows: list[dict] = []
    n = len(dates)

    for permno in Z.columns:
        z = Z[permno].to_numpy(dtype="float64")
        a = A[permno].to_numpy(dtype="float64")
        b = base[permno].to_numpy(dtype="float64")

        start = a >= hi * b                   # NaN -> False
        i = 0
        while i < n:
            if not start[i]:
                i += 1
                continue

            # Freeze the baseline at episode onset.  If the trailing mean kept
            # moving, a burst would raise its own reference level and every
            # plateau would be declared over after ~5 weeks even while
            # attention stayed high, which makes the plateau class impossible.
            b0 = b[i]
            j = i
            quiet_run = 0
            while j < n:
                if np.isnan(z[j]) or a[j] <= lo * b0:
                    quiet_run += 1
                    if quiet_run >= quiet_days:
                        break
                else:
                    quiet_run = 0
                j += 1

            if j >= n:
                rows.append(_episode(permno, dates, z, a, b0, i, n - 1, n - 1, "open"))
                break

            last_active = j - quiet_days
            if last_active < i:
                last_active = i
            rows.append(
                _episode(permno, dates, z, a, b0, i, last_active, j, "complete")
            )
            i = j + 1

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)

    ep = pd.DataFrame(rows, columns=_COLUMNS)
    ep = ep[ep["n_obs"] >= min_days].reset_index(drop=True)
    # A burst that never fades is a permanent level shift.  Those few episodes
    # (> 2% of the sample) have no meaningful single/double/plateau shape, so
    # they are kept in the table but excluded from labelling.
    ep.loc[(ep["status"] == "complete") & (ep["n_days"] > max_days), "status"] = "long"
    # If the maximum is the first or the last observed day, either the rise or
    # the fall happened while the firm was below the 10-message floor.  A shape
    # label would then describe a curve we never saw.
    ep.loc[(ep["status"] == "complete") & ep["peak_at_edge"], "status"] = "censored"
    ep["episode_id"] = np.arange(len(ep))
    return ep


_COLUMNS = [
    "permno", "start", "end", "confirm", "n_days", "n_obs", "base0",
    "peak_date", "peak_z", "peak_ratio", "mean_ratio", "peak_at_edge",
    "status", "path",
]


def _episode(permno, dates, z, a, b0, i, last, confirm, status) -> dict:
    seg_z = z[i : last + 1]
    seg_a = a[i : last + 1]
    seg_r = seg_a / b0 if b0 and np.isfinite(b0) else np.full(seg_a.shape, np.nan)
    obs = ~np.isnan(seg_z)

    if obs.any():
        idx = np.flatnonzero(obs)
        k = idx[int(np.argmax(seg_z[idx]))]
        peak_date = dates[i + k]
        peak_z = float(seg_z[k])
        peak_ratio = float(np.nanmax(seg_r))
        mean_ratio = float(np.nanmean(seg_r))
        # the peak sits on the first or last observed day: the rise or the fall
        # was never seen, so the episode has no classifiable shape
        at_edge = bool(k == idx[0] or k == idx[-1])
        path = _encode_path(seg_z)
    else:
        peak_date = dates[i]
        peak_z = np.nan
        peak_ratio = np.nan
        mean_ratio = np.nan
        at_edge = True
        path = ""

    return {
        "permno": int(permno),
        "start": dates[i],
        "end": dates[last],
        "confirm": dates[confirm],
        "n_days": int(last - i + 1),
        "n_obs": int(obs.sum()),
        "base0": float(b0) if np.isfinite(b0) else np.nan,
        "peak_date": peak_date,
        "peak_z": peak_z,
        "peak_ratio": peak_ratio,
        "mean_ratio": mean_ratio,
        "peak_at_edge": at_edge,
        "status": status,
        "path": path,
    }


def _encode_path(seg_z: np.ndarray, width: int = 60) -> str:
    """Compact 0-1 rendering of the episode so labels can be read from CSV."""
    v = np.asarray(seg_z, dtype="float64")
    keep = v[~np.isnan(v)]
    if keep.size == 0:
        return ""
    lo, hi = keep.min(), keep.max()
    rng = hi - lo
    if rng <= 1e-12:
        norm = np.full(v.shape, 0.5)
    else:
        norm = (v - lo) / rng
    return ",".join("" if np.isnan(x) else f"{x:.2f}" for x in norm[:width])


def decode_path(path: str) -> np.ndarray:
    if not path:
        return np.array([])
    return np.array([np.nan if s == "" else float(s) for s in path.split(",")])


# --------------------------------------------------------------------------
# shape hints
# --------------------------------------------------------------------------
def describe_path(seg_z: np.ndarray) -> dict:
    """Cheap shape descriptors used to pre-fill the human labelling sheet.

    A peak only counts if it clears a prominence bar and sits in the upper
    half of the episode; a plateau has to be a sustained run near the maximum
    that starts after the initial rise.  Without those two guards, ordinary
    day-to-day noise turns most episodes into spurious "double peaks".
    """
    v = np.asarray(seg_z, dtype="float64")
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n_peaks": 0, "plateau_share": np.nan, "peak_pos": np.nan,
                "rise_days": np.nan, "fall_days": np.nan, "shape_auto": "single"}

    lo, hi = v.min(), v.max()
    rng = hi - lo
    vn = np.full(v.shape, 0.5) if rng <= 1e-12 else (v - lo) / rng
    n = len(vn)

    peaks, prom = _find_peaks(vn, prominence=0.40, distance=max(2, int(round(0.10 * n))))
    sig = [p for p, pr in zip(peaks, prom) if pr >= 0.40 and vn[p] >= 0.5]
    sig = sorted(sig, key=lambda i: -vn[i])[:2]

    flat_share = float(np.mean(vn >= 0.75))

    if len(sig) >= 2:
        i1, i2 = sorted(sig)
        valley = float(vn[i1 : i2 + 1].min())
        if valley < 0.60:
            shape = "double"
        elif flat_share >= 0.40:
            shape = "plateau"
        else:
            shape = "single"
    elif flat_share >= 0.40:
        shape = "plateau"
    else:
        shape = "single"

    k = int(np.argmax(vn))
    return {
        "n_peaks": len(sig),
        "plateau_share": round(flat_share, 3),
        "peak_pos": round(k / max(1, n - 1), 3),
        "rise_days": k + 1,
        "fall_days": n - k,
        "shape_auto": shape,
    }


def _find_peaks(vn: np.ndarray, prominence: float = 0.3,
                distance: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima, with the two endpoints allowed to be peaks."""
    padded = np.concatenate(([-0.1], vn, [-0.1]))
    try:
        from scipy.signal import find_peaks

        peaks, props = find_peaks(padded, prominence=prominence, distance=distance)
        return peaks - 1, np.asarray(props.get("prominences", []), dtype="float64")
    except Exception:
        peaks = []
        for i in range(1, len(vn) - 1):
            if vn[i] >= vn[i - 1] and vn[i] >= vn[i + 1] and vn[i] >= 0.5:
                peaks.append(i)
        return np.asarray(peaks), np.ones(len(peaks))


def label_sheet(ep: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    """Completed episodes with descriptors and an empty ``shape`` column.

    Fill ``shape`` with one of ``single`` / ``double`` / ``plateau``; put the
    CSV back in ``data/interim/episodes_labeled.csv``.
    """
    done = ep.loc[ep["status"] == "complete"].copy()
    desc = pd.DataFrame(
        [describe_path(Z.loc[e.start : e.end, e.permno].to_numpy()) for e in done.itertuples()],
        index=done.index,
    )
    out = pd.concat([done, desc], axis=1)
    out["shape"] = ""
    return out.sort_values(["permno", "start"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# preview plots
# --------------------------------------------------------------------------
def plot_examples(ep: pd.DataFrame, Z: pd.DataFrame, out_path=None, n: int = 30,
                  context: int = 5) -> Path:
    """Grid of ``n`` episodes spread across the sample, for eyeballing."""
    import os

    import matplotlib

    # keep matplotlib's font cache inside the repo; the default location under
    # the user profile is outside the sandbox and fails to write
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done = ep.loc[ep["status"] == "complete"]
    if done.empty:
        raise ValueError("no completed episodes to plot")

    step = max(1, len(done) // n)
    picks = done.iloc[::step].head(n)
    cols = 5
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.3 * rows), squeeze=False)

    dates = Z.index
    for ax, e in zip(axes.ravel(), picks.itertuples()):
        i = dates.get_loc(e.start)
        j = dates.get_loc(e.end)
        lo = max(0, i - context)
        hi = min(len(dates) - 1, j + context)
        y = Z[e.permno].iloc[lo : hi + 1].to_numpy()
        ax.plot(np.arange(lo, hi + 1), y, lw=1.2, color="#1f4e79")
        ax.axvspan(i, j, color="#f2c14e", alpha=0.35, lw=0)
        ax.axvline(i, color="#c0392b", lw=0.8)
        ax.set_title(
            f"#{e.episode_id} {e.permno} {pd.Timestamp(e.start).date()}\n"
            f"{e.n_days}d  {describe_path(decode_path(e.path))['shape_auto']}",
            fontsize=7,
        )
        ax.tick_params(labelsize=6)

    for ax in axes.ravel()[len(picks) :]:
        ax.axis("off")

    fig.suptitle("Attention episodes (shaded = detected episode, z-scored index)", fontsize=11)
    fig.tight_layout()

    out = Path(out_path) if out_path else Path("notebooks/figures/episodes_preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    as_of = AS_OF_DEFAULT
    df = load_social_signal(as_of)
    Z = wide_attention(df, as_of)
    ep = segment(Z, as_of=as_of)

    print(f"as_of            : {as_of}")
    print(f"episodes         : {len(ep):,}")
    print(ep["status"].value_counts().to_string())
    if len(ep):
        complete = ep.loc[ep["status"] == "complete"]
        print(f"\nfirms with >=1 episode : {ep['permno'].nunique():,} / {Z.shape[1]:,}")
        print("\nepisodes per firm:")
        print(ep.groupby("permno").size().describe(percentiles=[0.5, 0.9]).round(1).to_string())
        print("\nepisode length in days (complete):")
        print(complete["n_days"].describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_string())
        print("\nepisodes per calendar year:")
        print(ep["start"].dt.year.value_counts().sort_index().to_string())

        sheet = label_sheet(ep, Z)
        print("\nauto shape hints (to be replaced by hand):")
        print(sheet["shape_auto"].value_counts().to_string())

        out = INTERIM / "episodes.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        ep.to_csv(out, index=False)
        sheet.drop(columns=["path"]).to_csv(INTERIM / "episodes_to_label.csv", index=False)
        print(f"\nwrote {out}")
        print(f"wrote {INTERIM / 'episodes_to_label.csv'}")

        png = plot_examples(ep, Z)
        print(f"wrote {png}")
