"""Shape classification from observed prefixes.

Given an episode that is still unfolding, we only see its first 20% / 40% /
60% (measured in trading days of the completed episode).  The task is to
predict which of three shapes it will end up as: ``single``, ``double`` or
``plateau``.

Two models, on purpose:

* regularised logistic regression -- the honest baseline, easy to defend;
* LightGBM -- the comparison, to see whether non-linearity buys anything.

Both are fitted on episodes that started before ``CUTOFF`` and tested on later
ones.  Splitting by episode date, not by row, keeps prefixes of the same
episode on the same side of the split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .episodes import SHAPES

PREFIX_FRACTIONS = (0.2, 0.4, 0.6)
CUTOFF = "2018-01-01"
LABEL_COL = "shape"


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def prefix_features(seg_z: np.ndarray, frac: float) -> dict:
    """Descriptors of the observed part of an episode.

    Everything is computed from the prefix only; the full path is never
    touched, otherwise the label leaks into the input.
    """
    v = np.asarray(seg_z, dtype="float64")
    k = max(3, int(round(len(v) * frac)))
    k = min(k, len(v))
    obs = v[:k]
    obs = obs[~np.isnan(obs)]

    out = {"frac": float(frac), "elapsed_days": float(k), "n_obs": float(len(obs))}
    if len(obs) < 2:
        return {**out, **{f"{c}": np.nan for c in _FEATURES}}

    lo, hi = float(obs.min()), float(obs.max())
    rng = hi - lo
    scale = rng if rng > 1e-12 else 1.0

    slope = _slope(obs)
    half = len(obs) // 2
    if half >= 2:
        slope_first = _slope(obs[:half])
        slope_second = _slope(obs[half:])
        accel = (slope_second - slope_first) / scale
    else:
        accel = np.nan

    out.update(
        {
            "level_last": float(obs[-1]),
            "level_max": hi,
            "level_min": lo,
            "level_mean": float(obs.mean()),
            "level_std": float(obs.std()),
            "range": float(rng),
            "slope": slope,
            "slope_norm": slope / scale,
            "accel_norm": accel,
            "peak_pos": float(np.argmax(obs) / max(1, len(obs) - 1)),
            "last_rel": float((obs[-1] - lo) / scale),
            "above_hi_share": float(np.mean(obs >= lo + 0.75 * rng)),
            "n_local_peaks": float(_n_local_peaks(obs)),
            "trend_ratio": float(np.mean(np.diff(obs) > 0)) if len(obs) > 2 else np.nan,
        }
    )
    return out


def _slope(y: np.ndarray) -> float:
    """Least-squares slope of y against 0..n-1, without polyfit warnings."""
    n = len(y)
    if n < 2:
        return np.nan
    t = np.arange(n, dtype="float64")
    dt = t - t.mean()
    denom = float(np.sum(dt * dt))
    if denom <= 0:
        return np.nan
    return float(np.sum(dt * (y - y.mean())) / denom)


_FEATURES = [
    "level_last", "level_max", "level_min", "level_mean", "level_std",
    "range", "slope", "slope_norm", "accel_norm", "peak_pos", "last_rel",
    "above_hi_share", "n_local_peaks", "trend_ratio",
]


def _n_local_peaks(v: np.ndarray) -> int:
    if len(v) < 3:
        return 1
    return int(np.sum((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])))


def build_table(ep: pd.DataFrame, Z: pd.DataFrame, fracs=PREFIX_FRACTIONS,
                label_col: str = LABEL_COL) -> pd.DataFrame:
    """One row per (episode, prefix fraction).

    ``ep`` must carry a filled ``shape`` column; open episodes and unlabelled
    rows are dropped rather than guessed.
    """
    if label_col not in ep.columns:
        raise KeyError(f"no '{label_col}' column; fill it in the labelling sheet first")

    d = ep.loc[(ep["status"] == "complete") & ep[label_col].isin(SHAPES)].copy()
    if d.empty:
        raise ValueError("no labelled completed episodes")

    rows = []
    for e in d.itertuples():
        seg = Z.loc[e.start : e.end, e.permno].to_numpy(dtype="float64")
        if len(seg) < 4:
            continue
        for frac in fracs:
            f = prefix_features(seg, frac)
            f.update(
                {
                    "episode_id": e.episode_id,
                    "permno": e.permno,
                    "start": e.start,
                    "label": getattr(e, label_col),
                }
            )
            rows.append(f)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def split_by_date(table: pd.DataFrame, cutoff: str = CUTOFF):
    train = table.loc[table["start"] < pd.Timestamp(cutoff)]
    test = table.loc[table["start"] >= pd.Timestamp(cutoff)]
    return train, test


def fit_logistic(train: pd.DataFrame, feature_cols=None):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = feature_cols or _FEATURES
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    model.fit(train[cols], train["label"])
    return model


def fit_lightgbm(train: pd.DataFrame, feature_cols=None):
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "LightGBM is not available in this environment. "
            "Install it with: pip install lightgbm"
        ) from exc

    cols = feature_cols or _FEATURES
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=0,
        verbose=-1,
    )
    model.fit(train[cols], train["label"])
    return model


def score(model, test: pd.DataFrame, feature_cols=None) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    cols = feature_cols or _FEATURES
    y = test["label"].to_numpy()
    proba = model.predict_proba(test[cols])
    classes = model.classes_
    pred = classes[np.argmax(proba, axis=1)]

    onehot = (y[:, None] == classes[None, :]).astype("float64")
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, proba, labels=classes)),
        "brier": brier,
    }


def majority_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    top = train["label"].value_counts().idxmax()
    y = test["label"].to_numpy()
    pred = np.full(len(y), top)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": np.nan,
        "brier": np.nan,
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from .load_data import AS_OF_DEFAULT, INTERIM, load_social_signal, wide_attention

    # Optional CLI arg: path to a label file.  The label file only needs
    # (episode_id, shape_final); everything else comes from episodes.csv.
    labeled = Path(sys.argv[1]) if len(sys.argv) > 1 else INTERIM / "episodes_labeled.csv"
    if not labeled.exists():
        print(f"No labels yet: {labeled}")
        print("1. open notebooks/label_review.html in a browser")
        print("2. label the 97 unresolved episodes and click '导出 CSV'")
        print("3. put the downloaded episodes_labeled.csv into data/interim/")
        print("4. run this module again")
        raise SystemExit(0)

    labels = pd.read_csv(labeled)
    if "shape_final" not in labels.columns:
        raise KeyError(f"{labeled} must have an 'episode_id' and 'shape_final' column")

    ep = pd.read_csv(INTERIM / "episodes.csv", parse_dates=["start", "end", "confirm", "peak_date"])
    ep = ep.merge(labels[["episode_id", "shape_final"]], on="episode_id", how="left")
    ep = ep.rename(columns={"shape_final": "shape"})
    ep = ep.loc[ep["shape"].isin(SHAPES)].copy()
    if ep.empty:
        raise SystemExit("no labelled episodes found")

    df = load_social_signal(AS_OF_DEFAULT)
    Z = wide_attention(df, AS_OF_DEFAULT)

    table = build_table(ep, Z)
    train, test = split_by_date(table)
    print(f"episodes labelled {ep['episode_id'].nunique():,}")
    print(f"rows {len(table):,} | train {len(train):,} | test {len(test):,}")
    print("train labels:", train["label"].value_counts().to_dict())
    print("test  labels:", test["label"].value_counts().to_dict())

    for name, fit in (("logistic", fit_logistic), ("lightgbm", fit_lightgbm)):
        try:
            model = fit(train)
            print(f"\n{name}: {score(model, test)}")
        except Exception as exc:
            print(f"\n{name}: skipped ({exc})")
