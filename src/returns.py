"""Forward return and drawdown models (proposal Models A-D).

Model A : prior price/volume state only
Model B : A + attention level
Model C : B + episode shape features
Model D : C + news  (only defined on the 2012-2020H1 overlap)

Two things this module is careful about:

1. Delisting returns are merged before any forward window is computed,
   otherwise the sample is survivorship-biased.
2. Drawdown is mechanically tied to volatility -- a high-vol stock has a deep
   MDD20 almost by construction.  ``diagnose_drawdown`` reports that link so
   the result is not mistaken for a forecasting success.

Run: python -m src.returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .load_data import INTERIM, RAW

A_FEATURES = ["ret20_back", "vol60", "turnover", "lsize", "mkt_ret"]
B_FEATURES = A_FEATURES + ["peak_z", "base0", "mean_ratio"]
C_FEATURES = B_FEATURES + ["n_days", "n_obs"]

TRAIN_END = "2018-01-01"
TEST_START = "2020-01-01"


def _forward_mdd(x: np.ndarray, h: int = 20) -> np.ndarray:
    p = np.cumprod(1.0 + x)
    n = len(p)
    out = np.full(n, np.nan)
    if n <= h:
        return out
    path = np.column_stack([p[k : k + n - h] for k in range(1, h + 1)])
    peak = np.maximum.accumulate(path, axis=1)
    out[: n - h] = (path / peak - 1.0).min(axis=1)
    return out


def build_panel(save: bool = True) -> pd.DataFrame:
    """Daily panel with forward returns, forward drawdown and lagged state."""
    d = pd.read_parquet(RAW / "crsp_daily.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["permno", "date"]).reset_index(drop=True)

    # delisting returns are applied to each firm's last observed day
    dl = pd.read_parquet(RAW / "crsp_delist.parquet")
    real = dl.dropna(subset=["dlret"])[["permno", "dlret"]]
    if len(real):
        last_idx = d.groupby("permno")["date"].idxmax()
        d = d.merge(real, on="permno", how="left")
        m = d.index.isin(last_idx) & d.dlret.notna()
        d.loc[m, "ret"] = (1 + d.loc[m, "ret"]) * (1 + d.loc[m, "dlret"]) - 1
        d = d.drop(columns=["dlret"])

    g = d.groupby("permno", sort=False)
    lr = np.log1p(d["ret"])
    cum = lr.groupby(d["permno"]).cumsum()
    d["r20"] = np.expm1(cum.groupby(d["permno"]).shift(-20) - cum)
    d["mdd20"] = g["ret"].transform(_forward_mdd)
    d["ret20_back"] = g["ret"].transform(lambda x: np.expm1(np.log1p(x).rolling(20).sum()))
    d["vol60"] = g["ret"].transform(lambda x: x.rolling(60).std() * np.sqrt(252))
    d["turnover"] = d["vol"] / (d["shrout"] * 1000)
    d["lsize"] = np.log(d["prc"].abs() * d["shrout"] * 1000)

    mkt = d.groupby("date")["ret"].mean().rename("mkt_ret")
    d = d.merge(mkt, on="date", how="left")

    keep = ["permno", "date", "ret", "prc", "vol", "shrout", "r20", "mdd20",
            "ret20_back", "vol60", "turnover", "lsize", "mkt_ret"]
    panel = d[keep]
    if save:
        panel.to_parquet(INTERIM / "crsp_panel.parquet", index=False)
    return panel


def episode_outcomes(panel: pd.DataFrame, drop_unfinished: bool = True) -> pd.DataFrame:
    """Episodes joined to the state and the forward outcomes at their start."""
    ep = pd.read_csv(INTERIM / "episodes.csv", parse_dates=["start"])
    lab = pd.read_csv(INTERIM / "episodes_labeled.csv")
    ep = ep[ep["status"] == "complete"].merge(lab, on="episode_id", how="inner")
    ep = ep.rename(columns={"shape_final": "shape"})

    cols = ["permno", "date", "ret20_back", "vol60", "turnover", "lsize",
            "mkt_ret", "r20", "mdd20"]
    out = ep.merge(panel[cols], left_on=["permno", "start"], right_on=["permno", "date"],
                   how="left")
    if drop_unfinished:
        out = out.dropna(subset=["r20", "mdd20"])
    return out.reset_index(drop=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y_up"] = (out["r20"] > 0).astype(int)
    out["y_bigup"] = (out["r20"] > 0.10).astype(int)
    out["y_dd"] = (out["mdd20"] < -0.20).astype(int)
    return out


def split(df: pd.DataFrame):
    return (df[df["start"] < TRAIN_END], df[df["start"] >= TEST_START])


def fit_predict(train, test, cols, target, kind="logit"):
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logit":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    else:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
                           class_weight="balanced", random_state=0, verbose=-1),
        )
    model.fit(train[cols], train[target])
    return model.predict_proba(test[cols])[:, 1]


def compare(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    train, test = split(df)
    rows = []
    for target in ("y_up", "y_bigup", "y_dd"):
        for name, cols in (("A", A_FEATURES), ("B", B_FEATURES), ("C", C_FEATURES)):
            for kind in ("logit", "lgbm"):
                p = fit_predict(train, test, cols, target, kind)
                rows.append({
                    "outcome": target, "model": f"{name}-{kind}",
                    "auc": roc_auc_score(test[target], p),
                    "brier": brier_score_loss(test[target], p),
                })
    return pd.DataFrame(rows)


def diagnose_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Show how much of MDD20 is explained by volatility alone."""
    from sklearn.metrics import roc_auc_score

    train, test = split(df)
    rows = []
    for col in ("vol60", "lsize", "turnover", "ret20_back", "peak_z"):
        p = fit_predict(train, test, [col], "y_dd")
        rows.append({"feature": col, "auc_dd": roc_auc_score(test["y_dd"], p)})
    return pd.DataFrame(rows).sort_values("auc_dd", ascending=False)


if __name__ == "__main__":
    panel = build_panel()
    df = add_targets(episode_outcomes(panel))
    print(f"episodes with outcomes: {len(df)}")
    print("base rates:", {k: round(df[k].mean(), 3) for k in ("y_up", "y_bigup", "y_dd")})
    print("\n=== Model A / B / C ===")
    print(compare(df).pivot(index="outcome", columns="model", values="auc").round(3).to_string())
    print("\n=== how much of drawdown is just volatility ===")
    print(diagnose_drawdown(df).round(3).to_string(index=False))
