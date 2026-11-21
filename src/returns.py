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


def actionable_window(panel: pd.DataFrame, tau: float = 0.6) -> pd.DataFrame:
    """What is left to capture when the shape first becomes predictable?

    For every test episode and every observation fraction we record the return
    already realised between episode onset and the decision date, and the
    return over the 20 trading days *after* the decision.  ``tau`` is the
    confidence needed before a decision is considered issued.

    The point of the table is not to find a profitable trade.  It answers the
    proposal's question -- if the call only becomes reliable after the price
    has moved, the model has no actionable value -- and it can answer it
    either way.
    """
    from .load_data import load_social_signal, wide_attention
    from .models import _FEATURES, build_table, fit_logistic, split_by_date

    ep = pd.read_csv(INTERIM / "episodes.csv", parse_dates=["start", "end"])
    lab = pd.read_csv(INTERIM / "episodes_labeled.csv")
    ep = ep[ep["status"] == "complete"].merge(lab, on="episode_id")
    ep = ep.rename(columns={"shape_final": "shape"})

    Z = wide_attention(load_social_signal("2021-12-31"), "2021-12-31")
    table = build_table(ep, Z)
    train, test = split_by_date(table)
    model = fit_logistic(train)
    classes = list(model.classes_)

    test = test.copy()
    p = model.predict_proba(test[_FEATURES])
    test["p_true"] = p[np.arange(len(test)), [classes.index(v) for v in test["label"]]]

    pn = panel.sort_values(["permno", "date"]).copy()
    pn["px"] = pn.groupby("permno")["ret"].transform(lambda s: (1 + s).cumprod())
    info = ep.set_index("episode_id")

    rows = []
    for e in test.itertuples():
        meta = info.loc[e.episode_id]
        sub = pn[pn["permno"] == meta.permno]
        dates = sub["date"].to_numpy()
        idx = int(np.searchsorted(dates, np.datetime64(meta.start, "D")))
        k = int(round(meta.n_days * e.frac))
        if idx + k + 20 >= len(dates):
            continue
        p0 = sub["px"].iloc[idx]
        p1 = sub["px"].iloc[idx + k]
        p2 = sub["px"].iloc[idx + k + 20]
        rows.append({"episode_id": e.episode_id, "frac": e.frac, "p_true": e.p_true,
                     "issued": bool(e.p_true >= tau),
                     "r_before": p1 / p0 - 1, "r_after": p2 / p1 - 1})
    return pd.DataFrame(rows)


def window_summary(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in sorted(w["frac"].unique()):
        for label, sub in (("all", w[w["frac"] == f]),
                           ("issued", w[(w["frac"] == f) & w["issued"]]),
                           ("not issued", w[(w["frac"] == f) & ~w["issued"]])):
            if len(sub) < 10:
                continue
            rows.append({
                "frac": f, "group": label, "n": len(sub),
                "median_r_before": sub["r_before"].median(),
                "median_r_after": sub["r_after"].median(),
                "share_negative_after": (sub["r_after"] < 0).mean(),
            })
    return pd.DataFrame(rows)


def residual_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Drawdown after removing what volatility/size/turnover already explain.

    A high-volatility stock has a deep MDD20 almost mechanically, so the raw
    drawdown AUC says nothing about attention.  MDD20 is regressed on those
    three state variables (train only) and the residuals are correlated with
    the attention features.  A correlation near zero means attention carries
    no independent drawdown information.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    train, test = split(df)
    cols = ["vol60", "lsize", "turnover"]
    reg = LinearRegression().fit(train[cols].fillna(train[cols].median()), train["mdd20"])
    resid = test["mdd20"] - reg.predict(test[cols].fillna(test[cols].median()))

    rows = []
    for feat in ("peak_z", "base0", "mean_ratio", "n_days", "n_obs"):
        rho, p = spearmanr(resid, test[feat], nan_policy="omit")
        rows.append({"feature": feat, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = build_panel()
    df = add_targets(episode_outcomes(panel))
    print(f"episodes with outcomes: {len(df)}")
    print("base rates:", {k: round(df[k].mean(), 3) for k in ("y_up", "y_bigup", "y_dd")})
    print("\n=== Model A / B / C ===")
    print(compare(df).pivot(index="outcome", columns="model", values="auc").round(3).to_string())
    print("\n=== how much of drawdown is just volatility ===")
    print(diagnose_drawdown(df).round(3).to_string(index=False))
    print("\n=== drawdown after removing volatility/size/turnover ===")
    print(residual_drawdown(df).round(3).to_string(index=False))
    print("\n=== actionable window (confidence >= 0.6) ===")
    print(window_summary(actionable_window(panel)).round(3).to_string(index=False))
