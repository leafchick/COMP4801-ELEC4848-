"""Model D: does news add anything beyond price state and attention?

The news panel is built from FNSPID daily counts mapped onto CRSP permnos
through the dated ticker table (``crsp.dsenames``), so a ticker that was
reused by another firm does not leak across.

What the data can and cannot support is documented in ``M2_report.md``:
FNSPID covers 2012-01 to 2020-06 and is 93.6% aggregator content, so this is
a test of news *intensity*, not of source independence.

Run: python -m src.news
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .load_data import INTERIM, RAW
from .returns import C_FEATURES, add_targets, build_panel, episode_outcomes

D_FEATURES = C_FEATURES + ["lnews", "news_ratio", "news_other_share", "news_days_5d"]
NEWS_END = "2020-06-11"


def build_news_panel() -> pd.DataFrame:
    """(permno, date) counts, with the ticker mapped inside its valid dates."""
    news = pd.read_csv(RAW / "fnspid_daily_v2.csv", parse_dates=["date"])
    names = pd.read_parquet(RAW / "crsp_names.parquet")
    for c in ("namedt", "nameendt"):
        names[c] = pd.to_datetime(names[c])

    m = news.merge(names[["permno", "ticker", "namedt", "nameendt"]],
                   left_on="symbol", right_on="ticker")
    m = m[(m["date"] >= m["namedt"]) & (m["date"] <= m["nameendt"])]

    pn = (m.groupby(["permno", "date"])
            .agg(n_news=("n_news", "sum"), n_unique=("n_unique", "sum"),
                 n_agg=("n_aggregator", "sum"))
            .reset_index())
    pn["n_other"] = pn["n_unique"] - pn["n_agg"]
    pn = pn.sort_values(["permno", "date"])
    pn["base60"] = pn.groupby("permno")["n_unique"].transform(
        lambda s: s.rolling(60, min_periods=10).mean())
    return pn


def attach_news(df: pd.DataFrame, pn: pd.DataFrame) -> pd.DataFrame:
    """Episode-level news around the onset (5 calendar days either side)."""
    out = df.merge(pn[["permno", "date", "base60"]], left_on=["permno", "start"],
                   right_on=["permno", "date"], how="left", suffixes=("", "_news"))

    rows = []
    for e in out.itertuples():
        s = pn[(pn["permno"] == e.permno)
               & (pn["date"] >= e.start - pd.Timedelta(days=5))
               & (pn["date"] <= e.start + pd.Timedelta(days=5))]
        rows.append((s["n_unique"].sum(), s["n_other"].sum(), len(s), s["n_unique"].max() if len(s) else np.nan))
    out[["news_5d", "news_other_5d", "news_days_5d", "base60_news"]] = pd.DataFrame(rows, index=out.index)

    out["lnews"] = np.log1p(out["news_5d"])
    out["news_other_share"] = out["news_other_5d"] / out["news_5d"].replace(0, np.nan)
    base = out["base60"].fillna(0)
    out["news_ratio"] = (out["news_5d"] + 1) / (base * 10 + 1)
    return out


def compare_c_vs_d(ep: pd.DataFrame) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sub = ep[(ep["start"] <= NEWS_END) & ep["news_5d"].notna()].copy()
    train = sub[sub["start"] < "2018-01-01"]
    test = sub[sub["start"] >= "2018-01-01"]

    def auc(cols, y):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(train[cols], train[y])
        return roc_auc_score(test[y], model.predict_proba(test[cols])[:, 1])

    rows = []
    for y in ("y_up", "y_bigup"):
        rows.append({"outcome": y, "n_test": len(test),
                     "C": auc(C_FEATURES, y), "D": auc(D_FEATURES, y),
                     "news_only": auc(["lnews"], y)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = build_panel()
    df = add_targets(episode_outcomes(panel))
    pn = build_news_panel()
    print(f"news panel: {pn.shape}, permnos {pn['permno'].nunique()}")
    ep = attach_news(df, pn)
    covered = ep[(ep["start"] <= NEWS_END) & ep["news_5d"].notna()]
    print(f"episodes inside the news window: {len(covered)}; "
          f"with any news in +-5d: {int((covered['news_5d'] > 0).sum())}")
    print("\n=== C vs D ===")
    print(compare_c_vs_d(ep).round(3).to_string(index=False))
