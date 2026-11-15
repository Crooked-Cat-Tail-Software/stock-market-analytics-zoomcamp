# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Research: 10-day stock selection for US large caps
#
# **Question.** Using only information known at today's close, which of ~190 large US stocks are most
# likely to go up (or to beat the S&P 500) over the **next 10 trading days** — and can a trading
# strategy built on those predictions beat simply holding the S&P 500 after costs?
#
# This notebook reads the outputs produced by `python main.py research` (data, models, predictions)
# and documents: data sources → feature sets → EDA → models → decision rules → simulations → deep
# analysis of where the predictions work and where they fail.

# %%
import os, sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
PROJECT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(PROJECT); sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading import config as C, pipeline as P
from trading.data_repo import DataRepository
from trading.transform import TransformData, NEW_FEATURES
from trading.simulate import PortfolioSimulator, Strategy, vectorized_sim, perf_metrics

pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)
plt.rcParams["figure.figsize"] = (11, 4.5)
R = C.REPORTS_DIR

# %% [markdown]
# ## 1. Data sources
#
# | Source | What | Used for |
# |---|---|---|
# | yfinance | daily OHLCV of 190 US stocks > $50B market cap (course list) since 1998 | price/volume features, TA-Lib, targets |
# | yfinance | S&P 500, Dow, Nasdaq, Russell 2000, VIX, gold, WTI oil, Bitcoin, US dollar index, SPY | market features, benchmark |
# | FRED | potential GDP, core CPI, Fed funds, unemployment, 3m/1y/5y/10y treasuries, Baa credit spread | macro features |
# | **CBOE** (new) | VIX9D, VIX3M, VVIX, SKEW official history | volatility term structure, tail risk |
# | **Wikipedia** (new, scraping) | GICS sector / sub-industry of S&P 500 members | sector dummies, sector-relative features |

# %%
repo = DataRepository().load()
t = TransformData(repo); df = t.load(); fs = t.feature_sets
print(f"raw stock rows: {len(repo.stocks_df):,} | modelling rows (2000+): {len(df):,} | columns: {df.shape[1]}")
print(f"tickers: {df.Ticker.nunique()} | dates: {df.Date.min().date()} .. {df.Date.max().date()}")
repo.sectors_df.sector.value_counts().rename("stocks per sector").to_frame().T

# %% [markdown]
# ## 2. One dataframe, defined feature sets
#
# * `TO_PREDICT` — targets and future information (never used as inputs)
# * `NUMERIC` — 300 numeric inputs: lecture features (growth windows, SMA, volatility, TA-Lib momentum /
#   volume / volatility / cycle indicators, 61 candlestick patterns, index & commodity growth, macro)
#   plus the new features below
# * `DUMMIES` — one-hot of `Month`, `Weekday`, `sector` (Ticker / week-of-month dummies were left out:
#   in homework 4 they mainly helped the model memorise the training period)
#
# Two transformations worth noting:
# * TA-Lib indicators measured in **dollars** (MACD, ATR, APO, MOM, …) are divided by the close so a
#   $20 and an $800 stock are comparable.
# * Monthly / quarterly FRED values are joined on the date they were **published**, not the period
#   they describe (CPI for October is only known mid-November) — avoids look-ahead.

# %%
for k in ("TO_PREDICT", "CATEGORICAL"):
    print(k, fs[k])
print("NUMERIC:", len(fs["NUMERIC"]), "features, of which candlestick patterns:", len(fs["PATTERNS"]))
print(f"\nNEW features (not in the lectures): {len(NEW_FEATURES)} groups")
print(", ".join(NEW_FEATURES))

# %% [markdown]
# ## 3. Exploratory data analysis
# ### 3.1 The targets

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
(df[C.TARGET_GROWTH] - 1).clip(-0.3, 0.3).hist(bins=120, ax=ax[0])
ax[0].set_title("10-day forward return (clipped ±30%)")
df.groupby("Year")[[C.TARGET, C.TARGET_REL]].mean().plot(ax=ax[1], title="share of positive labels per year")
ax[1].axhline(0.5, color="grey", lw=0.8)
df.groupby("Date").Ticker.count().plot(ax=ax[2], title="stocks available per day")
plt.tight_layout(); plt.show()
df[[C.TARGET_GROWTH, C.TARGET, C.TARGET_REL]].describe().T

# %% [markdown]
# The absolute label is positive 56% of the time and swings strongly by year (2008: ~40%, 2009/2013: ~65%),
# i.e. it is dominated by the direction of the whole market. The relative label ("beats the S&P 500")
# stays close to 50% every year — it only asks the model to *rank* stocks, which is what a top-k strategy
# needs. The universe grows from ~110 to 190 stocks (newer listings).

# %% [markdown]
# ### 3.2 Correlation of features with the targets
# Spearman rank correlation (robust to outliers) between every numeric feature and the 10-day forward
# return, on the training period only. Forward returns are noisy, so even the strongest single features
# have |ρ| of a few percent — a useful reality check before modelling.

# %%
train = df[df.Date <= C.TRAIN_END].sample(250_000, random_state=1)
num = [c for c in fs["NUMERIC"] if c not in fs["PATTERNS"]]
corr = pd.DataFrame({
    "rho_growth_future_10d": train[num].corrwith(train[C.TARGET_GROWTH], method="spearman"),
    "rho_outperform_snp500": train[num].corrwith(train[C.TARGET_REL], method="spearman"),
})
corr["abs"] = corr.rho_growth_future_10d.abs()
corr["is_new"] = [c in NEW_FEATURES or any(c.startswith(n.rstrip("*")) for n in NEW_FEATURES if n.endswith("*"))
                  for c in corr.index]
top_corr = corr.sort_values("abs", ascending=False).head(20)
top_corr

# %%
top_corr.rho_growth_future_10d.sort_values().plot.barh(figsize=(8, 6), title="top-20 |Spearman ρ| with 10d forward return (train)")
plt.show()

# %%
new_sel = ["excess_ret_10d", "dist_52w_high", "beta_60d", "idio_vol_60d", "rank_growth_90d", "vix_term_3m",
           "credit_spread", "yield_curve_10y_3m", "sma200_rel", "volume_zscore_20d", C.TARGET_GROWTH]
cm = train[new_sel].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(new_sel)), new_sel, rotation=70); ax.set_yticks(range(len(new_sel)), new_sel)
for i in range(len(new_sel)):
    for j in range(len(new_sel)):
        ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im); ax.set_title("Spearman correlation of selected NEW features"); plt.show()

# %% [markdown]
# ### 3.3 Forward return by market regime

# %%
df["vix_bucket"] = pd.cut(df.vix_close, [0, 15, 20, 30, 100], labels=["<15", "15-20", "20-30", ">30"])
regime = df.groupby(["snp500_above_sma200", "vix_bucket"], observed=True)[[C.TARGET_GROWTH, C.TARGET]].agg(["mean", "count"])
regime

# %% [markdown]
# ## 4. Models
# Four model families were tuned on the **validation** period (2019-01 .. 2022-06) for both targets:
# Decision Tree, Random Forest, Logistic Regression (lectures) and **XGBoost** (new). The best configuration
# (by validation ROC-AUC) is re-fitted on train+validation and evaluated once on **test** (2022-07 .. today).

# %%
tuning = pd.concat([pd.read_csv(R / "model_tuning_validation.csv"), pd.read_csv(R / "model_tuning_validation_rel.csv")])
tuning

# %%
preds = P.load_predictions()
summary = pd.concat([pd.read_csv(R / "model_summary_validation.csv"), pd.read_csv(R / "model_summary_test.csv")])
summary.sort_values(["split", "top5_mean_growth_10d"], ascending=[False, False])

# %%
imp = pd.read_csv(R / "feature_importance.csv")
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
for a, (target, g) in zip(ax, imp.groupby("target")):
    g.head(20).set_index("feature").xgb_gain.sort_values().plot.barh(ax=a, title=f"XGBoost gain - {target}")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Decision rules: from probabilities to trades
# Besides "buy the top-k each day", threshold rules turn a probability into a 0/1 decision.
# Below: precision and number of trades vs threshold for the best score on validation — higher thresholds
# give fewer, more confident trades.

# %%
best_strategy = json.loads(P.BEST_STRATEGY_FILE.read_text())
SCORE = best_strategy["score"]
LABEL = C.TARGET_REL if SCORE.endswith("_rel") else C.TARGET
valid, test = preds[preds.split == "validation"].copy(), preds[preds.split == "test"].copy()
rows = []
for thr in np.quantile(valid[SCORE], [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]):
    on = valid[SCORE] >= thr
    rows.append({"threshold": thr, "share_traded": on.mean(), "precision_label": valid.loc[on, LABEL].mean(),
                 "precision_up": valid.loc[on, C.TARGET].mean(), "avg_growth_10d": valid.loc[on, C.TARGET_GROWTH].mean()})
pd.DataFrame(rows)

# %% [markdown]
# ## 6. Vectorized simulation (lecture style, `sim1_` fields)
# $100 invested in every positive prediction on the **test** period, 0.1% fee per side; approximate CAGR
# from the capital needed to fund the 75th-percentile number of open trades.

# %%
for name in ("dt", "rf", "lr", "xgb"):
    for tag in ("", "_rel"):
        col = f"proba_{name}{tag}"
        thr = valid[col].quantile(0.9)              # threshold chosen on validation (top 10%)
        test[f"pred_{name}{tag}_top10pct"] = (test[col] >= thr).astype(int)
        test[f"pred_{name}{tag}_rank1"] = (test.groupby("Date")[col].rank(ascending=False) <= 1).astype(int)
test["pred_all_stocks"] = 1                          # benchmark: buy everything
pred_cols = [c for c in test.columns if c.startswith("pred_")]
vec = vectorized_sim(test, pred_cols)
vec

# %%
print("sim1_ columns added per prediction, e.g.:", [c for c in test.columns if c.startswith("sim1_")][:3])

# %% [markdown]
# ## 7. Exact simulation with capital re-investment
# $10,000, every day 1/10 of equity is invested in the day's picks (10 overlapping 2-week sleeves), exits after
# 10 trading days or by stop loss (gap-downs filled at the open), 0.1% fee per side.
#
# Strategies (S1–S4 use the best validation score, S5 is the best of a 288-combination grid **chosen on
# validation**: threshold × top-k × stop loss × regime filter × sizing):
# * **S1** top-5 per day, equal weight
# * **S2** S1 + 10% stop loss
# * **S3** S1 + market-regime filter (trade only if S&P 500 > SMA200 and VIX < 30)
# * **S4** top-3, probability-weighted sizing, only probabilities above the validation 80th percentile
# * **S5** best on validation

# %%
grid = pd.read_csv(R / "strategy_grid_validation.csv")
print(f"{len(grid)} strategies evaluated on validation; best 10 by Sharpe:")
grid.head(10)[["name", "CAGR", "sharpe", "max_drawdown", "n_trades", "win_rate"]]

# %%
res = pd.read_csv(R / "strategies_test.csv")
res

# %%
curves = pd.read_csv(R / "equity_curves_test.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
curves.plot(ax=ax[0], logy=True, title="TEST: equity of $10,000 (log scale)")
(curves / curves.cummax() - 1).plot(ax=ax[1], title="drawdown", legend=False)
plt.tight_layout(); plt.show()

# %%
roll = curves.pct_change(252).dropna()
roll.plot(title="rolling 1-year return (TEST)"); plt.axhline(0, color="k", lw=0.6); plt.show()
(roll.drop(columns="SPY_buy_and_hold").gt(roll.SPY_buy_and_hold, axis=0).mean()
 .rename("share of 1y windows beating SPY").to_frame())

# %% [markdown]
# ## 8. Deep analysis of the predictions (test period)
# Where do the daily top-5 picks of the best score work, and where do they fail?

# %%
test["rank"] = test.groupby("Date")[SCORE].rank(ascending=False, method="first")
top = test[test["rank"] <= 5].copy()
top["excess_vs_day_avg"] = top[C.TARGET_GROWTH] - top.groupby("Date")[C.TARGET_GROWTH].transform("mean")
day_avg = test.groupby("Date")[C.TARGET_GROWTH].mean().rename("day_avg")
top = top.join(day_avg, on="Date")
top["beats_day_avg"] = top[C.TARGET_GROWTH] > top.day_avg
top["vix_bucket"] = pd.cut(top.vix_close, [0, 15, 20, 30, 100], labels=["<15", "15-20", "20-30", ">30"])

def agg(g):
    return pd.Series({"picks": len(g), "hit_rate_up": g[C.TARGET].mean(), "beats_avg_stock": g.beats_day_avg.mean(),
                      "avg_growth_10d": g[C.TARGET_GROWTH].mean() - 1, "avg_universe_10d": g.day_avg.mean() - 1})

# %%
print("by sector"); top.groupby("sector").apply(agg).sort_values("picks", ascending=False)

# %%
print("by year"); top.groupby("Year").apply(agg)

# %%
print("by calendar month"); top.groupby("Month").apply(agg)

# %%
print("by VIX regime"); top.groupby("vix_bucket", observed=True).apply(agg)

# %%
by_ticker = top.groupby("Ticker").apply(agg).sort_values("picks", ascending=False)
print("most frequently picked tickers"); by_ticker.head(15)

# %%
print("concentration: share of all picks taken by the 10 most-picked tickers:",
      round(by_ticker.picks.head(10).sum() / by_ticker.picks.sum(), 3))
print("worst tickers by average 10d return (>= 20 picks):")
by_ticker[by_ticker.picks >= 20].sort_values("avg_growth_10d").head(10)

# %%
# calibration: do higher probabilities really mean higher returns? (test, deciles of the score)
test["decile"] = pd.qcut(test[SCORE], 10, labels=False, duplicates="drop")
cal = test.groupby("decile").agg(mean_proba=(SCORE, "mean"), label_rate=(LABEL, "mean"),
                                  avg_growth_10d=(C.TARGET_GROWTH, "mean"), avg_volatility=("volatility", "mean"))
cal

# %% [markdown]
# ## 9. Findings, difficulties and next steps
# (see the README "Results" section — numbers above are reproduced there)
