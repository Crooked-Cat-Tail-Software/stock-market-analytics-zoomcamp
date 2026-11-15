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
# # End-to-end run: download → transform → predict → simulate → latest trades
#
# This notebook runs the **whole automated workflow** with the project modules in `trading/`
# (the same code that `python main.py run` and the cron job use).
#
# * `MODE = "fetch"` downloads only the missing days from the APIs (incremental) and rebuilds features.
# * `MODE = "local"` uses the data already stored in `data/raw` — no network needed.
#
# The models and the chosen strategy come from the research run (`python main.py research`,
# see `01_research.ipynb`). Set `RETRAIN = True` to redo that research here (~30+ min).

# %%
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
PROJECT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(PROJECT); sys.path.insert(0, str(PROJECT))

import json
import pandas as pd
import matplotlib.pyplot as plt
from trading import config as C, pipeline as P

MODE = "local"      # "fetch" | "local"
RETRAIN = False

# %% [markdown]
# ## 1. Get the data (5 providers: yfinance, FRED, CBOE, Wikipedia + yfinance sectors)

# %%
repo = P.get_data(MODE)
print(f"stocks: {len(repo.stocks_df):,} rows, {repo.stocks_df.Ticker.nunique()} tickers, "
      f"{repo.stocks_df.Date.min().date()} .. {repo.stocks_df.Date.max().date()}")
print("market symbols:", sorted(repo.market_df.Ticker.unique()))
print("FRED series:", [c for c in repo.macro_df.columns if c != "Date"])
print("CBOE series:", [c for c in repo.cboe_df.columns if c != "Date"])

# %% [markdown]
# ## 2. Transform into one dataframe with feature sets

# %%
t = P.build_dataset(repo)
print(t.df.shape)
print({k: len(v) for k, v in t.feature_sets.items()})

# %% [markdown]
# ## 3. Predict with the saved models (or retrain)

# %%
if RETRAIN:
    preds = P.research(t)
else:
    preds = P.score_with_saved_models(t)
preds.split.value_counts()

# %% [markdown]
# ## 4. Simulate the strategies on the test period

# %%
strategy = json.loads(P.BEST_STRATEGY_FILE.read_text())
print("strategy chosen on validation:", strategy)
from trading.simulate import Strategy
s_best = Strategy(**strategy)
named = [Strategy("S1_top5_equal", strategy["score"], top_k=5), s_best]
res, curves, trades = P.test_report(preds, repo.market_df, named)
res[["name", "CAGR", "sharpe", "max_drawdown", "n_trades", "win_rate"]]

# %%
ax = curves.plot(figsize=(11, 5), logy=True, title="Test period: equity of $10,000 (log scale)")
ax.set_ylabel("$"); plt.show()

# %% [markdown]
# ## 5. Latest trades (what to buy at the last available close)

# %%
picks = P.latest_trades(preds, strategy)
print("last date:", preds.Date.max().date(), "| market regime OK:", picks.attrs["regime_ok"])
picks

# %%
print("last 10 closed trades of the best strategy in the test period:")
trades[strategy["name"]].tail(10)
