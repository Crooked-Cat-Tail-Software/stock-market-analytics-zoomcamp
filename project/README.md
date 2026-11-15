# 10-Day Stock Selection for US Large Caps — Stock Markets Analytics Zoomcamp 2026 Project

An end-to-end, automated algo-trading research system: it downloads data from five providers, builds one
dataset of **1.1 million rows × 300+ features**, trains and tunes four model families, turns predictions into
trading strategies, simulates them day by day with re-investment, stop losses and a market-regime filter,
compares them with the S&P 500, and prints the trades for the latest close. It can be re-run every day by cron.

### Results at a glance (test period 2022-07 → 2026-11, never used for any choice)

$10,000, 0.1% fee per side, exact day-by-day simulation with re-investment:

| Strategy | CAGR | Sharpe | Max drawdown | Trades | Win rate |
|---|---:|---:|---:|---:|---:|
| S1 top-5 per day, equal weight | 21.9% | 1.29 | -17.9% | 5,390 | 56.0% |
| S2 S1 + 10% stop loss | 19.9% | 1.31 | -17.3% | 5,391 | 55.7% |
| S3 S1 + market-regime filter | 15.9% | 1.29 | **-12.6%** | 4,370 | 56.0% |
| S4 top-3, probability-weighted, above validation 80th pct. | **23.2%** | **1.32** | -17.1% | 2,244 | 57.4% |
| **S5 best of 288 on validation** (RF "beats S&P 500", top-5, 10% stop) | 19.9% | 1.31 | -17.3% | 5,391 | 55.7% |
| *SPY buy & hold* | *18.6%* | *1.15* | *-18.8%* | | |
| *Equal-weight buy & hold of the same 190 stocks* | *29.9%* | *1.73* | *-17.9%* | | |

**Honest reading**

* Every strategy beats SPY on Sharpe ratio and most on CAGR (S4: 23.2% vs 18.6%), and S1/S4 beat SPY in
  66-68% of rolling 1-year windows.
* **But the models have almost no out-of-sample edge.** Test ROC-AUC is 0.50-0.54 for every model and both
  targets; the chosen model's daily top-5 picks earn +1.04% per 10 days vs +1.08% for the average stock.
  The strategies' returns come mostly from **being invested in this universe**, not from the stock selection.
* **The universe has survivorship bias:** it is *today's* 190 largest US companies, i.e. the winners of the last
  decades. Simply holding all of them equally returned 29.9%/year — more than any strategy. Any backtest on this
  list (including the lectures') overstates what was achievable in real time.
* The regime filter (S3) gives up return but cuts the max drawdown by a third (-12.6% vs -18.8% SPY).
* See [notebooks/01_research.ipynb](notebooks/01_research.ipynb) §8 for the deep analysis (by sector, year, month,
  VIX regime, ticker, calibration) and §9 / [below](#5-findings-difficulties-next-steps) for what to fix next.

---

## 1. Problem

**Can a model pick, every day, the large US stocks that will do best over the next two weeks?**

The lectures predict whether a stock is up 30 days later. This project changes the problem definition:

| | Lectures (2026) | This project |
|---|---|---|
| Horizon | 30 days | **10 trading days** (≈ 2 weeks): faster capital turnover, more independent bets per year |
| Target | stock goes up (`growth_future_30d > 1`) | two targets: **(a)** stock goes up, **(b)** stock **beats the S&P 500** over the same 10 days |
| Universe | 190 US stocks > $50B | same 190 stocks (comparable) |
| Data | yfinance + FRED | + **CBOE** volatility indexes + **Wikipedia** GICS sectors (web scraping) |
| Validation | random time split | **train / validation / test by date with a 10-day embargo**, strategy chosen on validation only |

Why two targets? Target (a) is mostly a bet on the direction of the whole market — whether *any* stock is up
in 10 days depends more on the S&P 500 than on the stock — and the market's direction is very hard to predict.
Target (b) removes the market from the label, so the model has to learn *stock selection*, which is exactly what
a "buy the top-k stocks each day" strategy needs. Both are trained and compared; the strategy uses whichever
works better on validation.

**End goal:** a daily list of at most *k* stocks to buy at the close and hold for 10 trading days (with an
optional stop loss), whose simulated performance after fees is compared with buy-and-hold SPY.

## 2. Data sources

| Provider | Series | Notes |
|---|---|---|
| **yfinance** | daily OHLCV of 190 US stocks (course list, `trading/config.py`) since 1998 | auto-adjusted for splits & dividends |
| **yfinance** | `^GSPC ^DJI ^IXIC ^RUT ^VIX GC=F CL=F BTC-USD DX-Y.NYB`, `SPY` | market features; SPY = total-return benchmark |
| **FRED** | `GDPPOT CPILFESL FEDFUNDS UNRATE DGS3MO DGS1 DGS5 DGS10 BAA10Y` | public CSV endpoint (no API key) |
| **CBOE** *(new)* | `VIX9D VIX3M VVIX SKEW` daily history from cdn.cboe.com | volatility term structure, tail-risk pricing |
| **Wikipedia** *(new, scraping)* | S&P 500 constituents table: GICS sector & sub-industry | yfinance `.info` fallback for the 7 non-members |

* **1,137,083** raw daily stock rows (190 tickers, 1998-11 → today); **1,101,330** modelling rows from 2000.
* Data is **not** committed (`data/` is git-ignored). It is fully re-downloadable with one command
  (`python main.py run --mode fetch`, ≈ 5 min the first time, seconds afterwards).

### Features (300 numeric + 28 dummies)

* **Lecture features:** growth over 1/3/7/30/90/365 days, SMA10/SMA20 crossover, high-low range, 30d volatility,
  ln(volume), all TA-Lib momentum / volume / volatility / cycle indicators, 61 candlestick patterns, growth of
  indexes / gold / oil / Bitcoin, FRED macro.
* **~56 NEW features** (`NEW_FEATURES` in `trading/transform.py`):
  * *trend & position*: price vs SMA50 / SMA200, golden cross, distance from 52-week high / low, 12-1 momentum, 10/20d growth
  * *volume & intraday*: volume z-score, ln dollar volume, overnight gap, close location in the day's range, OBV change
  * *risk*: downside volatility, 60d skewness, max daily return (lottery effect), 60d **beta** and **idiosyncratic volatility** vs the S&P 500, listing age
  * *relative*: excess return vs S&P 500 over 5/10/20/60d, **cross-sectional ranks** of momentum / volatility / RSI / volume per day, **sector-relative** 30d growth
  * *market regime*: S&P 500 above SMA200, S&P drawdown, 20d market volatility, VIX level / change / >30 flag, Nasdaq / Russell / US-dollar growth
  * *CBOE*: VIX/VIX3M (term-structure inversion = stress), VIX9D/VIX, VIX3M, VVIX, SKEW and its change
  * *macro derived*: yield-curve slopes 10y-3m and 10y-1y, real 10y rate, Baa credit spread and its 20d change, unemployment and its 3-month change
  * *categorical*: GICS sector dummies
* **Transformations that matter:** TA-Lib indicators measured in dollars (MACD, ATR, APO, MOM, …) are divided
  by the close so stocks of different price are comparable; monthly/quarterly FRED data is joined on the date it
  was **published** (no look-ahead); market data is joined "as of" each trading day.

## 3. Method

```
trading/
  config.py      universe, dates, horizon, fees, paths
  data_repo.py   Step 1  download (incremental) / persist / load raw data
  transform.py   Step 2  features + targets -> one dataframe + feature sets (TO_PREDICT / NUMERIC / CATEGORICAL)
  train.py       Step 3  dummies, temporal split with embargo, model tuning, predictions, manual rules
  simulate.py    Step 4  vectorized sim (sim1_ fields), exact portfolio simulator, metrics, benchmarks
  pipeline.py    glue used by main.py and notebooks
main.py          CLI: `research` and `run` (daily)
run_daily.sh     cron wrapper
notebooks/01_research.ipynb     EDA, model comparison, decision rules, simulations, deep analysis
notebooks/02_end_to_end.ipynb   one notebook running the whole workflow and showing the latest trades
reports/         small CSV results produced by the research run
```

**Split.** train 2000-01 → 2018-12 · validation 2019-01 → 2022-06 · test 2022-07 → last date with a known
10-day outcome · live = the last 10 days (no label yet, used for today's trades). Rows whose 10-day target
window crosses into the next split are dropped (embargo), otherwise the label would leak future prices.

**Models** (both targets): Decision Tree, Random Forest, Logistic Regression (lectures) and XGBoost (new),
each tuned with a small grid on validation (XGBoost also with early stopping). Every model is fitted twice:
on *train* to produce honest validation probabilities (used to choose thresholds and strategy parameters), and
on *train + validation* for the test and live predictions. Two hand-made rules (momentum, buy-the-dip) are
kept as baselines.

**Simulations.**
1. *Vectorized* (lecture style): $100 on every positive prediction, `sim1_gross_rev_*`, `sim1_fees_*`,
   `sim1_net_rev_*` per prediction column, approximate CAGR from the capital needed.
2. *Exact, day by day*: $10,000; every day 1/10 of current equity is invested in that day's picks (10 overlapping
   two-week sleeves → capital fully used and re-invested); exit after 10 trading days at the close or by stop loss
   (a gap through the stop is filled at the open); 0.1% fee per side. Options: top-k limit on new positions,
   probability threshold, probability-weighted sizing, market-regime filter (trade only if S&P 500 > SMA200 and
   VIX < 30).
3. **288 strategy combinations** are evaluated on validation; the best by Sharpe ratio is frozen and reported once
   on test against **SPY buy-and-hold** and an **equal-weight buy-and-hold of the same 190 stocks**, with CAGR,
   Sharpe, Sortino, max drawdown, Calmar and rolling 1-year returns.

## 4. How to reproduce

Prerequisites: Python 3.12, macOS or Linux. On macOS XGBoost needs OpenMP: `brew install libomp`.
TA-Lib ≥ 0.6 ships wheels with the C library included, so no separate TA-Lib install is needed.

```bash
cd project
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1) research: download everything, build the dataset, tune models, choose the strategy, test report (~30-40 min)
.venv/bin/python main.py research --mode fetch

# 2) daily run: fetch only the new days, rebuild features, score with saved models, print today's trades (~3 min)
.venv/bin/python main.py run --mode fetch

# offline variants (no network): use what is already in data/
.venv/bin/python main.py run --mode local
.venv/bin/python main.py run --mode local --skip-transform     # fastest, reuses data/processed/dataset.parquet
```

Notebooks: `.venv/bin/jupyter lab notebooks/` — `02_end_to_end.ipynb` runs the full workflow (set `MODE`),
`01_research.ipynb` documents the research.

**Automation with cron** (weekdays after the US close):

```bash
crontab -e
# add:
30 22 * * 1-5 /full/path/to/project/run_daily.sh
```

Every run appends the picks to `reports/latest_trades.csv` and logs to `logs/`.

**Incremental loading.** `--mode fetch` reads the stored raw files and asks yfinance only for the days after
the last stored date of each ticker. Because prices are dividend-adjusted, a new dividend or split rescales the
past; the overlap is compared and only that ticker is re-downloaded in full. Data lives in `data/` (outside git).

## 5. Findings, difficulties, next steps

**Models (validation → test).**

| Score | Target | Valid AUC | Test AUC | Test top-5 avg 10d growth |
|---|---|---:|---:|---:|
| Logistic regression | goes up | **0.557** | 0.507 | +0.90% |
| XGBoost | goes up | 0.486 | **0.541** | +0.47% |
| Random forest | beats S&P 500 | 0.516 | 0.508 | +1.04% |
| XGBoost | beats S&P 500 | 0.526 | 0.505 | +0.98% |
| Decision tree | beats S&P 500 | 0.512 | 0.496 | **+1.97%** |
| *all stocks (average)* | | | | *+1.08%* |

* **10-day returns are mostly noise.** The strongest single features have Spearman |ρ| of only a few percent
  with the forward return (EDA §3.2). No model reached a test AUC above 0.55.
* **Deeper trees got *worse* on validation** (DT AUC 0.49 → 0.46 from depth 3 to 9), and XGBoost early-stopped
  after 4-223 trees: the models first learn *which period* they are in from slow-moving macro levels
  (10y yield, unemployment, VIX term structure — the top features by gain), which does not transfer to new years.
* **The relative target helps the ranking, not the AUC.** On validation every "beats S&P 500" model put better
  stocks in its top 5 (+1.43% to +1.59% per 10 days) than every "goes up" model (+0.90% to +1.39%), which is why the
  strategy grid chose a relative-target score.
* **Where the picks work / fail (test, top-5 of the chosen score):** best in Information Technology (+2.4% vs
  +1.1% sector average) and Energy; worst in Health Care (+0.1% vs +1.4%) and Materials. When VIX > 30 the picks
  hit 75% (n=155). Repeatedly picked losers: CVS, NFLX, CMCSA, NEM. 23% of all picks go to 10 tickers.
* **Observation, not used:** the decision-tree "beats S&P 500" score with a top-10% threshold (fixed on
  validation) averaged +2.97% per 10 days on test (precision 66%, vectorized approx. CAGR 38%). It was found
  *after* looking at test, so selecting it now would be data snooping; it is a candidate for the next research round.

**Difficulties.** yfinance rate limits (retry + sleep + incremental loading); FRED now publishes only 3 years of
the ICE high-yield spread, replaced with Moody's Baa spread (`BAA10Y`); `pandas_datareader` breaks on Python 3.12
(replaced by FRED's CSV endpoint); the Fama-French data library was not reachable from the build environment, so
CBOE and Wikipedia became the new sources; memory (1.1M × 330 float32) — features are float32 and matrices are
built column by column (peak 8.3 GB RAM).

**Next steps.** (1) Point-in-time universe (historical S&P 500 membership from Wikipedia's "changes" table) to
remove survivorship bias. (2) Remove slow macro *levels* from the inputs, keep only changes / cross-sectional
ranks. (3) Rank-based objective (XGBoost `rank:pairwise` per day) instead of classification. (4) Walk-forward
re-training every year instead of one train/validation/test split. (5) Long-short version (top-k vs bottom-k).

## 6. Rubric checklist

| Criterion | Where |
|---|---|
| **Problem description** · context & end goal · new problem (10-day horizon, relative target) · steps, findings, reproduction | §1, §3, §4, §5 |
| **Lecture data & features** · 56 new features · new sources (CBOE, Wikipedia scraping) · 1.1M records | §2, `trading/transform.py` (`NEW_FEATURES`) |
| **One dataframe with TO_PREDICT / NUMERIC / DUMMIES** · 5+ new transformation features · EDA with target description & correlations | `data/processed/feature_sets.json`, notebook §2-3 |
| **Lecture model** · DT + RF + LR · custom decision rules (thresholds, top-k, manual momentum/dip) · hyper-parameter tuning · XGBoost | `trading/train.py`, `reports/model_tuning_validation*.csv`, notebook §4-5 |
| **Vectorized sim** · 5 strategies with `sim1_` fields · exact sim with re-investment · CAGR / Sharpe / max DD / rolling returns vs SPY · stop loss, probability sizing · regime filter + top-k limit · beats SPY on CAGR (see caveats) · deep analysis by ticker / sector / year / month / VIX | `trading/simulate.py`, `reports/strategies_test.csv`, notebook §6-8 |
| **Scripts + one end-to-end notebook** · `requirements.txt` + install steps · cron (`run_daily.sh`) · two modes (`--mode local` / `--mode fetch`) · incremental loading, data outside git | `main.py`, `notebooks/02_end_to_end.ipynb`, §4 |
| Bonus: modular, commented code | `trading/` |
