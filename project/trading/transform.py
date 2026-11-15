"""Step 2 - Transform: turn the raw downloads into ONE dataframe ready for ML.

Output: data/processed/dataset.parquet, one row per (Date, Ticker), plus a JSON file with the
feature sets used later by the models:

  TO_PREDICT  - targets / future information (never used as a model input)
  NUMERIC     - numeric model inputs (lecture features + NEW features)
  CATEGORICAL - columns turned into DUMMIES at training time (Month, Weekday, sector, ...)

Look-ahead protection
  * monthly / quarterly FRED releases are shifted to the date they were actually *published*
    (e.g. October CPI becomes known in mid-November) - the lecture code used them too early.
  * everything else uses information available at the close of day t; the target is the
    return from the close of day t to the close of day t+10.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import talib
from talib import abstract
from tqdm import tqdm

from . import config as C
from .data_repo import DataRepository

H = C.HORIZON
GROWTH_WINDOWS = [1, 3, 7, 30, 90, 365]   # same windows as the lectures

# TA-Lib groups used in the lectures (Overlap studies / math functions are price levels -> skipped)
TALIB_GROUPS = ["Momentum Indicators", "Volume Indicators", "Volatility Indicators", "Cycle Indicators"]
# indicators expressed in $ (price units) -> divided by Close so they are comparable across stocks
PRICE_UNIT = {"apo", "macd_macd", "macd_macdsignal", "macd_macdhist", "macdext_macd", "macdext_macdsignal",
              "macdext_macdhist", "macdfix_macd", "macdfix_macdsignal", "macdfix_macdhist", "mom",
              "minus_dm", "plus_dm", "atr", "trange", "ht_phasor_inphase", "ht_phasor_quadrature"}
# cumulative volume indicators: their level is arbitrary -> replaced by a normalised version below
VOLUME_LEVEL = {"ad", "obv", "adosc"}


# ======================================================================================
# per-ticker features
# ======================================================================================
def _talib_features(df: pd.DataFrame) -> pd.DataFrame:
    """All TA-Lib indicators from the lecture groups + every candlestick pattern."""
    # TA-Lib only accepts float64 arrays (Volume comes as int)
    inputs = {k: df[col].to_numpy(dtype=np.float64) for k, col in
              [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]}
    out = {}
    groups = talib.get_function_groups()
    for group in TALIB_GROUPS:
        for name in groups[group]:
            if name in ("MAVP",):  # needs a custom 'periods' input
                continue
            fn = abstract.Function(name)
            res = fn(inputs)
            names = fn.output_names
            res = res if isinstance(res, list) else [res]
            for out_name, values in zip(names, res):
                col = name.lower() if len(names) == 1 else f"{name.lower()}_{out_name.lower()}"
                out[col] = values
    for name in groups["Pattern Recognition"]:
        out[name.lower()] = abstract.Function(name)(inputs).astype(np.int16)  # -100 / 0 / 100

    ta = pd.DataFrame(out, index=df.index)
    for col in PRICE_UNIT & set(ta.columns):
        ta[col] = ta[col] / df.Close          # $ -> fraction of price
    avg_vol20 = df.Volume.rolling(20).mean().replace(0, np.nan)
    ta["adosc"] = ta["adosc"] / avg_vol20     # A/D oscillator in units of average daily volume
    ta["obv_change_10d"] = ta["obv"].diff(10) / (avg_vol20 * 10)
    return ta.drop(columns=["ad", "obv"])


def _ticker_features(df: pd.DataFrame, snp_ret: pd.Series) -> pd.DataFrame:
    """Lecture features, NEW per-ticker features and the targets for one ticker (sorted by Date)."""
    df = df.sort_values("Date").reset_index(drop=True)
    close, high, low, vol = df.Close, df.High, df.Low, df.Volume.astype(float)
    f = pd.DataFrame(index=df.index)

    # ---------------- lecture (Module 2) features
    for w in GROWTH_WINDOWS:
        f[f"growth_{w}d"] = close / close.shift(w)
    sma10, sma20 = close.rolling(10).mean(), close.rolling(20).mean()
    f["growing_moving_average"] = (sma10 > sma20).astype(int)
    f["sma10_rel"] = sma10 / close
    f["sma20_rel"] = sma20 / close
    f["high_minus_low_relative"] = (high - low) / close
    daily_ret = close.pct_change()
    f["volatility"] = daily_ret.rolling(30).std() * np.sqrt(252)
    f["ln_volume"] = np.log(vol.where(vol > 0))

    # ---------------- NEW per-ticker features
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    f["sma50_rel"] = close / sma50                                  # trend vs 50d average
    f["sma200_rel"] = close / sma200                                # trend vs 200d average
    f["golden_cross"] = (sma50 > sma200).astype(int)                # 50d above 200d average
    f["dist_52w_high"] = close / high.rolling(252, min_periods=120).max()  # 1.0 = at the yearly high
    f["dist_52w_low"] = close / low.rolling(252, min_periods=120).min()
    f["growth_10d"] = close / close.shift(10)                       # past return on the same horizon as the target
    f["growth_20d"] = close / close.shift(20)
    f["mom_12_1"] = close.shift(21) / close.shift(252)              # classic 12-1 month momentum
    ln_v = np.log(vol.where(vol > 0))
    f["volume_zscore_20d"] = (ln_v - ln_v.rolling(20).mean()) / ln_v.rolling(20).std()
    f["ln_dollar_volume_20d"] = np.log((close * vol).rolling(20).mean().where(lambda s: s > 0))
    f["gap_open"] = df.Open / close.shift(1)                        # overnight gap
    f["close_location"] = ((close - low) / (high - low)).where(high > low)  # where in the day's range we closed
    f["downside_vol_20d"] = daily_ret.clip(upper=0).rolling(20).std() * np.sqrt(252)
    f["ret_skew_60d"] = daily_ret.rolling(60).skew()
    f["max_ret_20d"] = daily_ret.rolling(20).max()                  # "lottery" effect
    # beta & idiosyncratic volatility vs the S&P 500 over 60 trading days
    mkt = snp_ret.reindex(df.Date).to_numpy()
    mkt = pd.Series(mkt, index=df.index)
    cov = daily_ret.rolling(60).cov(mkt)
    var = mkt.rolling(60).var()
    f["beta_60d"] = cov / var
    f["idio_vol_60d"] = (daily_ret - f["beta_60d"] * mkt).rolling(60).std() * np.sqrt(252)
    mkt_index = (1 + mkt.fillna(0)).cumprod()
    for w in (5, 10, 20, 60):                                       # return in excess of the market
        f[f"excess_ret_{w}d"] = close / close.shift(w) - mkt_index / mkt_index.shift(w)
    # young listings behave differently; capped at 3y so it is not a hidden "calendar time" feature
    f["days_since_ipo"] = np.minimum(np.arange(len(df)), 756)

    # ---------------- TA-Lib (lecture Module 2)
    ta = _talib_features(df)

    # ---------------- targets / future info (TO_PREDICT, never inputs)
    t = pd.DataFrame(index=df.index)
    t[C.TARGET_GROWTH] = close.shift(-H) / close
    t[C.TARGET] = np.where(t[C.TARGET_GROWTH].isna(), np.nan, (t[C.TARGET_GROWTH] > 1).astype(float))
    # max high / min low over the next H days (relative to today's close) -> stop-loss / take-profit sims
    t[f"future_max_high_{H}d"] = high[::-1].rolling(H).max()[::-1].shift(-1) / close
    t[f"future_min_low_{H}d"] = low[::-1].rolling(H).min()[::-1].shift(-1) / close
    t["growth_future_1d"] = close.shift(-1) / close

    base = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    return pd.concat([base, f, ta, t], axis=1)


# ======================================================================================
# market-wide features (indexes, CBOE, macro)
# ======================================================================================
def _market_features(market_df: pd.DataFrame, cboe_df: pd.DataFrame) -> pd.DataFrame:
    sym_to_prefix = {v: k for k, v in C.INDEXES.items()}
    wide = market_df.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    out = {}
    for sym, prefix in sym_to_prefix.items():
        if sym not in wide:
            continue
        s = wide[sym].dropna()
        if prefix == "vix":
            out["vix_close"] = s
            out["vix_change_5d"] = s / s.shift(5)
            out["vix_above_30"] = (s > 30).astype(int)
            continue
        for w in GROWTH_WINDOWS:                                    # lecture-style index growth
            out[f"growth_{prefix}_{w}d"] = s / s.shift(w)
        if prefix == "snp500":                                      # NEW market-regime features
            out["snp500_above_sma200"] = (s > s.rolling(200).mean()).astype(int)
            out["snp500_drawdown_252d"] = s / s.rolling(252).max()
            out["snp500_vol_20d"] = s.pct_change().rolling(20).std() * np.sqrt(252)
    mkt = pd.DataFrame(out).sort_index()

    # NEW (CBOE source): volatility term structure & tail risk
    cb = cboe_df.set_index("Date").sort_index()
    vix = wide.get("^VIX")
    cboe = pd.DataFrame(index=cb.index)
    if vix is not None:
        cboe["vix_term_3m"] = (vix.reindex(cb.index) / cb["vix3m"])   # >1 = inverted curve = stress
        cboe["vix9d_term"] = (cb["vix9d"] / vix.reindex(cb.index))   # short-term fear vs 1 month
    cboe["vix3m"] = cb["vix3m"]
    cboe["vvix"] = cb["vvix"]
    cboe["skew_index"] = cb["skew"]
    cboe["skew_change_20d"] = cb["skew"] - cb["skew"].rolling(20).mean()
    return mkt, cboe


def _macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """FRED series aligned on the date they became public (no look-ahead)."""
    m = macro_df.set_index("Date").sort_index()
    pieces = []

    def published(series: pd.Series, lag: pd.DateOffset) -> pd.Series:
        s = series.dropna()
        s.index = s.index + lag
        return s

    month_lag = pd.DateOffset(months=1, days=15)       # monthly data: released mid next month
    quarter_lag = pd.DateOffset(months=3, days=30)
    cpi = m["CPILFESL"].dropna()
    pieces.append(published(cpi / cpi.shift(12) - 1, month_lag).rename("cpi_core_yoy"))
    pieces.append(published(cpi / cpi.shift(1) - 1, month_lag).rename("cpi_core_mom"))
    gdp = m["GDPPOT"].dropna()
    pieces.append(published(gdp / gdp.shift(4) - 1, quarter_lag).rename("gdppot_us_yoy"))
    pieces.append(published(gdp / gdp.shift(1) - 1, quarter_lag).rename("gdppot_us_qoq"))
    pieces.append(published(m["FEDFUNDS"], month_lag).rename("FEDFUNDS"))
    un = m["UNRATE"].dropna()
    pieces.append(published(un, month_lag).rename("unrate"))
    pieces.append(published(un - un.shift(3), month_lag).rename("unrate_change_3m"))   # NEW
    for s in ["DGS3MO", "DGS1", "DGS5", "DGS10", "BAA10Y"]:                      # daily, next-day
        pieces.append(published(m[s], pd.DateOffset(days=1)).rename(s))
    out = pd.concat(pieces, axis=1).sort_index().ffill()
    out = out.rename(columns={"BAA10Y": "credit_spread"})
    # NEW derived macro features
    out["yield_curve_10y_3m"] = out["DGS10"] - out["DGS3MO"]
    out["yield_curve_10y_1y"] = out["DGS10"] - out["DGS1"]
    out["real_rate_10y"] = out["DGS10"] - 100 * out["cpi_core_yoy"]
    out["credit_spread_change_20d"] = out["credit_spread"] - out["credit_spread"].shift(20)
    return out


def _asof_join(df: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest known value of every market-wide column to each stock row."""
    right = right.sort_index().ffill()
    right = right[~right.index.duplicated(keep="last")]
    aligned = right.reindex(right.index.union(df.Date.unique())).ffill().reindex(df.Date.unique())
    return df.merge(aligned, left_on="Date", right_index=True, how="left")


# ======================================================================================
class TransformData:
    def __init__(self, repo: DataRepository):
        self.repo = repo
        self.df: pd.DataFrame | None = None
        self.feature_sets: dict[str, list[str]] = {}

    def transform(self) -> pd.DataFrame:
        repo = self.repo
        snp = repo.market_df[repo.market_df.Ticker == C.INDEXES["snp500"]].set_index("Date").Close
        snp_ret = snp.sort_index().pct_change()

        parts = []
        for ticker, g in tqdm(repo.stocks_df.groupby("Ticker"), desc="features per ticker"):
            if len(g) < 300:   # not enough history for 1-year windows
                continue
            parts.append(_ticker_features(g, snp_ret))
        df = pd.concat(parts, ignore_index=True)

        mkt, cboe = _market_features(repo.market_df, repo.cboe_df)
        df = _asof_join(df, mkt)
        df = _asof_join(df, cboe)
        df = _asof_join(df, _macro_features(repo.macro_df))

        # second (relative) target: does the stock beat the S&P 500 over the same 10 days?
        # Removes the market direction from the label, so the model has to learn stock *selection*.
        snp_sorted = snp.sort_index()
        snp_future = (snp_sorted.shift(-H) / snp_sorted).rename(f"snp500_growth_future_{H}d")
        df = df.merge(snp_future, left_on="Date", right_index=True, how="left")
        df[C.TARGET_REL] = np.where(df[C.TARGET_GROWTH].isna() | df[snp_future.name].isna(), np.nan,
                                    (df[C.TARGET_GROWTH] > df[snp_future.name]).astype(float))

        # calendar + sector (lecture categorical vars + NEW sector from Wikipedia)
        df["Year"] = df.Date.dt.year
        df["Month"] = df.Date.dt.month
        df["Weekday"] = df.Date.dt.weekday
        df["month_wom"] = df.Month.astype(str) + "_w" + ((df.Date.dt.day - 1) // 7 + 1).astype(str)
        df = df.merge(repo.sectors_df[["Ticker", "sector", "sub_industry"]], on="Ticker", how="left")
        df["sector"] = df["sector"].fillna("Unknown")

        df = df[df.Date >= C.START_DATE].reset_index(drop=True)
        df = self._cross_sectional_features(df)
        self.df = self._downcast(df)
        self.feature_sets = self._define_feature_sets(self.df)
        return self.df

    @staticmethod
    def _cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
        """NEW: rank of each stock vs all other stocks on the same day (0..1), and vs its sector."""
        by_date = df.groupby("Date")
        for col in ["growth_30d", "growth_90d", "excess_ret_10d", "volatility", "rsi", "volume_zscore_20d",
                    "dist_52w_high"]:
            df[f"rank_{col}"] = by_date[col].rank(pct=True)
        sector_med = df.groupby(["Date", "sector"])["growth_30d"].transform("median")
        df["sector_rel_growth_30d"] = df["growth_30d"] / sector_med
        df["sector_mean_growth_30d"] = sector_med
        df["n_stocks_today"] = by_date["Ticker"].transform("count")
        return df

    @staticmethod
    def _downcast(df: pd.DataFrame) -> pd.DataFrame:
        floats = df.select_dtypes("float64").columns
        df[floats] = df[floats].astype("float32")
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def _define_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
        to_predict = [c for c in df.columns if "future" in c]
        categorical = ["Month", "Weekday", "month_wom", "sector"]
        not_features = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "Year", "sub_industry"}
        numeric = [c for c in df.columns if c not in set(to_predict) | set(categorical) | not_features]
        patterns = [c for c in numeric if c.startswith("cdl")]
        return {"TO_PREDICT": to_predict, "CATEGORICAL": categorical, "NUMERIC": numeric,
                "PATTERNS": patterns, "NEW_FEATURES": NEW_FEATURES}

    def persist(self, out_dir=C.PROCESSED_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(out_dir / "dataset.parquet", compression="brotli", index=False)
        (out_dir / "feature_sets.json").write_text(json.dumps(self.feature_sets, indent=1))
        print(f"dataset {self.df.shape} saved to {out_dir}")

    def load(self, out_dir=C.PROCESSED_DIR) -> pd.DataFrame:
        self.df = pd.read_parquet(out_dir / "dataset.parquet")
        self.feature_sets = json.loads((out_dir / "feature_sets.json").read_text())
        return self.df


# documented list of the features that are NOT in the lectures (for the README / rubric)
NEW_FEATURES = [
    # per-ticker trend / position
    "sma50_rel", "sma200_rel", "golden_cross", "dist_52w_high", "dist_52w_low", "growth_10d", "growth_20d",
    "mom_12_1",
    # volume / liquidity / intraday
    "volume_zscore_20d", "ln_dollar_volume_20d", "gap_open", "close_location", "obv_change_10d",
    # risk
    "downside_vol_20d", "ret_skew_60d", "max_ret_20d", "beta_60d", "idio_vol_60d", "days_since_ipo",
    # relative to the market / peers
    "excess_ret_5d", "excess_ret_10d", "excess_ret_20d", "excess_ret_60d",
    "rank_growth_30d", "rank_growth_90d", "rank_excess_ret_10d", "rank_volatility", "rank_rsi",
    "rank_volume_zscore_20d", "rank_dist_52w_high", "sector_rel_growth_30d", "sector_mean_growth_30d",
    "n_stocks_today",
    # market regime
    "snp500_above_sma200", "snp500_drawdown_252d", "snp500_vol_20d", "vix_close", "vix_change_5d",
    "vix_above_30", "growth_nasdaq_*", "growth_russell2000_*", "growth_usd_index_*",
    # CBOE (new source)
    "vix_term_3m", "vix9d_term", "vix3m", "vvix", "skew_index", "skew_change_20d",
    # macro derived
    "unrate", "unrate_change_3m", "yield_curve_10y_3m", "yield_curve_10y_1y", "real_rate_10y", "credit_spread",
    "credit_spread_change_20d",
    # categorical (Wikipedia scraping)
    "sector (dummies)",
]
