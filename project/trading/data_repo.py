"""Step 1 - Data repository: download raw data from 5 providers and store it on disk.

Sources
  * yfinance  - daily OHLCV for 190 US stocks + market indexes/commodities/crypto + SPY benchmark
  * FRED      - macro series (rates, inflation, GDP, unemployment, credit spread)
  * CBOE      - volatility indexes VIX9D, VIX3M, VVIX, SKEW                       (new source)
  * Wikipedia - GICS sector / sub-industry of S&P 500 members (web scraping)      (new source)

Two run modes are supported by the caller (main.py):
  * fetch: download from the APIs. Price data is loaded *incrementally*: for every ticker only
           the days after the last stored date are requested and appended.
  * local: just read the parquet files already stored in data/raw (no network at all).

All files are kept in data/raw/ which is git-ignored (data must not be committed to GitHub).
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

from . import config as C

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]


def _to_naive_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """yfinance returns tz-aware timestamps (exchange time). Keep only the calendar date."""
    index = pd.to_datetime(index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def download_history(symbol: str, start: str | pd.Timestamp, retries: int = 3) -> pd.DataFrame:
    """Daily OHLCV for one symbol (auto-adjusted for splits and dividends)."""
    for attempt in range(retries):
        try:
            hist = yf.Ticker(symbol).history(start=pd.Timestamp(start), interval="1d", auto_adjust=True)
            if hist.empty:
                return pd.DataFrame()
            hist.index = _to_naive_dates(hist.index)
            hist = hist[~hist.index.duplicated(keep="last")]
            out = hist.reindex(columns=PRICE_COLS).rename_axis("Date").reset_index()
            out["Ticker"] = symbol
            return out
        except Exception as exc:  # network hiccup / rate limit: wait and retry
            wait = 5 * (attempt + 1)
            print(f"  {symbol}: {exc!r}, retrying in {wait}s")
            time.sleep(wait)
    return pd.DataFrame()


class DataRepository:
    """Holds the raw dataframes and knows how to fetch / persist / load them."""

    def __init__(self, tickers: list[str] | None = None):
        self.tickers = tickers or C.US_STOCKS
        self.market_symbols = list(C.INDEXES.values()) + [C.BENCHMARK]
        self.stocks_df: pd.DataFrame | None = None   # long: Date, Ticker, OHLCV
        self.market_df: pd.DataFrame | None = None   # long: same layout, index/commodity symbols
        self.macro_df: pd.DataFrame | None = None    # wide: Date x FRED series (as published)
        self.cboe_df: pd.DataFrame | None = None     # wide: Date x CBOE indexes
        self.sectors_df: pd.DataFrame | None = None  # Ticker -> sector, sub_industry

    # ------------------------------------------------------------------ price data (incremental)
    @staticmethod
    def _incremental_update(existing: pd.DataFrame | None, symbols: list[str], desc: str) -> pd.DataFrame:
        """Append only the missing days per symbol.

        Prices are dividend/split-adjusted, so a new dividend or split rescales the whole past
        series. We detect that by comparing the overlapping days and re-download the full history
        for that one symbol when the stored and fresh closes disagree.
        """
        full_start = pd.Timestamp(C.START_DATE) - pd.Timedelta(days=C.FETCH_BUFFER_DAYS)
        parts, n_full, n_inc = [], 0, 0
        bar = tqdm(symbols, desc=desc)
        for sym in bar:
            bar.set_postfix_str(sym)
            old = existing[existing.Ticker == sym] if existing is not None else pd.DataFrame()
            if old.empty:
                new = download_history(sym, full_start)
                n_full += 1
                parts.append(new)
            else:
                last = old.Date.max()
                fresh = download_history(sym, last - pd.Timedelta(days=10))
                overlap = old.merge(fresh, on="Date", suffixes=("_old", "_new"))
                rescaled = (not overlap.empty) and not np.allclose(
                    overlap.Close_old, overlap.Close_new, rtol=1e-3, equal_nan=True)
                if rescaled:   # dividend/split since last run -> refresh this symbol completely
                    parts.append(download_history(sym, full_start))
                    n_full += 1
                else:
                    parts.append(pd.concat([old[old.Date < fresh.Date.min()] if not fresh.empty else old, fresh]))
                    n_inc += 1
            time.sleep(0.3)  # be polite to the API
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        df = df.drop_duplicates(["Ticker", "Date"], keep="last").sort_values(["Ticker", "Date"])
        print(f"{desc}: {n_inc} symbols updated incrementally, {n_full} downloaded in full, {len(df):,} rows")
        return df.reset_index(drop=True)

    def fetch_prices(self):
        self.stocks_df = self._incremental_update(self.stocks_df, self.tickers, "stocks")
        missing = sorted(set(self.tickers) - set(self.stocks_df.Ticker))
        if missing:
            print(f"WARNING: no data for {missing}")
        self.market_df = self._incremental_update(self.market_df, self.market_symbols, "market")

    # ------------------------------------------------------------------ macro (FRED)
    def fetch_macro(self):
        frames = []
        for series in tqdm(C.FRED_SERIES, desc="FRED"):
            s = pd.read_csv(C.FRED_CSV_URL.format(series=series), parse_dates=[0], na_values=".")
            s.columns = ["Date", series]
            frames.append(s.set_index("Date"))
            time.sleep(0.3)
        self.macro_df = pd.concat(frames, axis=1).sort_index().rename_axis("Date").reset_index()

    # ------------------------------------------------------------------ CBOE volatility indexes
    def fetch_cboe(self):
        frames = []
        for sym in C.CBOE_SERIES:
            r = requests.get(C.CBOE_URL.format(symbol=sym), headers=C.HTTP_HEADERS, timeout=60)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            d["DATE"] = pd.to_datetime(d["DATE"], format="%m/%d/%Y")
            value_col = "CLOSE" if "CLOSE" in d.columns else sym  # VVIX/SKEW files have one value column
            frames.append(d.set_index("DATE")[[value_col]].rename(columns={value_col: sym.lower()}))
        self.cboe_df = pd.concat(frames, axis=1).sort_index().rename_axis("Date").reset_index()

    # ------------------------------------------------------------------ sectors (web scraping)
    def fetch_sectors(self):
        """GICS sectors from the Wikipedia S&P 500 table; yfinance .info for tickers not listed there."""
        r = requests.get(C.WIKI_SP500_URL, headers=C.HTTP_HEADERS, timeout=60)
        r.raise_for_status()
        wiki = pd.read_html(io.StringIO(r.text))[0]
        wiki["Ticker"] = wiki["Symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (yfinance style)
        wiki = wiki.rename(columns={"GICS Sector": "sector", "GICS Sub-Industry": "sub_industry"})
        wiki = wiki[["Ticker", "sector", "sub_industry"]]
        wiki["sector_source"] = "wikipedia"

        known = set(wiki.Ticker)
        extra = []
        for t in [t for t in self.tickers if t not in known]:
            try:
                info = yf.Ticker(t).info
                extra.append({"Ticker": t, "sector": info.get("sector", "Unknown"),
                              "sub_industry": info.get("industry", "Unknown"), "sector_source": "yfinance"})
            except Exception:
                extra.append({"Ticker": t, "sector": "Unknown", "sub_industry": "Unknown", "sector_source": "none"})
            time.sleep(0.3)
        # yfinance uses Morningstar names ("Technology"); map them onto GICS names for consistency
        yf_to_gics = {"Technology": "Information Technology", "Healthcare": "Health Care",
                      "Financial Services": "Financials", "Consumer Cyclical": "Consumer Discretionary",
                      "Consumer Defensive": "Consumer Staples", "Basic Materials": "Materials",
                      "Communication Services": "Communication Services"}
        extra_df = pd.DataFrame(extra)
        if not extra_df.empty:
            extra_df["sector"] = extra_df["sector"].replace(yf_to_gics)
        all_sec = pd.concat([wiki, extra_df], ignore_index=True)
        self.sectors_df = all_sec[all_sec.Ticker.isin(self.tickers)].reset_index(drop=True)

    # ------------------------------------------------------------------ orchestration
    def fetch(self):
        """Fetch everything (prices incrementally on top of whatever is already loaded)."""
        self.fetch_prices()
        self.fetch_macro()
        self.fetch_cboe()
        if self.sectors_df is None or set(self.tickers) - set(self.sectors_df.Ticker):
            self.fetch_sectors()

    _FILES = {"stocks_df": "stocks.parquet", "market_df": "market.parquet", "macro_df": "macro.parquet",
              "cboe_df": "cboe.parquet", "sectors_df": "sectors.parquet"}

    def persist(self, raw_dir=C.RAW_DIR):
        raw_dir.mkdir(parents=True, exist_ok=True)
        for attr, fname in self._FILES.items():
            df = getattr(self, attr)
            if df is not None:
                df.to_parquet(raw_dir / fname, compression="brotli", index=False)
        print(f"raw data saved to {raw_dir}")

    def load(self, raw_dir=C.RAW_DIR, missing_ok: bool = False) -> "DataRepository":
        for attr, fname in self._FILES.items():
            path = raw_dir / fname
            if path.exists():
                setattr(self, attr, pd.read_parquet(path))
            elif not missing_ok:
                raise FileNotFoundError(f"{path} not found - run with --mode fetch first")
        return self
