"""Step 4 - Trading simulations.

1) vectorized_sim     - lecture approach: invest $100 in every positive prediction (sim1_* fields)
2) PortfolioSimulator - exact day-by-day simulation with capital re-investment:
     * every day 1/H of current equity is invested in that day's picks (H overlapping sleeves),
       so capital is always ~fully used and profits are re-invested
     * positions are closed after H trading days at the close, or earlier by a stop loss
       (gap-downs through the stop are filled at the open - no "perfect" fills)
     * options: top-k limit, probability threshold, probability-weighted sizing,
       market-regime filter (S&P 500 above its 200d average and VIX < 30)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config as C

H = C.HORIZON
TRADING_DAYS = 252


# ======================================================================================
# metrics
# ======================================================================================
def perf_metrics(equity: pd.Series) -> dict:
    """CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, rolling 1y return stats."""
    equity = equity.dropna()
    ret = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = ret.mean() * TRADING_DAYS / downside if downside > 0 else np.nan
    dd = equity / equity.cummax() - 1
    roll = equity.pct_change(TRADING_DAYS).dropna()
    return {
        "CAGR": cagr, "ann_vol": vol, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": dd.min(), "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "rolling_1y_median": roll.median() if len(roll) else np.nan,
        "rolling_1y_min": roll.min() if len(roll) else np.nan,
        "rolling_1y_max": roll.max() if len(roll) else np.nan,
        "years": years,
    }


def benchmark_equity(market_df: pd.DataFrame, dates: pd.DatetimeIndex, symbol: str = C.BENCHMARK,
                     initial: float = 10_000) -> pd.Series:
    """Buy & hold of the benchmark (SPY, dividends re-invested via auto-adjusted prices)."""
    s = market_df[market_df.Ticker == symbol].set_index("Date").Close.sort_index()
    s = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    return initial * s / s.iloc[0]


def equal_weight_equity(pred: pd.DataFrame, dates: pd.DatetimeIndex, initial: float = 10_000) -> pd.Series:
    """Daily-rebalanced equal-weight portfolio of the whole universe (the 'stock picking' benchmark)."""
    close = pred.pivot(index="Date", columns="Ticker", values="Close").reindex(dates)
    daily = close.pct_change(fill_method=None).mean(axis=1).fillna(0)
    return initial * (1 + daily).cumprod()


# ======================================================================================
# 1) vectorized simulation (lecture Module 4 style)
# ======================================================================================
def vectorized_sim(pred: pd.DataFrame, pred_cols: list[str], stake: float = 100.0) -> pd.DataFrame:
    """$100 on every positive prediction; rough CAGR from the capital needed to fund the trades.

    Adds sim1_gross_rev_<col>, sim1_fees_<col>, sim1_net_rev_<col> to `pred` (in place) and
    returns one summary row per prediction column.
    """
    rows = []
    years = (pred.Date.max() - pred.Date.min()).days / 365.25
    n_days = pred.Date.nunique()
    for col in pred_cols:
        on = pred[col] == 1
        gross = np.where(on, stake * (pred[C.TARGET_GROWTH] - 1), 0.0)
        fees = np.where(on, -stake * 2 * C.FEE_PER_SIDE, 0.0)
        pred[f"sim1_gross_rev_{col}"] = gross
        pred[f"sim1_fees_{col}"] = fees
        pred[f"sim1_net_rev_{col}"] = gross + fees
        per_day = pred[on].groupby("Date").size()
        n = int(on.sum())
        # money is locked for H days -> capital needed ~ H * stake * (75th pct. of trades per day)
        q75 = per_day.quantile(0.75) if len(per_day) else 0
        capital = H * stake * q75
        net = float((gross + fees).sum())
        rows.append({
            "prediction": col, "n_trades": n, "trading_days_with_trades": len(per_day),
            "share_of_days": len(per_day) / n_days, "avg_trades_per_day": per_day.mean() if n else 0,
            "precision": float(pred.loc[on, C.TARGET].mean()) if n else np.nan,
            "avg_growth_10d": float(pred.loc[on, C.TARGET_GROWTH].mean()) if n else np.nan,
            "net_revenue": net, "capital_q75": capital,
            "approx_CAGR": ((capital + net) / capital) ** (1 / years) - 1 if capital > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("approx_CAGR", ascending=False).reset_index(drop=True)


# ======================================================================================
# 2) exact portfolio simulation
# ======================================================================================
@dataclass
class Strategy:
    name: str
    score: str                    # probability column used to rank stocks
    top_k: int = 5                # max new positions per day
    threshold: float = 0.0        # minimum probability to buy
    stop_loss: float | None = None   # e.g. 0.08 -> sell if price falls 8% below entry
    regime_filter: bool = False   # trade only if S&P 500 > SMA200 and VIX < 30
    sizing: str = "equal"         # "equal" or "proba" (weights ~ probability above threshold)


class PortfolioSimulator:
    def __init__(self, pred: pd.DataFrame, initial: float = 10_000.0):
        pred = pred.sort_values(["Date", "Ticker"])
        self.pred = pred
        self.dates = pd.DatetimeIndex(sorted(pred.Date.unique()))
        self.initial = initial
        piv = {c: pred.pivot(index="Date", columns="Ticker", values=c).reindex(self.dates)
               for c in ("Open", "Low", "Close")}
        self.tickers = piv["Close"].columns
        self.t_idx = {t: i for i, t in enumerate(self.tickers)}
        self.open = piv["Open"].to_numpy()
        self.low = piv["Low"].to_numpy()
        self.close = piv["Close"].ffill().to_numpy()   # last known price for marking to market
        regime = pred.groupby("Date")[["snp500_above_sma200", "vix_close"]].first().reindex(self.dates)
        self.regime_ok = ((regime.snp500_above_sma200 == 1) & (regime.vix_close < 30)).to_numpy()
        self.by_day = {d: g for d, g in pred.groupby("Date")}

    def run(self, s: Strategy) -> tuple[pd.Series, pd.DataFrame]:
        cash, fee = self.initial, C.FEE_PER_SIDE
        open_pos: list[dict] = []
        trades, equity = [], np.empty(len(self.dates))
        for i, d in enumerate(self.dates):
            # ---- 1. manage open positions (stop loss intraday, time exit at the close)
            still_open = []
            for p in open_pos:
                j = p["col"]
                exit_px, reason = None, None
                if s.stop_loss is not None and not np.isnan(self.low[i, j]) and self.low[i, j] <= p["stop"]:
                    o = self.open[i, j]
                    exit_px, reason = (min(o, p["stop"]) if not np.isnan(o) else p["stop"]), "stop"
                elif i >= p["exit_i"]:
                    exit_px, reason = self.close[i, j], "time"
                if exit_px is None:
                    still_open.append(p)
                    continue
                cash += p["shares"] * exit_px * (1 - fee)
                trades.append({**{k: p[k] for k in ("ticker", "entry_date", "entry_px", "proba", "alloc")},
                               "exit_date": d, "exit_px": exit_px, "reason": reason,
                               "return": exit_px / p["entry_px"] * (1 - fee) ** 2 - 1})
            open_pos = still_open

            # ---- 2. new entries at today's close (H sleeves: invest ~1/H of equity per day)
            mtm = sum(p["shares"] * self.close[i, p["col"]] for p in open_pos)
            eq_now = cash + mtm
            day = self.by_day.get(d)
            can_trade = i + 1 < len(self.dates) and (not s.regime_filter or self.regime_ok[i])
            if day is not None and can_trade:
                cand = day[day[s.score] >= s.threshold].nlargest(s.top_k, s.score)
                budget = min(cash, eq_now / H)
                if len(cand) and budget > 1:
                    if s.sizing == "proba":
                        w = (cand[s.score] - s.threshold + 1e-3).to_numpy()
                    else:
                        w = np.ones(len(cand))
                    w = w / w.sum()
                    for (_, row), wi in zip(cand.iterrows(), w):
                        alloc = budget * wi
                        px = row.Close
                        if not np.isfinite(px) or px <= 0:
                            continue
                        shares = alloc * (1 - fee) / px
                        cash -= alloc
                        open_pos.append({"ticker": row.Ticker, "col": self.t_idx[row.Ticker], "shares": shares,
                                         "entry_px": px, "entry_date": d, "exit_i": i + H, "alloc": alloc,
                                         "proba": row[s.score],
                                         "stop": px * (1 - s.stop_loss) if s.stop_loss else -np.inf})
            equity[i] = cash + sum(p["shares"] * self.close[i, p["col"]] for p in open_pos)
        eq = pd.Series(equity, index=self.dates, name=s.name)
        return eq, pd.DataFrame(trades)

    def evaluate(self, s: Strategy) -> tuple[dict, pd.Series, pd.DataFrame]:
        eq, trades = self.run(s)
        m = perf_metrics(eq)
        m.update({"n_trades": len(trades),
                  "win_rate": float((trades["return"] > 0).mean()) if len(trades) else np.nan,
                  "avg_trade_return": float(trades["return"].mean()) if len(trades) else np.nan,
                  "stop_exits": float((trades.reason == "stop").mean()) if len(trades) else np.nan})
        return {**asdict(s), **m}, eq, trades


def strategy_grid(score_cols: list[str], thresholds: dict[str, list[float]]) -> list[Strategy]:
    """All combinations explored on the validation period."""
    grid = []
    for score in score_cols:
        for thr in thresholds[score]:
            for k in (3, 5, 10):
                for sl in (None, 0.05, 0.10):
                    for rf in (False, True):
                        for sizing in ("equal", "proba"):
                            name = f"{score}|thr{thr:.3f}|k{k}|sl{sl}|rf{int(rf)}|{sizing}"
                            grid.append(Strategy(name, score, k, thr, sl, rf, sizing))
    return grid
