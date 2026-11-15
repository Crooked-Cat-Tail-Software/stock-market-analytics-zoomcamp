"""Command-line entry point.

  python main.py research --mode local   # full research: tune models, pick strategy on validation, test report
  python main.py run --mode fetch        # daily run (cron): new data -> features -> saved models -> today's trades
  python main.py run --mode local        # same, from data already on disk (no downloads)
  python main.py run --mode local --skip-transform   # fastest: reuse data/processed/dataset.parquet

Run modes (rubric "two regimes"):
  --mode fetch  downloads only the missing days from the APIs (incremental) and re-builds features
  --mode local  never touches the network; reads data/raw (or data/processed with --skip-transform)
"""
import argparse
import time
import warnings
from datetime import datetime

import pandas as pd

from trading import config as C
from trading import pipeline as P
from trading.data_repo import DataRepository

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _print_latest(preds, strategy):
    picks = P.latest_trades(preds, strategy)
    last = preds.Date.max().date()
    print(f"\n=== Trades for the close of {last} (strategy: {strategy['name']}) ===")
    print(f"market regime OK (S&P500 > SMA200 and VIX < 30): {picks.attrs['regime_ok']}")
    if picks.empty:
        print("no trade today")
    else:
        print(picks.to_string(index=False))
    log = C.REPORTS_DIR / "latest_trades.csv"
    picks.assign(run_at=datetime.now().strftime("%Y-%m-%d %H:%M")).to_csv(
        log, mode="a", header=not log.exists(), index=False)
    return picks


def cmd_research(args):
    t0 = time.time()
    repo = P.get_data(args.mode)
    t = P.build_dataset(repo)
    preds = P.research(t)
    _strategies_and_report(preds, repo)
    print(f"research finished in {(time.time() - t0) / 60:.1f} min")


def _strategies_and_report(preds, repo):
    P.model_summary(preds, "validation").to_csv(C.REPORTS_DIR / "model_summary_validation.csv", index=False)
    P.model_summary(preds, "test").to_csv(C.REPORTS_DIR / "model_summary_test.csv", index=False)
    grid, strategies, _ = P.research_strategies(preds, repo.market_df)
    print("\nbest 5 strategies on VALIDATION:")
    print(grid.head(5)[["name", "CAGR", "sharpe", "max_drawdown", "n_trades"]].to_string(index=False))
    res, curves, trades = P.test_report(preds, repo.market_df, strategies)
    res.to_csv(C.REPORTS_DIR / "strategies_test.csv", index=False)
    curves.to_csv(C.REPORTS_DIR / "equity_curves_test.csv")
    trades["S5_best_on_validation"].to_csv(C.REPORTS_DIR / "trades_best_strategy_test.csv", index=False)
    print("\nTEST period results:")
    print(res[["name", "CAGR", "sharpe", "max_drawdown", "n_trades", "win_rate"]].to_string(index=False))
    _print_latest(preds, P.json.loads(P.BEST_STRATEGY_FILE.read_text()))


def cmd_run(args):
    t0 = time.time()
    if args.skip_transform:
        t = P.build_dataset(rebuild=False)
        repo = DataRepository().load()
    else:
        repo = P.get_data(args.mode)
        t = P.build_dataset(repo)
    print(f"dataset: {t.df.shape[0]:,} rows, last date {t.df.Date.max().date()}")
    preds = P.score_with_saved_models(t)
    strategy = P.json.loads(P.BEST_STRATEGY_FILE.read_text())
    _print_latest(preds, strategy)
    if args.report:
        _strategies_and_report(preds, repo)
    print(f"run finished in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("research", "run"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=["fetch", "local"], default="local")
        if name == "run":
            p.add_argument("--skip-transform", action="store_true", help="reuse data/processed/dataset.parquet")
            p.add_argument("--report", action="store_true", help="also re-run strategy selection + test report")
    args = ap.parse_args()
    C.REPORTS_DIR.mkdir(exist_ok=True)
    {"research": cmd_research, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
