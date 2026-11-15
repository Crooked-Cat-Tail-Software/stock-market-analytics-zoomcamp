"""High-level steps shared by main.py (CLI / cron) and the notebooks."""
from __future__ import annotations

import gc
import json

import pandas as pd

from . import config as C
from .data_repo import DataRepository
from .train import TrainModel
from .transform import TransformData

MODELS = ("dt", "rf", "lr", "xgb")
TARGETS = (C.TARGET, C.TARGET_REL)
PRED_FILE = C.PROCESSED_DIR / "predictions.parquet"


def get_data(mode: str = "local") -> DataRepository:
    """mode='fetch': incremental download from the APIs; mode='local': read data/raw only."""
    repo = DataRepository()
    if mode == "fetch":
        repo.load(missing_ok=True)      # whatever is stored already -> only new days are downloaded
        repo.fetch()
        repo.persist()
    elif mode == "local":
        repo.load()
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'fetch' or 'local')")
    return repo


def build_dataset(repo: DataRepository | None = None, rebuild: bool = True) -> TransformData:
    t = TransformData(repo)
    if rebuild:
        t.transform()
        t.persist()
    else:
        t.load()
    return t


def research(t: TransformData, models=MODELS, targets=TARGETS) -> pd.DataFrame:
    """Tune every model on validation, refit on train+validation, save models and predictions."""
    preds, importances = None, []
    for target in targets:
        print(f"\n===== target: {target} =====")
        tm = TrainModel(t.df, t.feature_sets, target=target).prepare()
        tm.tune_and_fit(models).refit_final(models)
        tm.persist()
        p = tm.predict()
        imp = tm.feature_importance(40)
        imp.insert(0, "target", target)
        importances.append(imp)
        preds = p if preds is None else preds.merge(
            p[["Date", "Ticker"] + [c for c in p.columns if c.startswith("proba_")]], on=["Date", "Ticker"])
        del tm
        gc.collect()
    preds.to_parquet(PRED_FILE, index=False)
    pd.concat(importances).to_csv(C.REPORTS_DIR / "feature_importance.csv", index=False)
    return preds


def score_with_saved_models(t: TransformData, models=MODELS, targets=TARGETS) -> pd.DataFrame:
    """Daily run: no tuning - load the saved final models and score test + live rows."""
    preds = None
    for target in targets:
        tm = TrainModel(t.df, t.feature_sets, target=target).prepare()
        tm.load(models=models)
        p = tm.predict()
        preds = p if preds is None else preds.merge(
            p[["Date", "Ticker"] + [c for c in p.columns if c.startswith("proba_")]], on=["Date", "Ticker"])
        del tm
        gc.collect()
    # keep validation probabilities from the research run (needed to re-tune strategy thresholds)
    if PRED_FILE.exists():
        old = pd.read_parquet(PRED_FILE)
        old_valid = old[old.split == "validation"]
        preds = pd.concat([old_valid, preds[preds.split != "validation"]], ignore_index=True)
    preds.to_parquet(PRED_FILE, index=False)
    return preds


BEST_STRATEGY_FILE = C.MODEL_DIR / "best_strategy.json"


def model_summary(preds: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Out-of-sample quality of every probability column: AUC and what the daily top-5 would earn."""
    from sklearn.metrics import roc_auc_score
    from .train import daily_top_k_stats
    part = preds[preds.split == split]
    rows = []
    for col in [c for c in part.columns if c.startswith("proba_")]:
        target = C.TARGET_REL if col.endswith("_rel") else C.TARGET
        hit5, g5 = daily_top_k_stats(part, part[col].to_numpy(), target, 5)
        rows.append({"score": col, "target": target, "split": split,
                     "auc": roc_auc_score(part[target], part[col]),
                     "top5_hit_rate": hit5, "top5_mean_growth_10d": g5,
                     "all_stocks_mean_growth_10d": part[C.TARGET_GROWTH].mean()})
    return pd.DataFrame(rows).sort_values("top5_mean_growth_10d", ascending=False).reset_index(drop=True)


def research_strategies(preds: pd.DataFrame, market_df: pd.DataFrame, n_scores: int = 2,
                        metric: str = "sharpe"):
    """Pick the strategy on VALIDATION, then report it (and simpler variants) on TEST once."""
    from .simulate import PortfolioSimulator, Strategy, strategy_grid

    valid = preds[preds.split == "validation"]
    test = preds[preds.split == "test"]
    # only the n best scores (by validation top-5 return) enter the big grid - keeps it fast
    vsum = model_summary(preds, "validation")
    scores = vsum.score.head(n_scores).tolist()
    thresholds = {s: [0.0] + [round(float(valid[s].quantile(q)), 3) for q in (0.8, 0.9, 0.95)] for s in scores}

    sim_v = PortfolioSimulator(valid)
    grid_rows = []
    for s in strategy_grid(scores, thresholds):
        row, _, _ = sim_v.evaluate(s)
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows).sort_values(metric, ascending=False).reset_index(drop=True)
    grid.to_csv(C.REPORTS_DIR / "strategy_grid_validation.csv", index=False)

    best = grid.iloc[0]
    best_s = Strategy(name="S5_best_on_validation", score=best.score, top_k=int(best.top_k),
                      threshold=float(best.threshold),
                      stop_loss=None if pd.isna(best.stop_loss) else float(best.stop_loss),
                      regime_filter=bool(best.regime_filter), sizing=best.sizing)
    save_json(best_s.__dict__, BEST_STRATEGY_FILE)

    top_score = scores[0]
    named = [
        Strategy("S1_top5_equal", top_score, top_k=5),
        Strategy("S2_top5_stoploss10", top_score, top_k=5, stop_loss=0.10),
        Strategy("S3_top5_regime_filter", top_score, top_k=5, regime_filter=True),
        Strategy("S4_top3_proba_weighted", top_score, top_k=3, sizing="proba",
                 threshold=thresholds[top_score][1]),
        best_s,
    ]
    return grid, named, scores


def test_report(preds: pd.DataFrame, market_df: pd.DataFrame, strategies) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run strategies on TEST and compare with SPY and the equal-weight universe."""
    from .simulate import PortfolioSimulator, benchmark_equity, equal_weight_equity, perf_metrics
    test = preds[preds.split == "test"]
    sim = PortfolioSimulator(test)
    rows, curves, trades = [], {}, {}
    for s in strategies:
        row, eq, tr = sim.evaluate(s)
        rows.append(row)
        curves[s.name] = eq
        trades[s.name] = tr
    curves["SPY_buy_and_hold"] = benchmark_equity(market_df, sim.dates)
    curves["equal_weight_universe"] = equal_weight_equity(test, sim.dates)
    for b in ("SPY_buy_and_hold", "equal_weight_universe"):
        rows.append({"name": b, **perf_metrics(curves[b])})
    res = pd.DataFrame(rows)
    cols = ["name", "CAGR", "sharpe", "sortino", "max_drawdown", "calmar", "ann_vol", "total_return",
            "rolling_1y_median", "rolling_1y_min", "n_trades", "win_rate", "avg_trade_return", "stop_exits",
            "score", "top_k", "threshold", "stop_loss", "regime_filter", "sizing"]
    res = res[[c for c in cols if c in res.columns]]
    return res, pd.DataFrame(curves), trades


def latest_trades(preds: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    """What the chosen strategy would buy at the most recent close."""
    last = preds.Date.max()
    day = preds[preds.Date == last]
    regime_ok = bool((day.snp500_above_sma200.iloc[0] == 1) and (day.vix_close.iloc[0] < 30))
    picks = day[day[strategy["score"]] >= strategy["threshold"]].nlargest(strategy["top_k"], strategy["score"])
    if strategy["regime_filter"] and not regime_ok:
        picks = picks.iloc[0:0]
    out = picks[["Date", "Ticker", "sector", "Close", strategy["score"]]].copy()
    out["rank"] = range(1, len(out) + 1)
    if strategy["stop_loss"]:
        out["stop_price"] = out.Close * (1 - strategy["stop_loss"])
    out["planned_exit"] = f"close of the {C.HORIZON}th trading day after {last.date()}"
    out.attrs["regime_ok"] = regime_ok
    return out.reset_index(drop=True)


def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PRED_FILE)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
