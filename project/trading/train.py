"""Step 3 - Modeling: dummies, temporal split, model training / tuning, predictions.

Split (by date, with an embargo so overlapping 10-day targets cannot leak across splits)
  train       2000-01 .. 2018-12
  validation  2019-01 .. 2022-06   -> hyper-parameters AND trading-strategy parameters are chosen here
  test        2022-07 .. today-10d -> touched only once, for the final report
  live        last 10 days         -> target unknown yet, used for today's trade list

Every model is fitted twice:
  model_A on train            -> probabilities for validation (honest out-of-sample for tuning)
  model_B on train+validation -> probabilities for test and live rows
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as C

DUMMY_SOURCES = ["Month", "Weekday", "sector"]   # month_wom / Ticker dummies over-fit in HW4 -> left out
KEEP_FOR_SIM = ["Date", "Ticker", "Open", "High", "Low", "Close", "sector", "split",
                "snp500_above_sma200", "vix_close", "vix_above_30", "Year", "Month",
                "rank_growth_90d", "golden_cross", "rsi", "sma200_rel", "volatility"]


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    d = df.Date
    train_end, valid_end = pd.Timestamp(C.TRAIN_END), pd.Timestamp(C.VALID_END)
    embargo = pd.Timedelta(days=int(C.EMBARGO_DAYS * 7 / 5) + 2)   # trading -> calendar days
    split = np.select(
        [d <= train_end - embargo, d <= train_end,
         d <= valid_end - embargo, d <= valid_end],
        ["train", "embargo", "validation", "embargo"], default="test")
    split = np.where((split == "test") & df[C.TARGET].isna(), "live", split)
    df["split"] = split
    return df


def daily_top_k_stats(frame: pd.DataFrame, score: np.ndarray, target: str, k: int = 5) -> tuple[float, float]:
    """Hit rate and mean 10d growth of the k best-scored stocks per day (what a strategy would buy)."""
    tmp = frame[["Date", target, C.TARGET_GROWTH]].copy()
    tmp["score"] = score
    top = tmp[tmp.groupby("Date")["score"].rank(ascending=False, method="first") <= k]
    return float(top[target].mean()), float(top[C.TARGET_GROWTH].mean())


class TrainModel:
    def __init__(self, df: pd.DataFrame, feature_sets: dict, target: str = C.TARGET):
        self.df = df
        self.target = target
        # suffix used in file / column names: "" for the absolute target, "_rel" for "beats the S&P 500"
        self.tag = "_rel" if target == C.TARGET_REL else ""
        self.fs = feature_sets
        self.models_B: dict[str, object] = {}
        self.best_params: dict[str, dict] = {}
        self.scores: list[dict] = []

    # ------------------------------------------------------------------ data preparation
    def prepare(self):
        df = add_split(self.df)
        dummies = pd.get_dummies(df[DUMMY_SOURCES].astype(str), dtype="int8", prefix=DUMMY_SOURCES)
        self.DUMMIES = dummies.columns.tolist()
        self.NUMERIC = self.fs["NUMERIC"]
        self.TO_PREDICT = self.fs["TO_PREDICT"]
        self.FEATURES = self.NUMERIC + self.DUMMIES
        for col in self.DUMMIES:          # add in place: no copy of the 1.3 GB frame
            df[col] = dummies[col].to_numpy()
        self.df = df
        counts = self.df.split.value_counts()
        print(f"features: {len(self.NUMERIC)} numeric + {len(self.DUMMIES)} dummies = {len(self.FEATURES)}")
        print("rows per split:", counts.to_dict())
        return self

    def _xy(self, splits: list[str], fill: bool = True):
        mask = self.df.split.isin(splits).to_numpy()
        small = [c for c in dict.fromkeys(KEEP_FOR_SIM + self.TO_PREDICT) if c in self.df.columns]
        part = self.df.loc[mask, small]   # only the columns needed for evaluation / simulation
        X = np.empty((int(mask.sum()), len(self.FEATURES)), dtype=np.float32)   # column by column: no big temp
        for j, col in enumerate(self.FEATURES):
            X[:, j] = self.df[col].to_numpy(dtype=np.float32, na_value=np.nan)[mask]
        if fill:   # lecture approach for sklearn models: NaN -> 0 (XGBoost handles NaN natively)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y = part[self.target].to_numpy()
        return part, X, y

    # ------------------------------------------------------------------ model zoo
    @staticmethod
    def _make(name: str, params: dict):
        if name == "dt":
            return DecisionTreeClassifier(random_state=C.RANDOM_STATE, **params)
        if name == "rf":
            return RandomForestClassifier(n_estimators=200, max_samples=0.25, max_features="sqrt",
                                          n_jobs=-1, random_state=C.RANDOM_STATE, **params)
        if name == "lr":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, **params))
        if name == "xgb":
            return XGBClassifier(tree_method="hist", learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                                 eval_metric="auc", n_jobs=-1, random_state=C.RANDOM_STATE, **params)
        raise ValueError(name)

    GRIDS = {
        "dt": [{"max_depth": d, "min_samples_leaf": 200} for d in (3, 5, 7, 9, 12)],
        "rf": [{"max_depth": d, "min_samples_leaf": leaf} for d in (8, 14) for leaf in (50, 300)],
        "lr": [{"C": c} for c in (0.001, 0.01, 0.1)],
        "xgb": [{"max_depth": d, "min_child_weight": 100, "n_estimators": 1500, "early_stopping_rounds": 100}
                for d in (3, 5, 7)],
    }
    LR_SAMPLE = 300_000   # logistic regression is fitted on a random sample (speed)

    def tune_and_fit(self, models=("dt", "rf", "lr", "xgb")):
        """Grid search on validation, then refit the best config on train+validation."""
        tr, X_tr, y_tr = self._xy(["train"])
        va, X_va, y_va = self._xy(["validation"])
        _, X_tr_nan, _ = self._xy(["train"], fill=False)
        _, X_va_nan, _ = self._xy(["validation"], fill=False)
        rng = np.random.default_rng(C.RANDOM_STATE)
        self.valid_proba: dict[str, np.ndarray] = {}

        for name in models:
            best = None
            for params in self.GRIDS[name]:
                t0 = time.time()
                model = self._make(name, params)
                if name == "xgb":
                    model.fit(X_tr_nan, y_tr, eval_set=[(X_va_nan, y_va)], verbose=False)
                    proba = model.predict_proba(X_va_nan)[:, 1]
                    params = {**params, "n_estimators": int(model.best_iteration) + 1}
                    params.pop("early_stopping_rounds", None)
                elif name == "lr":
                    idx = rng.choice(len(X_tr), size=min(self.LR_SAMPLE, len(X_tr)), replace=False)
                    model.fit(X_tr[idx], y_tr[idx])
                    proba = model.predict_proba(X_va)[:, 1]
                else:
                    model.fit(X_tr, y_tr)
                    proba = model.predict_proba(X_va)[:, 1]
                auc = roc_auc_score(y_va, proba)
                hit5, growth5 = daily_top_k_stats(va, proba, self.target, 5)
                prec55 = precision_score(y_va, proba >= 0.55, zero_division=0)
                row = {"target": self.target, "model": name, "params": json.dumps(params), "valid_auc": round(auc, 4),
                       "valid_precision_at_0.55": round(prec55, 4), "valid_share_above_0.55": round(float((proba >= 0.55).mean()), 4),
                       "valid_top5_hit_rate": round(hit5, 4), "valid_top5_mean_growth": round(growth5, 4),
                       "fit_seconds": round(time.time() - t0, 1)}
                self.scores.append(row)
                print(row)
                if best is None or auc > best[0]:
                    best = (auc, params, proba)
            self.best_params[name] = best[1]
            self.valid_proba[name] = best[2]
        return self

    def refit_final(self, models=("dt", "rf", "lr", "xgb")):
        """model_B: best params, fitted on train+validation."""
        tv, X_tv, y_tv = self._xy(["train", "validation"], fill=False)
        X_tv_filled = np.nan_to_num(X_tv, nan=0.0, posinf=0.0, neginf=0.0)
        rng = np.random.default_rng(C.RANDOM_STATE)
        for name in models:
            params = dict(self.best_params[name])
            model = self._make(name, params)
            t0 = time.time()
            if name == "xgb":
                model.fit(X_tv, y_tv, verbose=False)
            elif name == "lr":
                idx = rng.choice(len(X_tv), size=min(self.LR_SAMPLE, len(X_tv)), replace=False)
                model.fit(X_tv_filled[idx], y_tv[idx])
            else:
                model.fit(X_tv_filled, y_tv)
            self.models_B[name] = model
            print(f"final {name} fitted on train+validation in {time.time() - t0:.0f}s")
        return self

    # ------------------------------------------------------------------ predictions
    def predict(self, splits=("test", "live")) -> pd.DataFrame:
        """Probabilities of model_B for test/live; validation rows get model_A out-of-sample probas."""
        part, X, _ = self._xy(list(splits), fill=False)
        X_filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        keep = [c for c in KEEP_FOR_SIM + self.TO_PREDICT if c in part.columns]
        out = part[keep].copy()
        for name, model in self.models_B.items():
            out[f"proba_{name}{self.tag}"] = model.predict_proba(X if name == "xgb" else X_filled)[:, 1]
        if getattr(self, "valid_proba", None):
            va = self.df.loc[self.df.split == "validation", keep].copy()
            for name, p in self.valid_proba.items():
                va[f"proba_{name}{self.tag}"] = p
            out = pd.concat([va, out], ignore_index=True)
        return add_manual_rules(out)

    def feature_importance(self, top: int = 30) -> pd.DataFrame:
        m = self.models_B.get("xgb")
        imp = pd.DataFrame({"feature": self.FEATURES, "xgb_gain": m.feature_importances_})
        if "rf" in self.models_B:
            imp["rf_importance"] = self.models_B["rf"].feature_importances_
        return imp.sort_values("xgb_gain", ascending=False).head(top).reset_index(drop=True)

    # ------------------------------------------------------------------ persistence
    def persist(self, model_dir=C.MODEL_DIR):
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, m in self.models_B.items():
            joblib.dump(m, model_dir / f"{name}{self.tag}.joblib")
        (model_dir / f"best_params{self.tag}.json").write_text(json.dumps(self.best_params, indent=1))
        (model_dir / "features.json").write_text(json.dumps(self.FEATURES))
        if self.scores:
            C.REPORTS_DIR.mkdir(exist_ok=True)
            pd.DataFrame(self.scores).to_csv(C.REPORTS_DIR / f"model_tuning_validation{self.tag}.csv", index=False)

    def load(self, model_dir=C.MODEL_DIR, models=("dt", "rf", "lr", "xgb")):
        self.best_params = json.loads((model_dir / f"best_params{self.tag}.json").read_text())
        saved_features = json.loads((model_dir / "features.json").read_text())
        for col in set(saved_features) - set(self.df.columns):   # e.g. a dummy for a month not yet seen
            self.df[col] = 0
        self.FEATURES = saved_features
        for name in models:
            path = model_dir / f"{name}{self.tag}.joblib"
            if path.exists():
                self.models_B[name] = joblib.load(path)
        return self


def add_manual_rules(pred: pd.DataFrame) -> pd.DataFrame:
    """Hand-made 'rule of thumb' predictions (lecture Module 3 idea) as baselines - need no training."""
    uptrend = pred["snp500_above_sma200"] == 1
    # momentum: top-decile 90-day growth, stock above its 50/200 cross, market in an uptrend
    pred["pred_manual_momentum"] = ((pred["rank_growth_90d"] >= 0.9) & (pred["golden_cross"] == 1)
                                    & uptrend).astype(int)
    # buy-the-dip: oversold (RSI < 30) quality trend (still above 200d average) in an up-market
    pred["pred_manual_dip"] = ((pred["rsi"] < 30) & (pred["sma200_rel"] > 1) & uptrend).astype(int)
    return pred
