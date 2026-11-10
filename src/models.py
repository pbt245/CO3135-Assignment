"""B0 seasonal naive, B1 per-hour Ridge with degree-day features, M1 LightGBM + Optuna."""
from __future__ import annotations

import itertools
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import features as F


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


# ======================================================================== B0
def predict_b0(df: pd.DataFrame) -> np.ndarray:
    """Seasonal naive: same local hour one week earlier. The floor every model must beat;
    it already captures the daily and weekly cycle for free."""
    return df["lag_d7"].to_numpy()


# ======================================================================== B1 Ridge
# 11 values, log-spaced 1e-2..1e3; the middle value 10**0.5 is used while choosing the
# degree-day bases (bases first, then alpha: sequential search keeps the grid at 24 + 11
# fits instead of 24 x 11, acceptable because Ridge reacts smoothly to alpha).
ALPHA_GRID = np.logspace(-2, 3, 11)
ALPHA_FOR_BASE = float(ALPHA_GRID[5])
# Heating base T_h and cooling base T_c are separate, so a flat comfort zone between
# them is allowed (one shared base would force the two branches to meet in one point).
# The grid is a heuristic; if the optimum hits its edge we report it.
# Widened after the first validation run put T_c on the lower edge (18), as the plan
# prescribes ("extend the grid before choosing").
TH_GRID = [10, 12, 14, 16, 18, 20]
TC_GRID = [14, 16, 18, 20, 22, 24, 26]
# Target option (pre-registered rule in the pipeline plan: if validation bias >= 1% of
# mean load, also try predicting the change from last week).
#   "level":   predict y directly
#   "delta7":  predict y - y(D-7, h), then add the lag back. The week-ago level carries
#              the slow drift (economy, trend) that a model fitted on older years
#              cannot extrapolate (the first run had a -1.3% bias on 2017).
TARGETS = ("level", "delta7")


def to_target(df: pd.DataFrame, target: str) -> np.ndarray:
    y = df["y"].to_numpy()
    return y if target == "level" else y - df["lag_d7"].to_numpy()


def from_target(df: pd.DataFrame, pred: np.ndarray, target: str) -> np.ndarray:
    return pred if target == "level" else pred + df["lag_d7"].to_numpy()


class RidgePerHour:
    """24 independent Ridge models (one per local hour), shared alpha and bases.

    Why per hour: with every feature interacted with the hour and one shared alpha, a
    global Ridge equals 24 separate Ridges, so this is the simplest way to let the
    effect of temperature, weekday and lags differ by hour of day (a 07:00 heating
    peak is not a 15:00 cooling peak).
    Why degree days (HDD/CDD) instead of raw temperature: load is U-shaped in
    temperature; a linear model needs the two branches spelled out as features.
    The coefficients are NOT interpreted (shrunk, collinear lags) -- explanation is
    done with SHAP on LightGBM.
    """

    def __init__(self, mode: str, alpha: float = ALPHA_FOR_BASE, th: float = 16, tc: float = 22,
                 target: str = "level"):
        self.mode, self.alpha, self.th, self.tc, self.target = mode, alpha, th, tc, target
        self.models: dict[int, object] = {}

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[F.LAG + F.HOL + ["doy_sin", "doy_cos"]].astype(float).copy()
        # fixed category sets -> identical columns in train, val and test
        for k in range(1, 7):
            X[f"dow_{k}"] = (df["dow"] == k).astype(float)
        for m in range(2, 13):
            X[f"month_{m}"] = (df["month"] == m).astype(float)
        for c in F.WX[self.mode]:
            X[f"hdd_{c}"] = np.maximum(self.th - df[c], 0.0)
            X[f"cdd_{c}"] = np.maximum(df[c] - self.tc, 0.0)
        return X

    def fit(self, df: pd.DataFrame) -> "RidgePerHour":
        X, y = self.design(df), to_target(df, self.target)
        for h in range(24):
            m = (df["hour"] == h).to_numpy()
            # scaler is inside the pipeline -> fitted on the training rows only
            self.models[h] = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X[m], y[m])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, out = self.design(df), np.full(len(df), np.nan)
        for h in range(24):
            m = (df["hour"] == h).to_numpy()
            if m.any():
                out[m] = self.models[h].predict(X[m])
        return from_target(df, out, self.target)


def tune_ridge(tr: pd.DataFrame, va: pd.DataFrame, mode: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """For each target option -- stage 1: degree-day bases at a middle alpha; stage 2:
    alpha at the chosen bases. The best (target, bases, alpha) by validation MAE wins.
    Everything is chosen in this weather mode only (no mixing of the two settings)."""
    base_rows, alpha_rows = [], []
    for target in TARGETS:
        rows = []
        for th, tc in itertools.product(TH_GRID, TC_GRID):
            if th > tc:
                continue
            m = RidgePerHour(mode, ALPHA_FOR_BASE, th, tc, target).fit(tr)
            rows.append({"target": target, "th": th, "tc": tc, "val_mae": mae(va["y"], m.predict(va))})
        bt = pd.DataFrame(rows).sort_values("val_mae")
        th, tc = (float(v) for v in bt.iloc[0][["th", "tc"]])
        base_rows.append(bt)
        for a in ALPHA_GRID:
            m = RidgePerHour(mode, float(a), th, tc, target).fit(tr)
            alpha_rows.append({"target": target, "th": th, "tc": tc, "alpha": float(a),
                               "val_mae": mae(va["y"], m.predict(va))})
    base_tab = pd.concat(base_rows, ignore_index=True)
    alpha_tab = pd.DataFrame(alpha_rows)
    best = alpha_tab.loc[alpha_tab["val_mae"].idxmin()]
    params = {"mode": mode, "target": best["target"], "th": float(best["th"]), "tc": float(best["tc"]),
              "alpha": float(best["alpha"]), "val_mae": float(best["val_mae"]),
              "th_on_grid_edge": float(best["th"]) in (min(TH_GRID), max(TH_GRID)),
              "tc_on_grid_edge": float(best["tc"]) in (min(TC_GRID), max(TC_GRID)),
              "alpha_on_grid_edge": float(best["alpha"]) in (ALPHA_GRID[0], ALPHA_GRID[-1])}
    return params, base_tab, alpha_tab


def ridge_from_params(p: dict) -> RidgePerHour:
    return RidgePerHour(p["mode"], p["alpha"], p["th"], p["tc"], p["target"])


# ======================================================================== M1 LightGBM
LGB_FIXED = {
    # Same squared-error loss as Ridge, so the comparison is about the function class,
    # not the loss. MAE is still what we select on and report (see evaluation notes).
    "objective": "regression",
    "metric": "mae",  # early stopping is judged on validation MAE, the headline metric
    "learning_rate": C.LGB_LEARNING_RATE,  # fixed: number of trees adapts via early stopping
    "seed": C.SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": C.N_THREADS,
    "verbosity": -1,
}


def lgb_matrix(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Raw features, no degree days: trees are invariant to monotone transforms of one
    variable and find temperature thresholds themselves."""
    return df[F.feature_cols(mode)].astype(float)


class LGBMModel:
    """Thin wrapper so that the target option is handled in one place."""

    def __init__(self, mode: str, params: dict, target: str = "level"):
        self.mode, self.params, self.target = mode, params, target
        self.booster: lgb.Booster | None = None

    def fit(self, tr: pd.DataFrame, num_rounds: int, va: pd.DataFrame | None = None) -> "LGBMModel":
        dtr = lgb.Dataset(lgb_matrix(tr, self.mode), to_target(tr, self.target), free_raw_data=False)
        kw = {}
        if va is not None:  # early stopping on validation MAE (offset by the same lag -> same MAE)
            dva = lgb.Dataset(lgb_matrix(va, self.mode), to_target(va, self.target), reference=dtr)
            kw = {"valid_sets": [dva], "callbacks": [lgb.early_stopping(C.LGB_EARLY_STOP, verbose=False)]}
        self.booster = lgb.train({**LGB_FIXED, **self.params}, dtr, num_boost_round=num_rounds, **kw)
        return self

    @property
    def best_iteration(self) -> int:
        return int(self.booster.best_iteration or self.booster.current_iteration())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(lgb_matrix(df, self.mode), num_iteration=self.best_iteration)
        return from_target(df, raw, self.target)


def tune_lgbm(tr: pd.DataFrame, va: pd.DataFrame, mode: str,
              n_trials: int = C.N_TRIALS) -> tuple[dict, pd.DataFrame]:
    """Optuna TPE on the target option + 4 capacity/regularisation knobs. Small budget
    (30 trials) on purpose: with ~2 years of training data, heavy tuning overfits the
    validation year. Learning rate is fixed; the number of trees comes from early stopping."""

    def objective(trial: optuna.Trial) -> float:
        target = trial.suggest_categorical("target", list(TARGETS))
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        t0 = time.time()
        m = LGBMModel(mode, params, target).fit(tr, C.LGB_MAX_ROUNDS, va)
        trial.set_user_attr("best_iteration", m.best_iteration)
        trial.set_user_attr("seconds", round(time.time() - t0, 2))
        return mae(va["y"], m.predict(va))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe()
    bt = study.best_trial
    lgb_params = {k: v for k, v in bt.params.items() if k != "target"}
    params = {"mode": mode, "target": bt.params["target"], "lgb_params": lgb_params,
              # trees found by early stopping on 2015-16 -> kept fixed when refitting on
              # 2015-17 (no validation set is left to stop on; the plan fixes this choice)
              "num_rounds": int(bt.user_attrs["best_iteration"]), "val_mae": float(bt.value),
              "n_trials": n_trials, "smoke": C.SMOKE,
              "versions": {"lightgbm": lgb.__version__, "optuna": optuna.__version__}}
    return params, trials


def lgbm_from_params(p: dict) -> LGBMModel:
    return LGBMModel(p["mode"], p["lgb_params"], p["target"])


def shap_values(model: LGBMModel, df: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP contributions via LightGBM's own pred_contrib (no extra package).
    Last column = expected value. With target "delta7" the contributions explain the
    predicted CHANGE vs last week, not the load level -- said explicitly in the report."""
    X = lgb_matrix(df, model.mode)
    contrib = model.booster.predict(X, pred_contrib=True, num_iteration=model.best_iteration)
    return pd.DataFrame(contrib, columns=list(X.columns) + ["bias"], index=df.index)
