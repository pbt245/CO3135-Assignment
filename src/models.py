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
ALPHA_MAX = 1e6  # upper limit when the grid is extended (see tune_ridge)
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

    Variants added after the first test run (lecturer's feedback; defaults = the frozen B1):
      fset="F0"       minimal inputs (features.F0): lag_d7, weekday/month dummies, RAW temperature
                      (no degree days), so "F1 vs F0" isolates the value of feature engineering.
      per_hour=False  ONE Ridge for all hours with hour dummies and no hour interactions, so the
                      structure matches the single global LightGBM ("24 models vs 1").
    """

    def __init__(self, mode: str, alpha: float = ALPHA_FOR_BASE, th: float | None = 16, tc: float | None = 22,
                 target: str = "level", fset: str = "F1", per_hour: bool = True):
        assert fset in F.FEATURE_SETS
        self.mode, self.alpha, self.th, self.tc, self.target = mode, alpha, th, tc, target
        self.fset, self.per_hour = fset, per_hour
        self.models: dict[int, object] = {}

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fset == "F1":
            X = df[F.LAG + F.HOL + ["doy_sin", "doy_cos"]].astype(float).copy()
        else:
            X = df[["lag_d7"]].astype(float).copy()
        # fixed category sets -> identical columns in train, val and test. One-hot coding of weekday and
        # month is needed for any linear model (an integer 0..6 would impose a straight line), so it is
        # preprocessing, not feature engineering, and is kept in F0 too.
        for k in range(1, 7):
            X[f"dow_{k}"] = (df["dow"] == k).astype(float)
        for m in range(2, 13):
            X[f"month_{m}"] = (df["month"] == m).astype(float)
        if self.fset == "F1":
            for c in F.WX[self.mode]:
                X[f"hdd_{c}"] = np.maximum(self.th - df[c], 0.0)
                X[f"cdd_{c}"] = np.maximum(df[c] - self.tc, 0.0)
        else:
            tcol = F.F0[self.mode][-1]  # raw temperature of this weather setting
            X[tcol] = df[tcol].astype(float)
        if not self.per_hour:
            # hour only shifts the level; it does not change the effect of other features
            for h in range(1, 24):
                X[f"hour_{h}"] = (df["hour"] == h).astype(float)
        return X

    def _groups(self, df: pd.DataFrame):
        if self.per_hour:
            for h in range(24):
                yield h, (df["hour"] == h).to_numpy()
        else:
            yield -1, np.ones(len(df), dtype=bool)

    def fit(self, df: pd.DataFrame) -> "RidgePerHour":
        X, y = self.design(df), to_target(df, self.target)
        for h, m in self._groups(df):
            # scaler is inside the pipeline -> fitted on the training rows only
            self.models[h] = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X[m], y[m])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, out = self.design(df), np.full(len(df), np.nan)
        for h, m in self._groups(df):
            if m.any():
                out[m] = self.models[h].predict(X[m])
        return from_target(df, out, self.target)


def tune_ridge(tr: pd.DataFrame, va: pd.DataFrame, mode: str, fset: str = "F1",
               per_hour: bool = True) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """For each target option -- stage 1: degree-day bases at a middle alpha; stage 2:
    alpha at the chosen bases. The best (target, bases, alpha) by validation MAE wins.
    Everything is chosen in this weather mode only (no mixing of the two settings).
    F0 has no degree days, so only stage 2 (alpha) runs; the rest of the procedure is identical."""
    base_rows, alpha_rows = [], []
    for target in TARGETS:
        if fset == "F1":
            rows = []
            for th, tc in itertools.product(TH_GRID, TC_GRID):
                if th > tc:
                    continue
                m = RidgePerHour(mode, ALPHA_FOR_BASE, th, tc, target, fset, per_hour).fit(tr)
                rows.append({"target": target, "th": th, "tc": tc, "val_mae": mae(va["y"], m.predict(va))})
            bt = pd.DataFrame(rows).sort_values("val_mae")
            th, tc = (float(v) for v in bt.iloc[0][["th", "tc"]])
            base_rows.append(bt)
        else:
            th = tc = None
        grid, done = [float(a) for a in ALPHA_GRID], 0
        while True:
            for a in grid[done:]:
                m = RidgePerHour(mode, a, th, tc, target, fset, per_hour).fit(tr)
                alpha_rows.append({"target": target, "th": th, "tc": tc, "alpha": a,
                                   "val_mae": mae(va["y"], m.predict(va))})
            done = len(grid)
            best_a = min((r for r in alpha_rows if r["target"] == target), key=lambda r: r["val_mae"])["alpha"]
            # Rule of the plan: an optimum on the upper edge means the grid was too narrow -> extend it by half
            # decades before choosing. Never triggered for B1 (interior optimum). The global Ridge needs it: one
            # model on ~24x more rows needs a larger penalty, because sklearn's alpha is not divided by n.
            if best_a < grid[-1] or grid[-1] >= ALPHA_MAX:
                break
            grid.append(grid[-1] * 10 ** 0.5)
    base_tab = pd.concat(base_rows, ignore_index=True) if base_rows else pd.DataFrame()
    alpha_tab = pd.DataFrame(alpha_rows)
    best = alpha_tab.loc[alpha_tab["val_mae"].idxmin()]
    has_base = fset == "F1"
    params = {"mode": mode, "target": best["target"],
              "th": float(best["th"]) if has_base else None, "tc": float(best["tc"]) if has_base else None,
              "alpha": float(best["alpha"]), "val_mae": float(best["val_mae"]),
              "th_on_grid_edge": has_base and float(best["th"]) in (min(TH_GRID), max(TH_GRID)),
              "tc_on_grid_edge": has_base and float(best["tc"]) in (min(TC_GRID), max(TC_GRID)),
              "alpha_on_grid_edge": float(best["alpha"]) in (
                  alpha_tab["alpha"].min(), alpha_tab.loc[alpha_tab["target"] == best["target"], "alpha"].max())}
    # keys only for the new variants, so a rerun of script 02 writes byte-identical frozen B1 files
    if fset != "F1" or not per_hour:
        params.update({"fset": fset, "per_hour": per_hour})
    return params, base_tab, alpha_tab


def ridge_from_params(p: dict) -> RidgePerHour:
    return RidgePerHour(p["mode"], p["alpha"], p.get("th"), p.get("tc"), p["target"],
                        p.get("fset", "F1"), p.get("per_hour", True))


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


def lgb_matrix(df: pd.DataFrame, mode: str, fset: str = "F1") -> pd.DataFrame:
    """Raw features, no degree days: trees are invariant to monotone transforms of one
    variable and find temperature thresholds themselves."""
    return df[F.feature_cols(mode, fset)].astype(float)


class LGBMModel:
    """Thin wrapper so that the target option is handled in one place.
    fset="F0" (added after the first test run): minimal inputs, to measure feature engineering for trees."""

    def __init__(self, mode: str, params: dict, target: str = "level", fset: str = "F1"):
        self.mode, self.params, self.target, self.fset = mode, params, target, fset
        self.booster: lgb.Booster | None = None

    def fit(self, tr: pd.DataFrame, num_rounds: int, va: pd.DataFrame | None = None) -> "LGBMModel":
        dtr = lgb.Dataset(lgb_matrix(tr, self.mode, self.fset), to_target(tr, self.target), free_raw_data=False)
        kw = {}
        if va is not None:  # early stopping on validation MAE (offset by the same lag -> same MAE)
            dva = lgb.Dataset(lgb_matrix(va, self.mode, self.fset), to_target(va, self.target), reference=dtr)
            kw = {"valid_sets": [dva], "callbacks": [lgb.early_stopping(C.LGB_EARLY_STOP, verbose=False)]}
        self.booster = lgb.train({**LGB_FIXED, **self.params}, dtr, num_boost_round=num_rounds, **kw)
        return self

    @property
    def best_iteration(self) -> int:
        return int(self.booster.best_iteration or self.booster.current_iteration())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(lgb_matrix(df, self.mode, self.fset), num_iteration=self.best_iteration)
        return from_target(df, raw, self.target)


class LGBMPerHour:
    """24 LightGBM models, one per local hour, sharing one set of hyper-parameters (added after the
    first test run, so that "24 models vs 1" can be compared at matched structure with RidgePerHour).
    Each hour has its own early stopping, so its own number of trees. Expected weakness: each model
    sees only ~1/24 of the rows (~730 per hour in 2015-16), and trees need data in every leaf."""

    def __init__(self, mode: str, params: dict, target: str = "level", fset: str = "F1"):
        self.mode, self.params, self.target, self.fset = mode, params, target, fset
        self.boosters: dict[int, lgb.Booster] = {}

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return lgb_matrix(df, self.mode, self.fset).drop(columns="hour")  # constant inside one hour

    def fit(self, tr: pd.DataFrame, num_rounds: int | dict, va: pd.DataFrame | None = None) -> "LGBMPerHour":
        """num_rounds: an int (maximum, with early stopping on `va`) or {hour: trees} when refitting."""
        for h in range(24):
            t = tr[tr["hour"] == h]
            dtr = lgb.Dataset(self._X(t), to_target(t, self.target), free_raw_data=False)
            n = num_rounds[h] if isinstance(num_rounds, dict) else num_rounds
            kw = {}
            if va is not None:
                v = va[va["hour"] == h]
                dva = lgb.Dataset(self._X(v), to_target(v, self.target), reference=dtr)
                kw = {"valid_sets": [dva], "callbacks": [lgb.early_stopping(C.LGB_EARLY_STOP, verbose=False)]}
            self.boosters[h] = lgb.train({**LGB_FIXED, **self.params}, dtr, num_boost_round=n, **kw)
        return self

    @property
    def best_iteration(self) -> dict[int, int]:
        return {h: int(b.best_iteration or b.current_iteration()) for h, b in self.boosters.items()}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out, it = np.full(len(df), np.nan), self.best_iteration
        for h in range(24):
            m = (df["hour"] == h).to_numpy()
            if m.any():
                out[m] = self.boosters[h].predict(self._X(df[m]), num_iteration=it[h])
        return from_target(df, out, self.target)


def tune_lgbm(tr: pd.DataFrame, va: pd.DataFrame, mode: str,
              n_trials: int = C.N_TRIALS, fset: str = "F1", per_hour: bool = False) -> tuple[dict, pd.DataFrame]:
    """Optuna TPE on the target option + 4 capacity/regularisation knobs. Small budget
    (30 trials) on purpose: with ~2 years of training data, heavy tuning overfits the
    validation year. Learning rate is fixed; the number of trees comes from early stopping.
    The new variants (fset="F0", per_hour=True) use exactly the same search space, budget and seed."""
    cls = LGBMPerHour if per_hour else LGBMModel

    def objective(trial: optuna.Trial) -> float:
        target = trial.suggest_categorical("target", list(TARGETS))
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        t0 = time.time()
        m = cls(mode, params, target, fset).fit(tr, C.LGB_MAX_ROUNDS, va)
        bi = m.best_iteration
        trial.set_user_attr("best_iteration", {str(k): v for k, v in bi.items()} if isinstance(bi, dict) else bi)
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
              # 2015-17 (no validation set is left to stop on; the plan fixes this choice).
              # Per-hour variant: one tree count per hour, {"0": n0, ..., "23": n23}.
              "num_rounds": bt.user_attrs["best_iteration"], "val_mae": float(bt.value),
              "n_trials": n_trials, "smoke": C.SMOKE,
              "versions": {"lightgbm": lgb.__version__, "optuna": optuna.__version__}}
    if not isinstance(params["num_rounds"], dict):
        params["num_rounds"] = int(params["num_rounds"])
    # keys only for the new variants, so a rerun of script 03 writes the same frozen M1 files
    if fset != "F1" or per_hour:
        params.update({"fset": fset, "per_hour": per_hour})
    return params, trials


def lgbm_from_params(p: dict) -> LGBMModel | LGBMPerHour:
    cls = LGBMPerHour if p.get("per_hour", False) else LGBMModel
    return cls(p["mode"], p["lgb_params"], p["target"], p.get("fset", "F1"))


def lgbm_rounds(p: dict) -> int | dict[int, int]:
    """Frozen number of trees: an int, or {hour: trees} for the per-hour variant (JSON keys are strings)."""
    n = p["num_rounds"]
    return {int(k): int(v) for k, v in n.items()} if isinstance(n, dict) else int(n)


def shap_values(model: LGBMModel, df: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP contributions via LightGBM's own pred_contrib (no extra package).
    Last column = expected value. With target "delta7" the contributions explain the
    predicted CHANGE vs last week, not the load level -- said explicitly in the report."""
    X = lgb_matrix(df, model.mode)
    contrib = model.booster.predict(X, pred_contrib=True, num_iteration=model.best_iteration)
    return pd.DataFrame(contrib, columns=list(X.columns) + ["bias"], index=df.index)
