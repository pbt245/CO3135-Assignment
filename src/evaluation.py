"""Metrics, paired block bootstrap and error breakdowns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# "Clearly better / about equal / undecided" threshold, fixed before looking at test:
# delta = 2% of the MAE of the SIMPLER model of the pair (heuristic, no source).
DELTA_FRAC = 0.02
N_BOOT = 2000


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    # Model predictions are never missing (asserted in the scripts); this only matters for the
    # TSO reference column, which may have gaps -> scored on its available hours, n reported.
    ok = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[ok], yhat[ok]
    e = y - yhat
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            # MAPE only for comparison with the literature (most papers report it);
            # safe here because load never gets near 0.
            "MAPE_%": float(100 * np.mean(np.abs(e) / y)),
            # bias = mean(prediction - actual): negative means under-forecasting
            "bias": float(np.mean(-e)), "n": int(len(y))}


def metric_table(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: metrics(df["y"], df[c]) for c in pred_cols}).T


def week_key(ts: pd.Series) -> pd.Series:
    iso = pd.to_datetime(ts).dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def paired_bootstrap(df: pd.DataFrame, a: str, b: str, n_boot: int = N_BOOT,
                     seed: int = C.SEED) -> dict:
    """CI for Delta = MAE(a) - MAE(b) (negative => a better).

    Resamples whole calendar weeks of the PAIRED differences |e_a| - |e_b|: errors of
    neighbouring hours are strongly autocorrelated, so resampling single hours would
    give far too narrow intervals.
    """
    d = (df["y"] - df[a]).abs() - (df["y"] - df[b]).abs()
    g = pd.DataFrame({"d": d.to_numpy(), "w": week_key(df["ts"]).to_numpy()}).groupby("w")["d"]
    s, n = g.sum().to_numpy(), g.count().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_weeks": int(len(s))}


def verdict(delta: float, lo: float, hi: float, ref_mae: float) -> str:
    dlt = DELTA_FRAC * ref_mae
    if (lo > 0 or hi < 0) and abs(delta) >= dlt:
        return "A clearly better" if delta < 0 else "B clearly better"
    if lo >= -dlt and hi <= dlt:
        return "about equal"
    return "undecided"


def compare_pairs(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Each pair is (complex, simple); the threshold uses the simpler model's MAE."""
    rows = []
    mean_load = float(df["y"].mean())
    month = pd.to_datetime(df["ts"]).dt.to_period("M")
    for a, b in pairs:
        r = paired_bootstrap(df, a, b)
        ref = metrics(df["y"], df[b])["MAE"]
        # how many calendar months A beats B: shows whether a gain is broad or comes
        # from a few unusual weeks
        by_m = df.assign(ea=(df["y"] - df[a]).abs(), eb=(df["y"] - df[b]).abs()).groupby(month)[["ea", "eb"]].mean()
        rows.append({"A": a, "B": b, "MAE_A": metrics(df["y"], df[a])["MAE"], "MAE_B": ref, **r,
                     "delta_%load": 100 * r["delta"] / mean_load,
                     "threshold_MW": DELTA_FRAC * ref,
                     "ci_halfwidth_MW": (r["ci_high"] - r["ci_low"]) / 2,
                     "months_A_better": f"{int((by_m['ea'] < by_m['eb']).sum())}/{len(by_m)}",
                     "verdict": verdict(r["delta"], r["ci_low"], r["ci_high"], ref)})
    return pd.DataFrame(rows)


def day_type(df: pd.DataFrame) -> pd.Series:
    t = np.select([df["hol_share"] >= 0.5, df["bridge"] == 1, df["dow"] == 6, df["dow"] == 5],
                  ["holiday", "bridge", "Sunday", "Saturday"], default="workday")
    return pd.Series(t, index=df.index)


def breakdown(df: pd.DataFrame, by: pd.Series | str, pred_cols: list[str]) -> pd.DataFrame:
    key = df[by] if isinstance(by, str) else by
    err = pd.DataFrame({c: (df["y"] - df[c]).abs() for c in pred_cols})
    out = err.groupby(key).mean()
    out["n_hours"] = err.groupby(key).size()
    return out
