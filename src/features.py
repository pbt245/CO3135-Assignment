"""Feature engineering on the clean local-time panel.

Every feature of target hour (D, h) only uses information available at the issue time
10:00 on D-1 -- except, in weather mode "actual", the temperature of day D itself
(that mode deliberately assumes a perfect weather forecast; see config.WEATHER_MODES).
`assert_no_lookahead` checks this automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

H8 = C.LAST_KNOWN_HOUR_D1

# ---------------------------------------------------------------- feature groups
LAG = ["lag_d2", "lag_d7", "lag_d1_h8", "ratio_morning"]
CAL = ["hour", "dow", "month", "doy_sin", "doy_cos"]
HOL = ["hol_nat", "hol_share", "bridge", "xmas", "hol_share_d1", "hol_share_d2", "hol_share_d7"]
WX = {
    # actual weather of the target hour + its recent history (thermal inertia of buildings)
    "actual": ["T", "T_ma24"],
    # newest observation at the cutoff, mean of the 24 h before the cutoff, and the same
    # hour two days ago (the latest same-hour value that is surely known)
    "lagged": ["T_d1_h8", "T_ma24_b", "T_d2_h"],
}
GROUPS = {"load lags": LAG, "calendar": CAL, "holidays": HOL, "weather": WX["actual"] + WX["lagged"]}

# F0 = minimal inputs, no feature engineering (added after the first test run, on the lecturer's feedback, to
# separate "fitting a model" from "engineering features"). Only what the dataset gives directly: hour, weekday,
# month, the raw temperature of the setting, and lag_d7 -- exactly the information the seasonal naive B0 uses, so
# that "Ridge F0 vs B0" isolates the value of fitting. No degree days, holidays, D-2 / D-1 lags, morning ratio or
# cyclic encodings. F1 = the full engineered set above.
F0 = {"actual": ["hour", "dow", "month", "lag_d7", "T"],
      "lagged": ["hour", "dow", "month", "lag_d7", "T_d1_h8"]}
FEATURE_SETS = ("F0", "F1")


def feature_cols(mode: str, fset: str = "F1") -> list[str]:
    assert fset in FEATURE_SETS
    return F0[mode] if fset == "F0" else LAG + CAL + HOL + WX[mode]


def _day_matrix(p: pd.DataFrame, col: str) -> pd.DataFrame:
    """date x hour matrix; dates are a complete daily range, so shift(k) == k days ago."""
    m = p.pivot(index="date", columns="hour", values=col)
    assert m.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()
    return m


def _stack(m: pd.DataFrame, name: str) -> pd.Series:
    """date x hour matrix -> Series indexed by (date, hour). melt keeps NaN and works the
    same on pandas 2.x (Kaggle) and 3.x (local), unlike DataFrame.stack."""
    s = m.rename_axis(columns="hour").reset_index().melt(id_vars="date", var_name="hour", value_name=name)
    s["hour"] = s["hour"].astype(int)
    return s.set_index(["date", "hour"])[name]


def _broadcast(day_series: pd.Series) -> pd.DataFrame:
    """Day-level value -> same value for all 24 hours of that day."""
    return pd.DataFrame({h: day_series for h in range(24)})


def load_holiday_table() -> pd.DataFrame:
    hol = pd.read_csv(C.EXTERNAL_DIR / "holidays_es.csv", parse_dates=["date"])
    return hol.set_index("date")[["hol_nat", "hol_share"]]


def build_features(p: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return panel + features for one weather mode (rows aligned with `p`)."""
    assert mode in C.WEATHER_MODES
    out = p.set_index(["date", "hour"]).copy()

    # ---- load lags (looked up by local day/hour, never by a shift in absolute hours)
    Y = _day_matrix(p, "y_src")
    out = out.join(_stack(Y.shift(2), "lag_d2")).join(_stack(Y.shift(7), "lag_d7"))
    # newest load hour known at 10:00 D-1: tells the model the *current* level, which a
    # 2-day-old lag misses after a sudden change (cold snap, start of holidays)
    out = out.join(_stack(_broadcast(Y.shift(1)[H8]), "lag_d1_h8"))
    # D-1 morning vs D-2 morning (hours 0..8): is the level moving right now?
    morning = list(range(0, H8 + 1))
    ratio = Y.shift(1)[morning].mean(axis=1) / Y.shift(2)[morning].mean(axis=1)
    out = out.join(_stack(_broadcast(ratio), "ratio_morning"))

    # ---- calendar
    d = out.index.get_level_values("date")  # ("hour" is an index level -> a column after reset)
    out["dow"] = d.day_of_week
    out["month"] = d.month
    doy = 2 * np.pi * (d.dayofyear - 1) / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(doy), np.cos(doy)

    # ---- holidays (and whether the days used as lags were holidays: a lag taken from a
    # holiday is a bad reference, the flag lets the model discount it)
    hol = load_holiday_table()
    # +1 day: the bridge rule of the last day needs the next day's holiday status (holiday
    # calendars are published in advance, so this is not look-ahead). Found by the audit:
    # 2018-12-31 (Mon before the Tue 2019-01-01 holiday) was otherwise not flagged.
    days = pd.date_range(d.min() - pd.Timedelta("8D"), d.max() + pd.Timedelta("1D"), freq="D")
    hd = hol.reindex(days).fillna(0.0)
    # A day counts as a holiday when at least half of the regions are off (national days,
    # Holy Thursday); used only for the bridge rule below.
    big_hol = hd["hol_share"].values >= 0.5
    dow = np.asarray(days.day_of_week)
    workday = (dow < 5) & ~big_hol
    # bridge ("puente"): Monday before a Tuesday holiday / Friday after a Thursday holiday;
    # many people take it off, so the load looks like a weekend that seasonal lags miss.
    next_hol = np.r_[big_hol[1:], False]
    prev_hol = np.r_[False, big_hol[:-1]]
    bridge = workday & (((dow == 0) & next_hol) | ((dow == 4) & prev_hol))
    # Christmas-Epiphany window: schools closed, industry slows down for two weeks.
    xmas = ((days.month == 12) & (days.day >= 24)) | ((days.month == 1) & (days.day <= 6))
    dayf = pd.DataFrame({"hol_nat": hd["hol_nat"].values, "hol_share": hd["hol_share"].values,
                         "bridge": np.asarray(bridge, dtype=float), "xmas": np.asarray(xmas, dtype=float)},
                        index=days)
    for k in (1, 2, 7):
        dayf[f"hol_share_d{k}"] = dayf["hol_share"].shift(k)
    out = out.join(dayf, on="date")

    # ---- weather
    assert p["ts"].is_monotonic_increasing
    T = _day_matrix(p, "T")
    # trailing 24-hour mean including hour t (buildings react to the last day, not one hour)
    p_ma = p.assign(T_ma24=p["T"].rolling(24, min_periods=24).mean())
    if mode == "actual":
        out = out.join(p_ma.set_index(["date", "hour"])["T_ma24"])
    else:
        MA = _day_matrix(p_ma, "T_ma24")
        out = out.join(_stack(_broadcast(T.shift(1)[H8]), "T_d1_h8"))
        out = out.join(_stack(_broadcast(MA.shift(1)[H8]), "T_ma24_b"))
        out = out.join(_stack(T.shift(2), "T_d2_h"))

    # ---- split label
    out["split"] = np.select([d < pd.Timestamp(C.VAL_START), d < pd.Timestamp(C.TEST_START)],
                             ["train", "val"], default="test")
    out = out.reset_index()
    # usable = target observed, not a synthetic DST slot, all features present. Always judged on F1 (F0 is a
    # subset of F1), so every model and feature set is trained and scored on exactly the same hours.
    out["usable"] = out["y"].notna() & (out["dst_gap"] == 0) & out[feature_cols(mode)].notna().all(axis=1)
    return out


# ---------------------------------------------------------------- automatic leakage test
def assert_no_lookahead(p: pd.DataFrame, mode: str, n_days: int = 8, seed: int = C.SEED) -> pd.DataFrame:
    """Perturb everything after the cutoff (10:00 D-1) and check that the features of day D
    do not move. In mode "actual" the weather features of day D are exempt by design.
    """
    rng = np.random.default_rng(seed)
    base = build_features(p, mode)
    cand = p["date"].drop_duplicates().iloc[14:-1]
    days = pd.to_datetime(rng.choice(cand.values, size=n_days, replace=False))
    exempt = set(WX["actual"]) if mode == "actual" else set()
    cols = [c for c in feature_cols(mode) if c not in exempt]
    rows = []
    for D in days:
        cutoff = D - pd.Timedelta("1D") + pd.Timedelta(hours=H8)
        q = p.copy()
        after = q["ts"] > cutoff
        noise = rng.normal(0, 5000, after.sum())
        q.loc[after, "y_src"] = q.loc[after, "y_src"] + noise
        q.loc[after, "y"] = q.loc[after, "y"] + noise
        q.loc[after, "T"] = q.loc[after, "T"] + rng.normal(0, 10, after.sum())
        pert = build_features(q, mode)
        a, b = base.loc[base["date"] == D, cols], pert.loc[pert["date"] == D, cols]
        max_abs = float((a - b).abs().max().max())
        rows.append({"day": D.date(), "cutoff": cutoff, "max_abs_change": max_abs})
        assert np.allclose(a.values, b.values, equal_nan=True), f"look-ahead on {D.date()}"
    return pd.DataFrame(rows)
