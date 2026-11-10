"""Load and clean the raw Kaggle files into one hourly panel on a local-time grid.

Output: one row per (local date, local hour 0..23), columns
    ts, date, hour            local wall-clock time (tz-naive)
    y                         target load [MW]; NaN where not observed / implausible / DST gap
    y_src                     load used to build lag features (y forward-filled <= 3 h)
    tso_fc                    TSO's own day-ahead forecast (reference only, never a feature)
    T_<city>, T               temperature [deg C] per city and the 5-city mean
    dst_gap, spike, y_was_nan quality flags
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- raw readers
def read_energy(raw_dir=C.RAW_DIR) -> pd.DataFrame:
    e = pd.read_csv(raw_dir / "energy_dataset.csv", usecols=["time", "total load actual",
                                                              "total load forecast"])
    # `time` carries an explicit +01:00/+02:00 offset -> parse as UTC, convert later.
    e["ts_utc"] = pd.to_datetime(e["time"], utc=True)
    e = e.rename(columns={"total load actual": "y", "total load forecast": "tso_fc"})
    # generation/price/other forecast columns are dropped on purpose: they are either
    # simultaneous with the target or published after the 10:00 D-1 cutoff (leakage).
    return e.set_index("ts_utc")[["y", "tso_fc"]].sort_index()


def read_weather(raw_dir=C.RAW_DIR) -> tuple[pd.DataFrame, dict]:
    w = pd.read_csv(raw_dir / "weather_features.csv", usecols=["dt_iso", "city_name", "temp"])
    w["city_name"] = w["city_name"].str.strip()  # " Barcelona" has a leading space
    w["ts_utc"] = pd.to_datetime(w["dt_iso"], utc=True)
    n_dup = int(w.duplicated(["city_name", "ts_utc"]).sum())
    # Duplicates were checked to carry identical temperatures, so keeping the first is
    # lossless; asserted here so that a different file version cannot slip through.
    n_conflict = int((w.groupby(["city_name", "ts_utc"])["temp"].nunique() > 1).sum())
    assert n_conflict == 0, f"{n_conflict} duplicated (city, hour) with different temps"
    w = w.drop_duplicates(["city_name", "ts_utc"])
    wide = w.pivot(index="ts_utc", columns="city_name", values="temp")[C.CITIES] - 273.15  # K -> C
    wide.columns = [f"T_{c}" for c in wide.columns]
    info = {"weather_rows_raw": len(w) + n_dup, "weather_dup_rows_removed": n_dup,
            "weather_conflicting_dups": n_conflict}
    return wide.sort_index(), info


# ---------------------------------------------------------------- cleaning steps
def flag_spikes(y: pd.Series, thr: float = C.SPIKE_THR, nb_tol: float = C.SPIKE_NB_TOL) -> pd.Series:
    """Isolated one-hour spikes/dips (on the continuous UTC series, so neighbours are real hours)."""
    nb_mean = (y.shift(1) + y.shift(-1)) / 2
    dev = (y - nb_mean).abs() / nb_mean
    nb_gap = (y.shift(1) - y.shift(-1)).abs() / nb_mean
    return (dev > thr) & (nb_gap < nb_tol)


def to_local_grid(df_utc: pd.DataFrame) -> pd.DataFrame:
    """Re-index a UTC hourly frame on a complete (local date x hour 0..23) grid.

    DST: the spring-forward day lacks local hour 02 -> the row is created (dst_gap=1)
    and left NaN here; the autumn day has hour 02 twice -> the two values are averaged.
    A fixed 24-slot day lets lags be looked up as "same local hour, k days ago" instead
    of shift(24*k), which would be off by one hour for a week after each clock change.
    """
    loc = df_utc.index.tz_convert(C.TZ)
    key_date = loc.tz_localize(None).normalize()
    g = df_utc.groupby([key_date, loc.hour]).mean()
    g.index.names = ["date", "hour"]
    dates = pd.date_range(key_date.min(), key_date.max(), freq="D")
    full = pd.MultiIndex.from_product([dates, range(24)], names=["date", "hour"])
    present = full.isin(g.index)
    g = g.reindex(full)
    g["dst_gap"] = (~present).astype(int)
    return g.reset_index()


def build_panel(end: str | None = C.TEST_START, raw_dir=C.RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Clean panel. `end` (local, exclusive) cuts the raw data BEFORE any processing.

    Development notebooks call it with end=TEST_START so that the 2018 test year cannot
    influence anything (not even a forward-fill); notebook 04 calls it with end=None.
    """
    e = read_energy(raw_dir)
    w, info = read_weather(raw_dir)
    df = e.join(w, how="outer")
    if end is not None:
        local = df.index.tz_convert(C.TZ).tz_localize(None)
        df = df[local < pd.Timestamp(end)]

    info["hours_utc"] = len(df)
    info["y_nan_raw"] = int(df["y"].isna().sum())
    spike = flag_spikes(df["y"])
    info["spikes"] = [str(t.tz_convert(C.TZ)) for t in df.index[spike]]
    df["y_was_nan"] = df["y"].isna().astype(int)
    df["spike"] = spike.astype(int)
    df.loc[spike, "y"] = np.nan  # implausible -> treated as missing (not trained on, not scored)

    p = to_local_grid(df)
    p["ts"] = p["date"] + pd.to_timedelta(p["hour"], unit="h")
    p[["spike", "y_was_nan"]] = p[["spike", "y_was_nan"]].fillna(0).astype(int)
    tcols = [f"T_{c}" for c in C.CITIES]
    # Weather has no gaps except the DST slot; fill it from the previous hour (past only).
    p[tcols] = p[tcols].ffill()
    # Simple 5-city mean: needs no external data, and Ridge and LightGBM see the same
    # variable. It may not represent national demand (no population weights) -> limitation.
    p["T"] = p[tcols].mean(axis=1)
    # Lag source: forward fill only (never uses the future), at most FFILL_LIMIT hours.
    p["y_src"] = p["y"].ffill(limit=C.FFILL_LIMIT)

    info["dst_gap_rows"] = int(p["dst_gap"].sum())
    info["y_nan_after_clean"] = int(p["y"].isna().sum())
    info["y_src_nan_after_ffill"] = int(p["y_src"].isna().sum())
    info["first_ts"], info["last_ts"] = str(p["ts"].min()), str(p["ts"].max())
    cols = ["ts", "date", "hour", "y", "y_src", "tso_fc", *tcols, "T", "dst_gap", "spike", "y_was_nan"]
    return p[cols], info
