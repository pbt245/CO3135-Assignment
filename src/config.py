"""Project-wide constants and paths.

Every "why" for a constant is written next to it, because the constants ARE the
experimental design (split dates, information cutoff, seeds).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- reproducibility
# One seed for numpy, Optuna's sampler, LightGBM and the bootstrap, so that a rerun
# with the same package versions gives the same numbers.
SEED = 42
# LightGBM thread count is fixed because the order of floating-point sums (and so the
# result) can depend on it.
N_THREADS = 4

# Smoke mode: tiny Optuna budget so the pipeline can be checked end-to-end on a laptop.
# Real runs (Kaggle) leave it unset.
SMOKE = os.environ.get("P4AIDS_SMOKE", "0") == "1"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- time handling
# Demand follows the local wall clock (people wake up at 07:00 local, not UTC), so every
# feature is built on Europe/Madrid local time.
TZ = "Europe/Madrid"

# ---------------------------------------------------------------- splits
# Train 2015-2016 (2 full years -> 2 summers, 2 winters), validation 2017 (one full
# year, so every season is represented when choosing hyper-parameters), test 2018
# (one full year, touched once after the configuration is frozen).
TRAIN_START = "2015-01-01"
VAL_START = "2017-01-01"
TEST_START = "2018-01-01"
TEST_END = "2019-01-01"  # exclusive

# ---------------------------------------------------------------- information cutoff
# The forecast for every hour of day D is issued at 10:00 local on D-1 (2 h before the
# 12:00 day-ahead gate closure). The last load/weather hour surely published by then is
# the hour stamped 08 on D-1: it ends at 09:00 if stamps are interval starts (what the
# solar peak suggests) and at 08:00 if they are interval ends -- safe either way.
LAST_KNOWN_HOUR_D1 = 8

# ---------------------------------------------------------------- weather
CITIES = ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"]
# Two weather settings, evaluated separately and never mixed inside one comparison:
#   "actual": observed weather of day D (= a perfect weather forecast) -> optimistic bound
#   "lagged": only weather observed up to D-1 08:00 -> what is achievable without a
#             weather-forecast feed. The truth with a real forecast lies in between.
WEATHER_MODES = ("actual", "lagged")

# ---------------------------------------------------------------- data cleaning
# A one-hour value that deviates > SPIKE_THR from the mean of its two neighbours while
# the neighbours agree within SPIKE_NB_TOL is a recording error (a 12 GW national drop
# that recovers within the hour is not physical). 15%, 20% and 25% flag the same 4
# hours, so the rule is not sensitive to the exact threshold.
SPIKE_THR = 0.15
SPIKE_NB_TOL = 0.10
# Missing load hours used as *lag inputs* are forward-filled for at most 3 h (uses the
# past only). Longer gaps stay missing and the affected training rows are dropped.
FFILL_LIMIT = 3

# ---------------------------------------------------------------- Optuna budget
N_TRIALS = 3 if SMOKE else 30
LGB_MAX_ROUNDS = 300 if SMOKE else 5000
LGB_EARLY_STOP = 50 if SMOKE else 200
LGB_LEARNING_RATE = 0.03

# ---------------------------------------------------------------- paths
ON_KAGGLE = Path("/kaggle/input").exists()
ROOT = Path(__file__).resolve().parents[1]


def _find_raw_dir() -> Path:
    if ON_KAGGLE:
        hits = sorted(Path("/kaggle/input").rglob("energy_dataset.csv"))
        if not hits:
            raise FileNotFoundError("Attach the Kaggle dataset "
                                    "nicholasjhana/energy-consumption-generation-prices-and-weather")
        return hits[0].parent
    return ROOT / "data" / "raw"


RAW_DIR = _find_raw_dir()
EXTERNAL_DIR = ROOT / "data" / "external"  # ships with the code (holiday table)
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else ROOT
# Smoke/dry runs can be redirected so that they never overwrite the real results/.
if os.environ.get("P4AIDS_OUT"):
    OUT_DIR = Path(os.environ["P4AIDS_OUT"])
PROCESSED_DIR = OUT_DIR / "data" / "processed"
RESULTS_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
# Frozen configuration = hyper-parameters chosen on validation. It is written by
# notebooks 02/03 and must be committed to git before notebook 04 (test) is run.
FROZEN_DIR = RESULTS_DIR / "frozen"
# Second frozen set: the extra models added after the first test run (lecturer's feedback; scripts 05/06).
# Kept in its own folder so the original frozen files and their hashes are never touched.
FROZEN_V2_DIR = RESULTS_DIR / "frozen_v2"


def frozen_search_dirs(sub: str = "frozen") -> list[Path]:
    """Where the test scripts look for a frozen config (Kaggle: output dir, then the code dataset)."""
    marker = "ridge_actual.json" if sub == "frozen" else "SHA256SUMS"
    dirs = [RESULTS_DIR / sub, ROOT / "results" / sub]
    if ON_KAGGLE:
        dirs += [p.parent for p in Path("/kaggle/input").glob(f"**/{sub}/{marker}")]
    out = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


for _d in (PROCESSED_DIR, RESULTS_DIR, FIG_DIR, FROZEN_DIR, FROZEN_V2_DIR):
    _d.mkdir(parents=True, exist_ok=True)
