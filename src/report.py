"""Console reporting for the scripts: titled tables in the log + CSV copies in results/."""
from __future__ import annotations

import pandas as pd

from . import config as C

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def show(obj, title: str, csv: str | None = None) -> None:
    """Print a table under a title; optionally save it as results/<csv>."""
    print(f"\n--- {title} ---")
    print(obj.to_string() if hasattr(obj, "to_string") else obj, flush=True)
    if csv is not None:
        obj.to_csv(C.RESULTS_DIR / csv)


def find_result(name: str):
    """A results/ file from this run's output dir, else the copy shipped with the code."""
    for d in (C.RESULTS_DIR, C.ROOT / "results"):
        if (d / name).exists():
            return d / name
    return None
