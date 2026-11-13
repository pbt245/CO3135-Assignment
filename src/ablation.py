"""Extra models added after the first test run, on the lecturer's feedback (scripts 05 and 06).

Why. Two claims of the first report were too broad:
  Q1  "B0 -> B1" changed two things at once: from a fixed rule to a FITTED model, and from minimal inputs to
      ENGINEERED features. The gain cannot be credited to feature engineering alone.
  Q2  B1 is 24 per-hour Ridge models, M1 is one global LightGBM, so "M1 vs B1" also compares structures.
Fix. A grid of model x feature set x structure, every cell tuned with the SAME procedure, budget and seed as
the original models, so that each question is answered by a pair that differs in ONE thing only.

| Name              | Model    | Features | Structure | Status                         |
|-------------------|----------|----------|-----------|--------------------------------|
| ridge_hour_F1     | Ridge    | F1       | 24 models | = original B1 (frozen/)        |
| lgbm_global_F1    | LightGBM | F1       | 1 model   | = original M1 (frozen/)        |
| ridge_hour_F0     | Ridge    | F0       | 24 models | new (frozen_v2/)               |
| ridge_global_F0   | Ridge    | F0       | 1 model   | new                            |
| ridge_global_F1   | Ridge    | F1       | 1 model   | new                            |
| lgbm_global_F0    | LightGBM | F0       | 1 model   | new                            |
| lgbm_hour_F1      | LightGBM | F1       | 24 models | new                            |
F0 / F1: see features.F0. Each model exists in both weather settings (column name + "_actual" / "_lagged").
"""
from __future__ import annotations

# name -> (model family, feature set, per_hour)
NEW = {
    "ridge_hour_F0": ("ridge", "F0", True),
    "ridge_global_F0": ("ridge", "F0", False),
    "ridge_global_F1": ("ridge", "F1", False),
    "lgbm_global_F0": ("lgbm", "F0", False),
    "lgbm_hour_F1": ("lgbm", "F1", True),
}
# the two original models under their grid names
ORIGINAL = {"B1": "ridge_hour_F1", "M1": "lgbm_global_F1"}
ALL = ["ridge_hour_F0", "ridge_global_F0", "ridge_hour_F1", "ridge_global_F1",
       "lgbm_global_F0", "lgbm_global_F1", "lgbm_hour_F1"]

# (A, B, question): Delta = MAE(A) - MAE(B); the threshold delta uses MAE(B), so B is the reference
# (the simpler or the original model of the pair). Each pair differs in exactly one thing.
PAIRS = [
    # Q1: the step from B0, split into its two parts
    ("ridge_hour_F0", "B0", "Q1 fitting: Ridge on B0's information vs the fixed rule"),
    ("ridge_hour_F1", "ridge_hour_F0", "Q1 features: engineered vs minimal, per-hour Ridge"),
    ("ridge_global_F1", "ridge_global_F0", "Q1 features: engineered vs minimal, global Ridge"),
    ("lgbm_global_F1", "lgbm_global_F0", "Q1 features: engineered vs minimal, LightGBM"),
    # Q2: LightGBM vs Ridge at the same features AND the same structure
    ("lgbm_global_F1", "ridge_global_F1", "Q2 model: one model each, F1"),
    ("lgbm_hour_F1", "ridge_hour_F1", "Q2 model: 24 models each, F1"),
    ("lgbm_global_F0", "ridge_global_F0", "Q2 model: one model each, F0"),
    ("lgbm_global_F1", "ridge_hour_F1", "Q2 original: each model in its own structure (M1 vs B1)"),
    # structure: same model and features, 24 models vs 1
    ("ridge_hour_F1", "ridge_global_F1", "structure: 24 vs 1, Ridge"),
    ("lgbm_hour_F1", "lgbm_global_F1", "structure: 24 vs 1, LightGBM"),
]


def col(name: str, mode: str) -> str:
    return "B0" if name == "B0" else f"{name}_{mode}"


def pairs_for(mode: str) -> list[tuple[str, str]]:
    return [(col(a, mode), col(b, mode)) for a, b, _ in PAIRS]


def question_of(a_col: str, b_col: str) -> str:
    strip = lambda c: c.rsplit("_", 1)[0] if c != "B0" else c
    return next(q for a, b, q in PAIRS if a == strip(a_col) and b == strip(b_col))


def rename_original(df, modes):
    """B1_<mode> / M1_<mode> columns of the saved prediction files -> grid names."""
    return df.rename(columns={f"{k}_{m}": f"{v}_{m}" for k, v in ORIGINAL.items() for m in modes})
