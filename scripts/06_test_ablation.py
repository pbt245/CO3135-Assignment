"""06. Test 2018 cho các model bổ sung của script 05 (MỘT LẦN). Model gốc B0/B1/M1 KHÔNG chạy lại.

GIAO THỨC (giống script 04):
  - Tham số đọc từ results/frozen_v2/ và phải khớp SHA256SUMS ghi lúc đóng băng (script 05), file đó phải được commit
    TRƯỚC khi chạy script này. Từ chối tham số smoke và từ chối chạy lần hai (results/test_ablation_run_log.json).
  - Mỗi model fit lại MỘT LẦN trên 2015-2017 với tham số và số cây cố định, rồi dự báo cả năm 2018.
  - Dự báo của B0/B1/M1 lấy từ results/test_predictions.parquet của lần chạy test gốc (không fit lại), nên các con số
    gốc trong báo cáo không đổi.
GHI CHÚ TRUNG THỰC: đây là lần nạp 2018 thứ hai của dự án, cho các model được thêm sau khi đã thấy kết quả test gốc
(theo góp ý của giảng viên). Báo cáo phải ghi rõ điều này.

P4AIDS_DRYRUN=1: 2017 đóng vai "test" (fit 2015-2016), KHÔNG nạp 2018, để kiểm tra code.
"""
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src import config as C, data, features as F, models as M, evaluation as E, ablation as A
from src.report import section, show, find_result

C.seed_everything()
DRY = os.environ.get("P4AIDS_DRYRUN", "0") == "1"
END = C.TEST_START if DRY else None            # dry run: 2018 is never loaded
FIT_END = C.VAL_START if DRY else C.TEST_START  # final models are fitted on dates < FIT_END
EVAL = "val" if DRY else "test"
TAG = "dryrun_ablation" if DRY else "test_ablation"
print("ROOT:", ROOT, "| Kaggle:", C.ON_KAGGLE, "| DRY RUN:", DRY, "| evaluating on:", EVAL, "| output:", C.OUT_DIR)

# ------------------------------------------------------------------ 0. preconditions
section("0. Preconditions")
names = [f"{n}_{m}" for n in A.NEW for m in C.WEATHER_MODES]
manifest = next((d / "SHA256SUMS" for d in C.frozen_search_dirs("frozen_v2") if (d / "SHA256SUMS").exists()), None)
assert manifest is not None, "frozen_v2/SHA256SUMS missing: run script 05 and commit results/frozen_v2/ first"
fdir = manifest.parent
expected = dict(line.split()[::-1] for line in manifest.read_text().splitlines() if line.strip())
frozen, frozen_sha = {}, {}
for k in names:
    path = fdir / f"{k}.json"
    assert path.exists(), f"{path} missing"
    frozen_sha[k] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert expected.get(f"{k}.json") == frozen_sha[k], f"{k}.json differs from the frozen manifest"
    frozen[k] = json.load(open(path))
print(f"{len(frozen)} frozen configurations match {manifest}")
smoke = [k for k, v in frozen.items() if v.get("smoke")]
assert DRY or not smoke, f"{smoke} come from a SMOKE run; rerun script 05 with the full budget"

orig_name = "val_predictions.parquet" if DRY else "test_predictions.parquet"
orig_path = find_result(orig_name)
assert orig_path is not None, f"{orig_name} of the original run is needed (B0/B1/M1 are not refitted)"
if not DRY:
    assert find_result("test_run_log.json") is not None, "the original test run (script 04) must come first"
    done = [p for p in (C.RESULTS_DIR / f"{TAG}_run_log.json", ROOT / "results" / f"{TAG}_run_log.json") if p.exists()]
    assert not done, f"Already run ({done[0]}). This test is run once."
run_started = dt.datetime.now().isoformat(timespec="seconds")

# ------------------------------------------------------------------ 1. data + consistency check (as in script 04)
section("1. Data and consistency with development data")
panel, info = data.build_panel(end=END)
panel_dev, _ = data.build_panel(end=C.TEST_START)
feats = {}
for mode in C.WEATHER_MODES:
    full = F.build_features(panel, mode)
    dev = F.build_features(panel_dev, mode)
    cols = F.feature_cols(mode) + ["y", "usable"]  # F1 contains every F0 column
    a = full.loc[full["date"] < pd.Timestamp(C.TEST_START), cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, dev[cols].reset_index(drop=True))
    feats[mode] = full[full["usable"]].reset_index(drop=True)
    print(mode, "consistent with development data;", feats[mode].groupby("split").size().to_dict())

# ------------------------------------------------------------------ 2. refit + predict the new models
section("2. Refit the new models on all data before the evaluation year and predict")
res = A.rename_original(pd.read_parquet(orig_path), C.WEATHER_MODES)
for mode in C.WEATHER_MODES:
    f = feats[mode]
    tr, ev = f[f["date"] < pd.Timestamp(FIT_END)], f[f["split"] == EVAL]
    assert np.array_equal(ev["ts"].to_numpy(), res["ts"].to_numpy()), "rows differ from the original run"
    assert np.allclose(ev["y"].to_numpy(), res["y"].to_numpy())
    for name, (fam, _, _) in A.NEW.items():
        p = frozen[f"{name}_{mode}"]
        if fam == "ridge":
            res[A.col(name, mode)] = M.ridge_from_params(p).fit(tr).predict(ev)
        else:
            res[A.col(name, mode)] = M.lgbm_from_params(p).fit(tr, M.lgbm_rounds(p)).predict(ev)
    print(f"{mode}: fitted on {len(tr)} rows, predicting {len(ev)} rows")
COLS = ["B0"] + [A.col(n, m) for m in C.WEATHER_MODES for n in A.ALL]
assert res[COLS].notna().all().all()
res.to_parquet(C.RESULTS_DIR / f"{TAG}_predictions.parquet", index=False)

# ------------------------------------------------------------------ 3. results
section(f"3. Results on {EVAL}")
metrics = E.metric_table(res, COLS)
show(metrics, f"{EVAL} metrics, all models", csv=f"{TAG}_metrics.csv")
mae_grid = pd.DataFrame({m: {n: metrics.loc[A.col(n, m), "MAE"] for n in A.ALL} for m in C.WEATHER_MODES})
show(mae_grid.round(0), f"{EVAL} MAE (MW) by model and weather setting")
cmp = pd.concat([E.compare_pairs(res, A.pairs_for(m)).assign(weather=m) for m in C.WEATHER_MODES], ignore_index=True)
cmp["question"] = [A.question_of(a, b) for a, b in zip(cmp["A"], cmp["B"])]
cmp.to_csv(C.RESULTS_DIR / f"{TAG}_comparison.csv", index=False)
show(cmp[["weather", "question", "MAE_A", "MAE_B", "delta", "ci_low", "ci_high", "delta_%load", "threshold_MW",
          "months_A_better", "verdict"]], "paired comparisons (delta < 0: A better)")

# where does the model choice matter, at matched structure? (same breakdown as script 04, section 5)
res["day_type"] = E.day_type(res)
res["T_band"] = pd.cut(res["T"], [-10, 5, 10, 15, 20, 25, 45], labels=["<5", "5-10", "10-15", "15-20", "20-25", ">25"])
for a, b in [("lgbm_global_F1", "ridge_global_F1"), ("lgbm_hour_F1", "ridge_hour_F1")]:
    g = {m: (res["y"] - res[A.col(a, m)]).abs() - (res["y"] - res[A.col(b, m)]).abs() for m in C.WEATHER_MODES}
    gd = pd.DataFrame(g)
    show(pd.concat({"day_type": gd.groupby(res["day_type"]).mean(),
                    "T_band": gd.groupby(res["T_band"], observed=True).mean()}).round(0),
         f"mean |e_{a}| - |e_{b}| (MW, negative = LightGBM better)", csv=f"{TAG}_gain_{a}_vs_{b}.csv")

import platform
import sklearn
log = {"run": TAG, "started": run_started, "finished": dt.datetime.now().isoformat(timespec="seconds"),
       "note": "second loading of 2018, for models added after the original test run (lecturer feedback)",
       "frozen_v2": frozen, "frozen_v2_sha256": frozen_sha, "original_predictions": str(orig_path),
       "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__, "sklearn": sklearn.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__},
       "mae": metrics["MAE"].to_dict()}
json.dump(log, open(C.RESULTS_DIR / f"{TAG}_run_log.json", "w"), indent=2, default=str)
print(f"\nsaved results for {TAG} in {C.RESULTS_DIR}")
