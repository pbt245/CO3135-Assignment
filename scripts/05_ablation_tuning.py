"""05. Thí nghiệm bổ sung (sau lần chạy test đầu tiên, theo góp ý của giảng viên): tune trên validation 2017.

VÌ SAO. Hai kết luận của báo cáo đầu nói rộng hơn những gì thí nghiệm đo được:
  Q1  B0 -> B1 đổi HAI thứ cùng lúc: từ quy tắc cố định sang model ĐƯỢC FIT, và từ thông tin tối thiểu sang feature
      ĐÃ ENGINEER. Không thể gán toàn bộ phần giảm sai số cho feature engineering.
  Q2  B1 là 24 Ridge theo giờ, M1 là 1 LightGBM toàn cục, nên M1 vs B1 còn so cả cấu trúc, không chỉ tuyến tính vs cây.
CÁCH SỬA. Lưới model x bộ feature x cấu trúc (src/ablation.py). Mỗi câu hỏi được trả lời bằng một cặp chỉ khác nhau
MỘT thứ. Model mới: ridge_hour_F0, ridge_global_F0, ridge_global_F1, lgbm_global_F0, lgbm_hour_F1.

| F0 (tối thiểu) | giờ, thứ, tháng, nhiệt độ THÔ của cấu hình, lag_d7 | chỉ những gì dữ liệu có sẵn; lag_d7 = đúng thông tin B0 dùng,
|                |                                                   | nên "Ridge F0 vs B0" tách riêng giá trị của việc FIT model |
| F1 (đầy đủ)    | bộ feature gốc (lag, lịch, ngày lễ, thời tiết)     | như B1/M1 |
| global         | 1 model cho 24 giờ; Ridge nhận giờ dạng one-hot, KHÔNG tương tác với giờ | cùng cấu trúc với M1 |
| theo giờ       | 24 model; LightGBM 24 model chung siêu tham số, mỗi giờ early stopping riêng | cùng cấu trúc với B1 |

QUY TRÌNH Y HỆT MODEL GỐC (để đặt cạnh nhau trong một bảng so sánh): train 2015-16, chọn trên validation 2017 theo MAE,
riêng từng cấu hình thời tiết; Ridge: cùng lưới base HDD/CDD (chỉ F1) và alpha, cùng hai dạng target; LightGBM: cùng
không gian Optuna, 30 trial, TPE seed 42, early stopping 200 vòng, learning rate 0.03. Tham số chốt ghi vào
results/frozen_v2/ kèm SHA256SUMS; results/frozen/ của model gốc KHÔNG bị đụng tới.

GHI CHÚ TRUNG THỰC. Các model này được thêm SAU khi đã thấy kết quả test 2018. Cấu hình vẫn chọn hoàn toàn trên
validation và đóng băng trước khi nạp 2018 (script 06, chạy một lần), nhưng báo cáo phải ghi rõ đây là phân tích bổ sung.

CHẠY: Kaggle (Optuna cho hai biến thể LightGBM). Trên máy: P4AIDS_SMOKE=1 P4AIDS_OUT=<thư mục tạm>.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from src import config as C, data, features as F, models as M, evaluation as E, ablation as A
from src.report import section, show, find_result

C.seed_everything()
print("ROOT:", ROOT, "| Kaggle:", C.ON_KAGGLE, "| smoke:", C.SMOKE, "| output:", C.OUT_DIR,
      "| lightgbm", lgb.__version__, "| optuna", optuna.__version__)

panel, _ = data.build_panel(end=C.TEST_START)
feats = {m: (lambda f: f[f["usable"]].reset_index(drop=True))(F.build_features(panel, m)) for m in C.WEATHER_MODES}
for f in feats.values():
    assert (f["split"] != "test").all(), "test rows must not exist in development scripts"
split = {m: (f[f["split"] == "train"], f[f["split"] == "val"]) for m, f in feats.items()}

# ------------------------------------------------------------------ 0. original models on validation
# Rebuilt from results/frozen/ (cheap, deterministic). If the saved predictions of script 03 are available, they must
# match, which proves that the code changes for the new variants did not alter B1 or M1.
section("0. Original models B0, B1, M1 on validation (rebuilt from results/frozen/)")
frozen = {}
for name in ["ridge_actual", "ridge_lagged", "lgbm_actual", "lgbm_lagged"]:
    hit = next((d / f"{name}.json" for d in C.frozen_search_dirs() if (d / f"{name}.json").exists()), None)
    assert hit is not None, f"frozen/{name}.json missing"
    frozen[name] = json.load(open(hit))
tr0, va0 = split["actual"]
val = va0[["ts", "date", "hour", "y", "dow", "month", "hol_share", "bridge", "T"]].copy()
val["B0"] = M.predict_b0(va0)
for mode in C.WEATHER_MODES:
    tr, va = split[mode]
    assert np.array_equal(va["ts"].to_numpy(), val["ts"].to_numpy())
    val[A.col("ridge_hour_F1", mode)] = M.ridge_from_params(frozen[f"ridge_{mode}"]).fit(tr).predict(va)
    # exactly the frozen number of trees; the first n trees are the same as with early stopping in script 03
    lp = frozen[f"lgbm_{mode}"]
    val[A.col("lgbm_global_F1", mode)] = M.lgbm_from_params(lp).fit(tr, M.lgbm_rounds(lp)).predict(va)
saved = find_result("val_predictions.parquet")
if saved is not None:
    old = A.rename_original(pd.read_parquet(saved), C.WEATHER_MODES)
    for c in [A.col(n, m) for n in ("ridge_hour_F1", "lgbm_global_F1") for m in C.WEATHER_MODES]:
        assert np.allclose(old[c], val[c], rtol=0, atol=1e-6), f"{c} differs from script 03"
    print("B1 and M1 identical to the saved validation predictions of script 03")

# ------------------------------------------------------------------ 1. new Ridge variants
section("1. New Ridge variants: tuning on validation (same procedure as B1)")
params, tables = {}, {}
for name, (fam, fset, per_hour) in A.NEW.items():
    if fam != "ridge":
        continue
    for mode in C.WEATHER_MODES:
        tr, va = split[mode]
        p, base_tab, alpha_tab = M.tune_ridge(tr, va, mode, fset=fset, per_hour=per_hour)
        params[f"{name}_{mode}"], tables[f"{name}_{mode}"] = p, alpha_tab
        val[A.col(name, mode)] = M.ridge_from_params(p).fit(tr).predict(va)
        alpha_tab.to_csv(C.RESULTS_DIR / f"ablation_alpha_grid_{name}_{mode}.csv", index=False)
        if len(base_tab):
            base_tab.to_csv(C.RESULTS_DIR / f"ablation_base_grid_{name}_{mode}.csv", index=False)
        print(f"{name:16s} {mode:6s} target={p['target']:6s} th={p['th']} tc={p['tc']} alpha={p['alpha']:.3g} "
              f"val MAE={p['val_mae']:.1f}", flush=True)

# ------------------------------------------------------------------ 2. new LightGBM variants
section(f"2. New LightGBM variants: Optuna, {C.N_TRIALS} trials each (same search space and seed as M1)")
for name, (fam, fset, per_hour) in A.NEW.items():
    if fam != "lgbm":
        continue
    for mode in C.WEATHER_MODES:
        tr, va = split[mode]
        t0 = time.time()
        p, trials = M.tune_lgbm(tr, va, mode, fset=fset, per_hour=per_hour)
        params[f"{name}_{mode}"] = p
        trials.to_csv(C.RESULTS_DIR / f"ablation_optuna_{name}_{mode}.csv", index=False)
        m = M.lgbm_from_params(p).fit(tr, C.LGB_MAX_ROUNDS, va)  # same early stopping as in tuning
        assert m.best_iteration == M.lgbm_rounds(p)
        val[A.col(name, mode)] = m.predict(va)
        print(f"{name:16s} {mode:6s} target={p['target']:6s} {p['lgb_params']} val MAE={p['val_mae']:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ 3. comparison on validation
# Same scoring as the original study: Delta = MAE(A) - MAE(B), weekly block bootstrap, threshold 2% of MAE(B).
section("3. Validation comparison (test not touched)")
cols = ["B0"] + [A.col(n, m) for m in C.WEATHER_MODES for n in A.ALL]
assert val[cols].notna().all().all()
show(E.metric_table(val, cols), "validation 2017 metrics, all models", csv="val_ablation_metrics.csv")
mae_grid = pd.DataFrame({m: {n: M.mae(val["y"], val[A.col(n, m)]) for n in A.ALL} for m in C.WEATHER_MODES})
show(mae_grid.round(0), "validation MAE (MW) by model and weather setting")

cmp = pd.concat([E.compare_pairs(val, A.pairs_for(m)).assign(weather=m) for m in C.WEATHER_MODES], ignore_index=True)
cmp["question"] = [A.question_of(a, b) for a, b in zip(cmp["A"], cmp["B"])]
cmp.to_csv(C.RESULTS_DIR / "val_ablation_comparison.csv", index=False)
show(cmp[["weather", "question", "MAE_A", "MAE_B", "delta", "ci_low", "ci_high", "threshold_MW", "months_A_better",
          "verdict"]], "paired comparisons on validation (delta < 0: A better)")
val.to_parquet(C.RESULTS_DIR / "val_ablation_predictions.parquet", index=False)

# ------------------------------------------------------------------ 4. freeze
# Each JSON is written once, then hashed. Script 06 refuses to run if any file differs from this manifest.
section("4. Freeze the new configuration")
for k, p in params.items():
    (C.FROZEN_V2_DIR / f"{k}.json").write_text(json.dumps(p, indent=2))
lines = [f"{hashlib.sha256((C.FROZEN_V2_DIR / f'{k}.json').read_bytes()).hexdigest()}  {k}.json" for k in sorted(params)]
(C.FROZEN_V2_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nfrozen in {C.FROZEN_V2_DIR}. Next: copy results/frozen_v2/ into the repo, commit it, then run script 06 ONCE.")
