"""03. M1: LightGBM + Optuna trên validation 2017. CHẠY TRÊN KAGGLE (30 trial x 2 cấu hình thời tiết).
Trên máy cá nhân chỉ smoke test với P4AIDS_SMOKE=1 (3 trial, tối đa 300 cây); tham số smoke bị script 04 từ chối.

THIẾT KẾ TUNING VÀ LÝ DO:
| Model       | Một LightGBM TOÀN CỤC cho 24 giờ (giờ là feature) | mỗi giờ chỉ ~730 mẫu train; cây cần nhiều mẫu mỗi nhánh. Model toàn cục vẫn tách theo giờ khi dữ liệu ủng hộ |
| Loss        | L2 (giống Ridge)                                 | so sánh về lớp hàm, không về loss; chọn và báo cáo bằng MAE ở cả hai |
| Feature     | như Ridge, nhiệt độ thô, không HDD/CDD            | cây bất biến với biến đổi đơn điệu của một biến, tự tìm ngưỡng |
| Optuna      | target {level, delta7}, num_leaves 8-256, min_data_in_leaf 10-300, feature_fraction 0.5-1, lambda_l2 1e-3-10 | núm độ phức tạp + regularization; dạng target cùng quy tắc như Ridge |
| LR          | cố định 0.03                                      | số cây thích nghi nhờ early stopping; tune cả hai là thừa |
| Số cây      | early stopping 200 vòng theo MAE validation, giữ cố định khi refit 2015-2017 | khi refit không còn validation để dừng; kế hoạch đã chốt |
| Ngân sách   | 30 trial TPE, seed cố định                        | ~2 năm train, tune nhiều thì overfit năm validation. Ridge có 94 lần thử (2 target x (36 base + 11 alpha)) -> công khai |
| Tái lập     | deterministic, force_row_wise, 4 thread, seed 42  | cùng phiên bản gói thì cùng kết quả |
Lưu ý: early stopping và chọn tham số dùng chung validation 2017 -> MAE validation của M1 hơi lạc quan. Ridge cũng chọn
trên chính validation đó. So sánh công bằng thật nằm ở test 2018.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from src import config as C, data, features as F, models as M, evaluation as E, plotting as P
from src.report import section, show, find_result

C.seed_everything(); P.setup()
print("ROOT:", ROOT, "| Kaggle:", C.ON_KAGGLE, "| smoke:", C.SMOKE, "| output:", C.OUT_DIR,
      "| lightgbm", lgb.__version__, "| optuna", optuna.__version__)

panel, _ = data.build_panel(end=C.TEST_START)
feats = {m: (lambda f: f[f["usable"]].reset_index(drop=True))(F.build_features(panel, m)) for m in C.WEATHER_MODES}
for f in feats.values():
    assert (f["split"] != "test").all()

ridge_params = {}
for mode in C.WEATHER_MODES:
    hit = next((d / f"ridge_{mode}.json" for d in C.frozen_search_dirs() if (d / f"ridge_{mode}.json").exists()), None)
    assert hit is not None, "Run script 02 first (frozen Ridge params missing)"
    ridge_params[mode] = json.load(open(hit))
    print(f"ridge_{mode} <- {hit}")

# ------------------------------------------------------------------ Optuna
section(f"1. Optuna: {C.N_TRIALS} trials per weather setting")
lgb_params, trials = {}, {}
for mode in C.WEATHER_MODES:
    f = feats[mode]
    tr, va = f[f["split"] == "train"], f[f["split"] == "val"]
    t0 = time.time()
    lgb_params[mode], trials[mode] = M.tune_lgbm(tr, va, mode)
    print(f"{mode}: {C.N_TRIALS} trials in {time.time() - t0:.0f}s, best val MAE {lgb_params[mode]['val_mae']:.1f}", flush=True)
show(pd.DataFrame({m: {**p["lgb_params"], "target": p["target"], "num_rounds": p["num_rounds"], "val_mae": p["val_mae"]}
                   for m, p in lgb_params.items()}), "best trial per setting")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
for ax, mode in zip(axes, C.WEATHER_MODES):
    t = trials[mode]
    for tgt, col in zip(M.TARGETS, [P.COLORS["B0"], P.COLORS["B1"]]):
        s = t[t["params_target"] == tgt]
        ax.scatter(s["number"], s["value"], s=14, color=col, label=f"target = {tgt}")
    ax.plot(t["number"], t["value"].cummin(), color=P.INK, lw=1.2, label="best so far")
    ax.set_title(f"Optuna trials ({mode})"); ax.set_xlabel("trial")
axes[0].set_ylabel("validation MAE (MW)"); axes[1].legend()
plt.tight_layout(); P.save(fig, "03_optuna_trials"); plt.close(fig)
# Nếu "best so far" phẳng từ nửa sau thì 30 trial là đủ; thêm trial chủ yếu khớp sâu hơn vào năm validation.

# ------------------------------------------------------------------ validation comparison (test not touched)
section("2. Validation comparison (test not touched)")
vp = find_result("val_predictions_baselines.parquet")
if vp is not None:
    val = pd.read_parquet(vp)
else:  # e.g. on Kaggle when script 02 ran elsewhere: recompute from the frozen params (cheap)
    f = feats["actual"]; va = f[f["split"] == "val"]
    val = va[["ts", "date", "hour", "y", "tso_fc", "dow", "month", "hol_share", "bridge", "T"]].copy()
    val["B0"], val["TSO"] = M.predict_b0(va), va["tso_fc"].to_numpy()
    for mode in C.WEATHER_MODES:
        f = feats[mode]
        val[f"B1_{mode}"] = M.ridge_from_params(ridge_params[mode]).fit(f[f["split"] == "train"]).predict(f[f["split"] == "val"])
for mode in C.WEATHER_MODES:
    f = feats[mode]
    tr, va = f[f["split"] == "train"], f[f["split"] == "val"]
    m = M.lgbm_from_params(lgb_params[mode]).fit(tr, C.LGB_MAX_ROUNDS, va)  # same early stopping as in tuning
    assert m.best_iteration == lgb_params[mode]["num_rounds"]
    assert np.array_equal(va["ts"].to_numpy(), val["ts"].to_numpy())
    val[f"M1_{mode}"] = m.predict(va)
cols = ["B0", "B1_actual", "M1_actual", "B1_lagged", "M1_lagged", "TSO"]
show(E.metric_table(val, cols), "validation 2017 metrics", csv="val_metrics.csv")

# "Tốt hơn rõ / xấp xỉ / không phân định" (chốt trước khi chạy test). Delta = MAE(A) - MAE(B), âm = A tốt hơn.
# CI 95% từ block bootstrap theo TUẦN (2.000 lần) trên hiệu sai số theo cặp (sai số giờ liền nhau tương quan mạnh;
# bootstrap từng giờ cho CI hẹp giả tạo). Ngưỡng delta = 2% MAE của model ĐƠN GIẢN hơn trong cặp (heuristic):
#   tốt hơn rõ: CI không chứa 0 và |Delta| >= delta;  xấp xỉ: CI nằm trọn trong [-delta, +delta];  còn lại: không phân định.
# Độ phân giải: nếu nửa độ rộng CI thường > delta thì phần lớn sẽ "không phân định" -> phải biết TRƯỚC khi chạy test.
pairs = [("B1_actual", "B0"), ("M1_actual", "B1_actual"), ("B1_lagged", "B0"), ("M1_lagged", "B1_lagged"),
         ("B1_actual", "B1_lagged"), ("M1_actual", "M1_lagged")]
cmp_val = E.compare_pairs(val, pairs)
show(cmp_val[["A", "B", "MAE_A", "MAE_B", "delta", "ci_low", "ci_high", "threshold_MW", "ci_halfwidth_MW",
              "months_A_better", "verdict"]], "paired comparison on validation")
cmp_val.to_csv(C.RESULTS_DIR / "val_comparison.csv", index=False)

# Phần dư lệch thế nào? Cả hai model tối ưu L2 (trung bình) còn MAE ứng với trung vị. Kiểm tra 0 fit: MAE giảm bao
# nhiêu nếu trừ trung vị phần dư. Chỉ là chẩn đoán, KHÔNG dùng để hiệu chỉnh dự báo.
diag = []
for c in ["B1_actual", "M1_actual", "B1_lagged", "M1_lagged"]:
    r = val["y"] - val[c]
    diag.append({"model": c, "mean_resid": r.mean(), "median_resid": r.median(), "skew": r.skew(),
                 "MAE": r.abs().mean(), "MAE_if_median_removed": (r - r.median()).abs().mean()})
show(pd.DataFrame(diag), "residual skew diagnostic (L2 training vs MAE scoring)")

for mode in C.WEATHER_MODES:
    json.dump(lgb_params[mode], open(C.FROZEN_DIR / f"lgbm_{mode}.json", "w"), indent=2)
    trials[mode].to_csv(C.RESULTS_DIR / f"optuna_trials_{mode}.csv", index=False)
val.to_parquet(C.RESULTS_DIR / "val_predictions.parquet", index=False)
print("\nfrozen:", sorted(p.name for p in C.FROZEN_DIR.glob("*.json")))
# Tiếp theo: tải results/ về repo, commit results/frozen/, auditor kiểm tra, rồi mới chạy 04 (test 2018) MỘT LẦN.
