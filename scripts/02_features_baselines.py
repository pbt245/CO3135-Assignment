"""02. Feature engineering và baseline B0, B1 (chỉ dữ liệu đến hết 2017; train 2015-16, validation 2017).

Mọi lựa chọn (base HDD/CDD, alpha, dạng target) được chọn theo MAE trên validation, riêng cho từng cấu hình thời tiết.

MỐC THÔNG TIN. Dự báo cho mọi giờ của ngày D phát hành lúc 10:00 ngày D-1 (2 giờ trước gate closure 12:00).
Giờ phụ tải cuối cùng chắc chắn đã có là giờ 08 của D-1, nên lag cùng giờ gần nhất là D-2, không phải D-1.

HAI CẤU HÌNH THỜI TIẾT (file không có DỰ BÁO thời tiết; báo cáo cả hai, không trộn trong một so sánh):
| actual | quan sát thật của ngày D  | như có dự báo thời tiết hoàn hảo -> cận trên (lạc quan) |
| lagged | chỉ quan sát đến 08:00 D-1 | làm được khi không có nguồn dự báo thời tiết -> cận dưới |
Năng lực khi có dự báo thời tiết thật nằm giữa hai cận, nhưng không đo được với dữ liệu này.

FEATURE (src/features.py):
| Lag phụ tải | lag_d2 = y(D-2,h), lag_d7 = y(D-7,h)       | chu kỳ ngày và tuần; D-2 là cùng giờ mới nhất đã biết |
|             | lag_d1_h8 = y(D-1, 08:00)                   | giờ mới nhất đã biết: MỨC HIỆN TẠI khi có thay đổi đột ngột |
|             | ratio_morning = TB(0-8h,D-1)/TB(0-8h,D-2)   | mức đang lên hay xuống |
| Lịch        | giờ, thứ, tháng, sin/cos ngày trong năm     | chu kỳ ngày, tuần, mùa |
| Ngày lễ     | hol_nat, hol_share (tỉ lệ 17 vùng), bridge, xmas | ngày lễ là nơi lag theo mùa sai nhiều nhất; hol_share bắt cả lễ vùng |
|             | hol_share_d1/d2/d7                          | ngày dùng làm lag là ngày lễ -> lag là tham chiếu tồi |
| actual      | T (giờ t), T_ma24 (TB 24 giờ trước)         | nhiệt độ hiện tại + quán tính nhiệt của nhà cửa |
| lagged      | T_d1_h8, T_ma24_b, T_d2_h                   | thông tin thời tiết mới nhất ở mốc phát hành |
Ridge nhận HDD = max(T_h - T, 0), CDD = max(T - T_c, 0) thay nhiệt độ thô (model tuyến tính không tự tạo chữ U).
LightGBM nhận nhiệt độ thô (cây tự tìm ngưỡng).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C, data, features as F, models as M, evaluation as E, plotting as P
from src.report import section, show

C.seed_everything(); P.setup()
print("ROOT:", ROOT, "| Kaggle:", C.ON_KAGGLE, "| smoke:", C.SMOKE, "| output:", C.OUT_DIR)

# ------------------------------------------------------------------ features
section("1. Features")
panel, _ = data.build_panel(end=C.TEST_START)
feats = {}
for mode in C.WEATHER_MODES:
    f = F.build_features(panel, mode)
    assert (f["split"] != "test").all(), "test rows must not exist in development scripts"
    feats[mode] = f[f["usable"]].reset_index(drop=True)
    print(mode, feats[mode].groupby("split").size().to_dict(), "| features:", len(F.feature_cols(mode)))
# Hai cấu hình có CÙNG tập hàng -> mọi model được chấm trên đúng cùng các giờ.
assert feats["actual"]["ts"].equals(feats["lagged"]["ts"])
show(feats["actual"][F.feature_cols("actual")].describe().T[["mean", "std", "min", "max"]], "feature summary (actual)")

# ------------------------------------------------------------------ B0
# Seasonal naive (cùng giờ tuần trước): sàn mọi model phải vượt; nắm chu kỳ ngày/tuần mà không học gì.
# TSO = dự báo của nhà vận hành có trong file: CHỈ là tham chiếu (TSO có dữ liệu nội bộ, dự báo thời tiết, và không rõ
# cột được cập nhật lúc nào) -> không kết luận thắng/thua TSO.
section("2. B0 seasonal naive on validation")
va_rows = feats["actual"]["split"] == "val"
val = feats["actual"].loc[va_rows, ["ts", "date", "hour", "y", "tso_fc", "dow", "month", "hol_share", "bridge", "T"]].copy()
val["B0"] = M.predict_b0(feats["actual"].loc[va_rows])
val["TSO"] = val["tso_fc"]
show(E.metric_table(val, ["B0", "TSO"]), "validation 2017")

# ------------------------------------------------------------------ B1 Ridge per hour
# 24 Ridge (mỗi giờ địa phương một model), chung alpha và base. Vì sao theo giờ: khi mọi feature tương tác với giờ và
# chung alpha, Ridge toàn cục = 24 Ridge riêng; đây là cách đơn giản nhất để tác động của nhiệt độ/thứ/lag khác nhau
# theo giờ. StandardScaler nằm trong pipeline -> chỉ fit trên train.
# Chọn trên validation 2017, với mỗi dạng target: (1) base T_h in {10..20}, T_c in {14..26}, T_h <= T_c, ở alpha giữa
# thang; (2) alpha 11 giá trị log 1e-2..1e3 ở base đã chọn. Grid ban đầu (T_c >= 18) cho tối ưu ở mép -> đã mở rộng
# xuống 14 trước khi chọn (quy tắc của kế hoạch).
# Dạng target: lần chạy đầu bias -1,3% trên 2017; kế hoạch viết trước: |bias| >= 1% thì thử thêm y - y(D-7,h) và chọn
# trên validation. Hai dạng (level, delta7) có cùng ngân sách.
section("3. B1 per-hour Ridge: tuning on validation")
ridge_params, ridge_tabs = {}, {}
for mode in C.WEATHER_MODES:
    f = feats[mode]
    tr, va = f[f["split"] == "train"], f[f["split"] == "val"]
    params, base_tab, alpha_tab = M.tune_ridge(tr, va, mode)
    ridge_params[mode], ridge_tabs[mode] = params, (base_tab, alpha_tab)
    val[f"B1_{mode}"] = M.ridge_from_params(params).fit(tr).predict(va)
show(pd.DataFrame(ridge_params).T, "chosen Ridge parameters")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
for ax, mode in zip(axes, C.WEATHER_MODES):
    at = ridge_tabs[mode][1]
    for tgt, col in zip(M.TARGETS, [P.COLORS["B0"], P.COLORS["B1"]]):
        s = at[at["target"] == tgt]
        ax.plot(s["alpha"], s["val_mae"], marker="o", ms=3, color=col, label=f"target = {tgt}")
    ax.set_xscale("log"); ax.set_xlabel("alpha"); ax.set_title(f"Ridge validation MAE vs alpha ({mode})")
axes[0].set_ylabel("validation MAE (MW)"); axes[1].legend()
plt.tight_layout(); P.save(fig, "02_ridge_alpha"); plt.close(fig)
# Đường cong có đáy bên trong grid (không ở mép) -> grid alpha đủ rộng.

base_tab = ridge_tabs["actual"][0]
best_t = ridge_params["actual"]["target"]
show(base_tab[base_tab["target"] == best_t].pivot(index="th", columns="tc", values="val_mae").round(0),
     f"validation MAE by (T_h, T_c), weather=actual, target={best_t}")
# Bảng base khá phẳng (vài MW) -> kết quả không phụ thuộc mạnh vào base chính xác; cũng là lý do KHÔNG đọc base hay
# hệ số Ridge thành "MW/°C" (hệ số bị co lại, chia cho feature tương quan). Giải thích dùng SHAP của LightGBM (04).

# ------------------------------------------------------------------ validation results
section("4. Validation results and bias by month")
show(E.metric_table(val, ["B0", "B1_actual", "B1_lagged", "TSO"]), "validation 2017")
bias = pd.DataFrame({c: (val[c] - val["y"]).groupby(val["month"]).mean() for c in ["B0", "B1_actual", "B1_lagged"]})
show(bias.T.round(0), "mean error (prediction - actual) by month")
# Bias âm lớn nhất ở THÁNG 6/2017: đợt nắng nóng kỷ lục, phụ tải tháng 6 cao hơn ~2.000 MW so với tháng 6 hai năm
# train. Train chỉ có 2 mùa hè -> giới hạn dữ liệu, ghi vào báo cáo. Không thêm feature để "vá" sau khi đã nhìn
# validation (sẽ là overfit validation).

for mode in C.WEATHER_MODES:
    json.dump(ridge_params[mode], open(C.FROZEN_DIR / f"ridge_{mode}.json", "w"), indent=2)
    ridge_tabs[mode][0].to_csv(C.RESULTS_DIR / f"ridge_base_grid_{mode}.csv", index=False)
    ridge_tabs[mode][1].to_csv(C.RESULTS_DIR / f"ridge_alpha_grid_{mode}.csv", index=False)
val.to_parquet(C.RESULTS_DIR / "val_predictions_baselines.parquet", index=False)
print("\nfrozen:", sorted(p.name for p in C.FROZEN_DIR.glob("ridge_*.json")))
