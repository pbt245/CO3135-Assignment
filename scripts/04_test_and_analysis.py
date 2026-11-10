"""04. Chạy test 2018 (MỘT LẦN) và phân tích.

Điều kiện trước khi chạy: cấu hình trong results/frozen/ đã được commit và methodology-auditor đã kiểm tra.
Script từ chối chạy nếu thiếu tham số đã đóng băng, nếu tham số đến từ smoke run, hoặc nếu test đã chạy rồi.
Chạy lại test chỉ khi sửa một LỖI CÓ GHI NHẬN (đặt P4AIDS_RERUN_REASON), không đổi tham số, giữ kết quả cũ.

P4AIDS_DRYRUN=1: chạy toàn bộ với 2017 đóng vai "test" (train 2015-2016) để kiểm tra code MÀ KHÔNG NẠP 2018.
Kết quả dry-run không dùng cho báo cáo.
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from src import config as C, data, features as F, models as M, evaluation as E, plotting as P
from src.report import section, show, find_result

C.seed_everything(); P.setup()
DRY = os.environ.get("P4AIDS_DRYRUN", "0") == "1"
END = C.TEST_START if DRY else None            # dry run: 2018 is never loaded
FIT_END = C.VAL_START if DRY else C.TEST_START  # final models are fitted on dates < FIT_END
EVAL = "val" if DRY else "test"
TAG = "dryrun" if DRY else "test"
print("ROOT:", ROOT, "| Kaggle:", C.ON_KAGGLE, "| DRY RUN:", DRY, "| evaluating on:", EVAL, "| output:", C.OUT_DIR)

# ------------------------------------------------------------------ 0. preconditions
section("0. Preconditions")
frozen = {}
for name in ["ridge_actual", "ridge_lagged", "lgbm_actual", "lgbm_lagged"]:
    hit = next((d / f"{name}.json" for d in C.frozen_search_dirs() if (d / f"{name}.json").exists()), None)
    assert hit is not None, f"frozen/{name}.json missing: run scripts 02 and 03 first"
    frozen[name] = json.load(open(hit))
    print(f"{name:13s} <- {hit}")
smoke_params = [k for k, v in frozen.items() if v.get("smoke")]
assert DRY or not smoke_params, f"{smoke_params} come from a SMOKE run; rerun script 03 with the full budget"
if not DRY:
    done = [p for p in (C.RESULTS_DIR / "test_run_log.json", ROOT / "results" / "test_run_log.json") if p.exists()]
    reason = os.environ.get("P4AIDS_RERUN_REASON")
    assert not done or reason, f"Test already run ({done[0]}). Set P4AIDS_RERUN_REASON only to fix a documented bug."
run_started = dt.datetime.now().isoformat(timespec="seconds")

# ------------------------------------------------------------------ 1. full data + consistency check
# Nạp lại dữ liệu, lần này gồm cả 2018, rồi khẳng định feature của mọi hàng trước 2018 GIỐNG HỆT feature tính từ dữ
# liệu đã cắt ở script 01-03. Thêm 2018 mà làm đổi bất cứ gì của quá khứ thì có leakage.
section("1. Full data and consistency with development data")
panel, info = data.build_panel(end=END)
panel_dev, _ = data.build_panel(end=C.TEST_START)
feats = {}
for mode in C.WEATHER_MODES:
    full = F.build_features(panel, mode)
    dev = F.build_features(panel_dev, mode)
    cols = F.feature_cols(mode) + ["y", "usable"]
    a = full.loc[full["date"] < pd.Timestamp(C.TEST_START), cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, dev[cols].reset_index(drop=True))
    feats[mode] = full[full["usable"]].reset_index(drop=True)
    print(mode, "consistent with development data;", feats[mode].groupby("split").size().to_dict())
print("spikes set to NaN:", info["spikes"])

# ------------------------------------------------------------------ 2. refit + predict
# Tham số cố định từ validation. Mỗi model fit lại MỘT LẦN trên toàn bộ 2015-2017 rồi dự báo cả năm 2018.
# Vì sao không refit hằng tháng: giữ đúng giao thức của validation (fit các năm trước, dự báo trọn năm sau) để độ
# lệch validation-test so sánh được. Model có thể cũ đi về cuối năm: giới hạn đã biết, như nhau cho mọi model.
section("2. Refit on all data before the evaluation year and predict")
fit_mask = lambda f: f["date"] < pd.Timestamp(FIT_END)
ev_mask = lambda f: f["split"] == EVAL
base = feats["actual"]
res = base.loc[ev_mask(base), ["ts", "date", "hour", "y", "tso_fc", "dow", "month", "hol_share", "bridge", "T"]].copy()
res["B0"] = M.predict_b0(base.loc[ev_mask(base)])
res["TSO"] = res["tso_fc"]
models = {}
for mode in C.WEATHER_MODES:
    f = feats[mode]
    tr, ev = f[fit_mask(f)], f[ev_mask(f)]
    assert np.array_equal(ev["ts"].to_numpy(), res["ts"].to_numpy())
    rp, lp = frozen[f"ridge_{mode}"], frozen[f"lgbm_{mode}"]
    res[f"B1_{mode}"] = M.ridge_from_params(rp).fit(tr).predict(ev)
    models[mode] = M.lgbm_from_params(lp).fit(tr, lp["num_rounds"])
    res[f"M1_{mode}"] = models[mode].predict(ev)
    print(f"{mode}: fitted on {len(tr)} rows, predicting {len(ev)} rows")
COLS = ["B0", "B1_actual", "M1_actual", "B1_lagged", "M1_lagged"]
assert res[COLS].notna().all().all()
res.to_parquet(C.RESULTS_DIR / f"{TAG}_predictions.parquet", index=False)

# ------------------------------------------------------------------ 3. main table
# MAE (MW) là metric chính (đơn vị vật lý). RMSE nhấn mạnh giờ sai lớn. MAPE chỉ để đối chiếu tài liệu.
# TSO là tham chiếu, không phải đối thủ.
section(f"3. Main results on {EVAL}")
metrics = E.metric_table(res, COLS + ["TSO"])
vm = find_result("val_metrics.csv")
if vm is not None and not DRY:
    val_metrics = pd.read_csv(vm, index_col=0)
    metrics["val_MAE"] = val_metrics["MAE"]
    metrics["test_minus_val_MAE"] = metrics["MAE"] - metrics["val_MAE"]  # overfit / period-shift check
show(metrics, f"{EVAL} metrics", csv=f"{TAG}_metrics.csv")

pairs = [("B1_actual", "B0"), ("M1_actual", "B1_actual"), ("M1_actual", "B0"),
         ("B1_lagged", "B0"), ("M1_lagged", "B1_lagged"), ("M1_lagged", "B0"),
         ("B1_actual", "B1_lagged"), ("M1_actual", "M1_lagged")]
cmp = E.compare_pairs(res, pairs)
cmp.to_csv(C.RESULTS_DIR / f"{TAG}_comparison.csv", index=False)
show(cmp[["A", "B", "MAE_A", "MAE_B", "delta", "ci_low", "ci_high", "delta_%load", "threshold_MW",
          "months_A_better", "verdict"]], "paired comparisons (delta < 0: A better)")
# Hai dòng cuối so CÙNG model giữa actual và lagged: giá trị của việc BIẾT thời tiết thật so với chỉ có thời tiết cũ,
# không phải giá trị của một dự báo thời tiết thật.

# ------------------------------------------------------------------ 4. breakdowns
section("4. Error breakdowns")
by_month = E.breakdown(res, "month", COLS + ["TSO"])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
for ax, mode in zip(axes, C.WEATHER_MODES):
    for c, key in [("B0", "B0"), (f"B1_{mode}", "B1"), (f"M1_{mode}", "M1")]:
        ax.plot(by_month.index, by_month[c], marker="o", ms=3, color=P.COLORS[key], label=c)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("month"); ax.set_title(f"MAE by month ({mode} weather)")
    ax.legend()
axes[0].set_ylabel("MAE (MW)")
plt.tight_layout(); P.save(fig, f"04_{TAG}_mae_by_month"); plt.close(fig)
show(by_month.round(0), "MAE by month", csv=f"{TAG}_mae_by_month.csv")

by_hour = E.breakdown(res, "hour", COLS)
fig, ax = plt.subplots(figsize=(12, 3.2))
for c, key in [("B0", "B0"), ("B1_actual", "B1"), ("M1_actual", "M1")]:
    ax.plot(by_hour.index, by_hour[c], marker="o", ms=3, color=P.COLORS[key], label=c)
ax.set_xticks(range(24)); ax.set_xlabel("local hour"); ax.set_ylabel("MAE (MW)")
ax.set_title("MAE by hour of day (actual weather)"); ax.legend()
P.save(fig, f"04_{TAG}_mae_by_hour"); plt.close(fig)
show(by_hour.round(0), "MAE by hour", csv=f"{TAG}_mae_by_hour.csv")

res["day_type"] = E.day_type(res)
show(E.breakdown(res, "day_type", COLS + ["TSO"]).round(0), "MAE by day type", csv=f"{TAG}_mae_by_daytype.csv")
res["T_band"] = pd.cut(res["T"], [-10, 5, 10, 15, 20, 25, 45], labels=["<5", "5-10", "10-15", "15-20", "20-25", ">25"])
show(E.breakdown(res, "T_band", COLS).round(0), "MAE by temperature band", csv=f"{TAG}_mae_by_tband.csv")

# Ngoài miền train: cây không ngoại suy được, Ridge ngoại suy tuyến tính (HDD/CDD). Dưới 100 giờ -> "chưa kiểm chứng được".
trn = feats["actual"][fit_mask(feats["actual"])]
out_T = (res["T"] > trn["T"].max()) | (res["T"] < trn["T"].min())
out_y = (res["y"] > trn["y"].max()) | (res["y"] < trn["y"].min())
print(f"\ntrain T range {trn['T'].min():.1f}..{trn['T'].max():.1f} °C, load {trn['y'].min():,.0f}..{trn['y'].max():,.0f} MW")
print(f"evaluation hours outside train range: temperature {out_T.sum()}, load {out_y.sum()}")
if out_T.sum() + out_y.sum():
    show(E.breakdown(res[out_T | out_y].assign(k="outside"), "k", COLS).round(0), "MAE on hours outside train range")

# ------------------------------------------------------------------ 5. where does M1 win/lose vs B1
# |e_M1| - |e_B1| theo giờ (âm = LightGBM tốt hơn), gộp theo nhóm: bằng chứng để trả lời "vì sao" thay vì đoán.
section("5. Where LightGBM gains or loses against Ridge")
for mode in C.WEATHER_MODES:
    res[f"gain_{mode}"] = (res["y"] - res[f"M1_{mode}"]).abs() - (res["y"] - res[f"B1_{mode}"]).abs()
gain = pd.concat({
    "day_type": res.groupby("day_type")[["gain_actual", "gain_lagged"]].mean(),
    "T_band": res.groupby("T_band", observed=True)[["gain_actual", "gain_lagged"]].mean(),
    "month": res.groupby("month")[["gain_actual", "gain_lagged"]].mean(),
})
show(gain.round(0), "mean |e_M1| - |e_B1| (MW, negative = LightGBM better)", csv=f"{TAG}_gain_breakdown.csv")
daily_gain = res.groupby("date")[["gain_actual"]].mean().join(res.groupby("date")["day_type"].first())
daily_gain["T_mean"] = res.groupby("date")["T"].mean()
show(daily_gain.nsmallest(10, "gain_actual").round(1), "10 days where LightGBM gains most (actual weather)")
show(daily_gain.nlargest(10, "gain_actual").round(1), "10 days where LightGBM loses most (actual weather)")

# ------------------------------------------------------------------ 6. SHAP
# pred_contrib của LightGBM (TreeSHAP, không cần gói shap). Đọc THEO NHÓM (lag, lịch, ngày lễ, thời tiết) vì feature
# trong nhóm tương quan mạnh và TreeSHAP chỉ tính một phần sự phụ thuộc đó. SHAP = MODEL DÙNG GÌ, không phải nhân quả.
# Nếu target = delta7 thì SHAP giải thích MỨC THAY ĐỔI so với cùng giờ tuần trước, không phải mức phụ tải.
section("6. SHAP explanation of LightGBM")
groups = {"load lags": F.LAG, "calendar": F.CAL, "holidays": F.HOL}
shap_tabs = {}
for mode in C.WEATHER_MODES:
    f = feats[mode]; ev = f[ev_mask(f)]
    sv = M.shap_values(models[mode], ev)
    g = pd.DataFrame({k: sv[v].sum(axis=1) for k, v in {**groups, "weather": F.WX[mode]}.items()})
    shap_tabs[mode] = (sv, g)
    print(f"{mode}: target = {models[mode].target}")
    show(sv.drop(columns="bias").abs().mean().sort_values(ascending=False).round(0).to_frame("mean|SHAP|"),
         f"per-feature mean |SHAP| ({mode})", csv=f"{TAG}_shap_features_{mode}.csv")
imp = pd.DataFrame({m: shap_tabs[m][1].abs().mean() for m in C.WEATHER_MODES})
fig, ax = plt.subplots(figsize=(7, 2.8))
yy = np.arange(len(imp))
ax.barh(yy - 0.2, imp["actual"], height=0.38, color=P.COLORS["actual"], label="actual weather")
ax.barh(yy + 0.2, imp["lagged"], height=0.38, color=P.COLORS["lagged"], label="lagged weather")
ax.set_yticks(yy, imp.index); ax.invert_yaxis(); ax.set_xlabel("mean |SHAP| (MW)")
ax.set_title("LightGBM: contribution by feature group"); ax.legend()
P.save(fig, f"04_{TAG}_shap_groups"); plt.close(fig)
show(imp.round(0), "mean |SHAP| by feature group", csv=f"{TAG}_shap_groups.csv")

sv, g = shap_tabs["actual"]
ev = feats["actual"][ev_mask(feats["actual"])]
show(g.abs().groupby(ev["month"].to_numpy()).mean().round(0).T, "mean |SHAP| by month, actual weather",
     csv=f"{TAG}_shap_groups_by_month.csv")

# Đáp ứng nhiệt độ của LightGBM: chữ U/V (tăng khi lạnh và khi nóng) nghĩa là model tự học được dạng mà Ridge phải
# được cấp sẵn qua HDD/CDD.
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
wx = sv[F.WX["actual"]].sum(axis=1)
for ax, hours, title in [(axes[0], [4], "04:00"), (axes[1], [13], "13:00")]:
    m = ev["hour"].isin(hours).to_numpy()
    ax.scatter(ev.loc[m, "T"], wx[m], s=5, alpha=0.5, color=P.COLORS["M1"])
    ax.axhline(0, color=P.MUTED, lw=0.8)
    ax.set_xlabel("temperature T (°C)"); ax.set_title(f"SHAP of weather features vs T, {title}")
axes[0].set_ylabel("SHAP(T) + SHAP(T_ma24)  (MW)")
plt.tight_layout(); P.save(fig, f"04_{TAG}_shap_temperature"); plt.close(fig)

log = {"run": TAG, "started": run_started, "finished": dt.datetime.now().isoformat(timespec="seconds"),
       "rerun_reason": os.environ.get("P4AIDS_RERUN_REASON"), "frozen": frozen,
       "versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
       "mae": metrics["MAE"].to_dict()}
json.dump(log, open(C.RESULTS_DIR / f"{TAG}_run_log.json", "w"), indent=2, default=str)
print(f"\nsaved results for {TAG} in {C.RESULTS_DIR} and figures in {C.FIG_DIR}")
# Phân tích viết sau khi có kết quả (theo mức tốt hơn rõ / xấp xỉ / không phân định): B1 vs B0; M1 vs B1 và ở đâu
# (mục 5); actual vs lagged; độ lệch validation-test; SHAP; TSO chỉ mô tả.
