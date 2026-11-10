"""01. Nạp và làm sạch dữ liệu.

Bài toán: dự báo phụ tải điện Tây Ban Nha cho 24 giờ của ngày D, phát hành lúc 10:00 ngày D-1 (day-ahead).
Dataset: Kaggle nicholasjhana/energy-consumption-generation-prices-and-weather (ENTSO-E + thời tiết 5 thành phố).

Script này chỉ dùng dữ liệu đến hết 2017. Năm 2018 (test) bị cắt khỏi dữ liệu thô TRƯỚC mọi bước xử lý,
nên test không thể ảnh hưởng đến quyết định nào ở đây (kể cả một phép forward-fill).

Các bước làm sạch và lý do (code ở src/data.py):
| Bước              | Làm gì                                                       | Vì sao |
| Múi giờ           | time có offset +01/+02 -> UTC -> Europe/Madrid                | Nhu cầu đi theo đồng hồ địa phương; dùng UTC thì đỉnh sáng trôi 1 giờ giữa đông và hè |
| Lưới ngày x 24 giờ| Ngày 23 giờ: tạo ô giờ 02, không chấm. Ngày 25 giờ: TB hai giờ 02 | Lag tra theo "cùng giờ địa phương, k ngày trước"; shift(168) lệch 1 giờ cả tuần sau đổi giờ |
| Weather trùng     | Khử trùng theo (thành phố, giờ); các bản trùng có nhiệt độ giống hệt | ~3.000 dòng trùng, không khử thì merge nhân bản dòng |
| Tên thành phố     | " Barcelona" -> strip                                         | Nếu không, thành một "thành phố" riêng |
| Đơn vị            | Kelvin -> °C                                                  | Dễ đọc; base HDD/CDD theo °C |
| Gai 1 giờ vô lý   | Lệch > 15% so với TB 2 giờ kề, 2 giờ kề khớp nhau (< 10%) -> NaN | 12 GW mất rồi hồi lại trong 1 giờ là lỗi ghi; ngưỡng 15/20/25% cho cùng tập điểm |
| Thiếu ở target    | Không train và không chấm trên giờ không có quan sát hợp lệ    | Không chấm trên giá trị bịa |
| Thiếu khi làm lag | ffill tối đa 3 giờ                                            | Chỉ dùng quá khứ; nội suy dùng giá trị tương lai (leakage nhỏ) |
| Cột bỏ            | generation *, price *, forecast *                             | Đồng thời với target hoặc công bố sau 10:00 D-1. total load forecast (TSO) chỉ là tham chiếu |
| Outlier "thật"    | Không tự loại (tụt lúc nửa đêm, ngày lễ)                      | Là sự kiện thật, phải chấm trên chúng; dự báo TSO cũng tụt theo |
| Thời tiết         | Trung bình đơn giản 5 thành phố                               | Không cần dữ liệu ngoài; hai model dùng cùng biến. Giới hạn: không trọng số dân số |
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C, data, features as F, plotting as P
from src.report import section, show

C.seed_everything(); P.setup()
print("ROOT:", ROOT, "| Kaggle:", C.ON_KAGGLE, "| smoke:", C.SMOKE, "| output:", C.OUT_DIR)

# ------------------------------------------------------------------ 1. clean panel
section("1. Data quality after cleaning (data up to 2017-12-31)")
panel, info = data.build_panel(end=C.TEST_START)
show(pd.Series(info, name="value").to_frame(), "cleaning report")

# ------------------------------------------------------------------ 2. time alignment
# Nếu timestamp là giờ BẮT ĐẦU thì đỉnh solar rơi vào 13-14h địa phương (trưa mặt trời ~13:15 đông, ~14:15 hè)
# và đỉnh nhiệt độ giữa chiều. Mốc thông tin (dữ liệu đến giờ 08 ngày D-1, phát hành 10:00) an toàn với cả hai cách hiểu.
section("2. Time alignment of load, solar and weather")
raw = pd.read_csv(C.RAW_DIR / "energy_dataset.csv", usecols=["time", "generation solar"])
raw["local"] = pd.to_datetime(raw["time"], utc=True).dt.tz_convert(C.TZ)
raw = raw[raw["local"].dt.tz_localize(None) < pd.Timestamp(C.TEST_START)]
solar_by_h = raw.groupby(raw["local"].dt.hour)["generation solar"].mean()
temp_by_h = panel.groupby("hour")["T"].mean()
load_by_h = panel.groupby("hour")["y"].mean()
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, s, title, unit in [(axes[0], solar_by_h, "Solar generation", "MW"),
                           (axes[1], temp_by_h, "Temperature (5-city mean)", "°C"),
                           (axes[2], load_by_h, "Load", "MW")]:
    ax.plot(s.index, s.values, color=P.COLORS["B0"])
    ax.set_title(f"{title} by local hour"); ax.set_xlabel("local hour"); ax.set_ylabel(unit)
    ax.axvline(s.idxmax(), color=P.MUTED, lw=0.8, ls="--")
plt.tight_layout(); P.save(fig, "01_alignment_by_hour"); plt.close(fig)
print("peak hours -> solar:", solar_by_h.idxmax(), "| temperature:", temp_by_h.idxmax(), "| load:", load_by_h.idxmax())
# Kết quả: đỉnh solar giờ 13 (khớp timestamp = giờ bắt đầu), đỉnh nhiệt 16h -> load và weather cùng trục, không dịch.

# ------------------------------------------------------------------ 3. NaN-ed values
section("3. Values set to NaN: one-hour spikes and DST gaps")
flag = panel[(panel["spike"] == 1) | (panel["dst_gap"] == 1)][["ts", "y", "spike", "dst_gap"]]
ctx = panel.set_index("ts")["y_src"]
flag = flag.assign(prev_hour=[ctx.get(t - pd.Timedelta("1h")) for t in flag["ts"]],
                   next_hour=[ctx.get(t + pd.Timedelta("1h")) for t in flag["ts"]])
show(flag, "spikes (spike=1) and spring-forward slots (dst_gap=1)")

# ------------------------------------------------------------------ 4. EDA on train only
# EDA chỉ dùng TRAIN 2015-2016, để những gì nhìn thấy không ảnh hưởng đến lựa chọn trên validation.
section("4. EDA on train 2015-2016")
tr = panel[panel["date"] < pd.Timestamp(C.VAL_START)].copy()
daily = panel.groupby("date")["y"].mean()
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(daily.index, daily.values, lw=0.8, color=P.COLORS["B0"])
ax.plot(daily.rolling(28, center=True).mean(), color=P.INK, lw=1.5, label="28-day mean")
ax.axvline(pd.Timestamp(C.VAL_START), color=P.MUTED, ls="--", lw=0.8)
ax.text(pd.Timestamp(C.VAL_START), daily.max(), " validation 2017", color=P.MUTED, va="top")
ax.set_title("Daily mean load, 2015-2017 (test year 2018 not loaded)"); ax.set_ylabel("MW"); ax.legend()
P.save(fig, "01_daily_load"); plt.close(fig)

hol = F.load_holiday_table()
tr["dow"] = tr["date"].dt.day_of_week
tr["hol"] = tr["date"].map(hol["hol_share"]).fillna(0)
tr["daytype"] = np.where((tr["dow"] >= 5) | (tr["hol"] >= 0.5), "weekend/holiday", "workday")
tr["Tbin"] = tr["T"].round()
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
for ax, hours, title in [(axes[0], [4], "04:00 (night)"), (axes[1], [12, 13], "12:00-13:00 (midday)")]:
    sub = tr[tr["hour"].isin(hours)]
    for (dt_, g), col in zip(sub.groupby("daytype"), [P.COLORS["B0"], P.COLORS["B1"]]):
        b = g.groupby("Tbin")["y"].agg(["mean", "size"])
        b = b[b["size"] >= 15]
        ax.plot(b.index, b["mean"], marker="o", ms=3, color=col, label=dt_)
    ax.set_title(f"Load vs temperature, {title}, train 2015-16"); ax.set_xlabel("5-city mean temperature (°C)")
axes[0].set_ylabel("mean load (MW)"); axes[1].legend()
plt.tight_layout(); P.save(fig, "01_load_vs_temperature"); plt.close(fig)
print("figures saved to", C.FIG_DIR)
# Hình chữ U thấy rõ ở giờ trưa ngày làm việc; ban đêm nhánh nóng yếu hơn. Nhưng hình này LẪN VỚI MÙA, nên
# chưa phải bằng chứng nhiệt độ có tác động riêng -> kiểm tra S2 dưới đây.

# ------------------------------------------------------------------ 5. S2: temperature effect within month
# Trong mỗi ô (giờ x thứ x tháng x năm, và biến thể giờ x nhóm ngày x tuần x năm), bỏ ngày lễ và cửa sổ Giáng sinh,
# trừ trung bình của cả phụ tải và nhiệt độ rồi tính beta = sum(y*T)/sum(T^2). Chỉ còn biến động TRONG CÙNG THÁNG/TUẦN.
# Quy tắc chốt trước (pipeline-v1): một nhánh ĐẠT khi đúng dấu (đông âm, hè dương) ở CẢ HAI năm và tác động >= 0,5% phụ tải.
# Đây là thống kê mô tả trên train (không chọn gì), không dùng làm bằng chứng cho kết quả mô hình.
section("5. S2: within-month temperature effect (train only)")
tr["xmas"] = ((tr["date"].dt.month == 12) & (tr["date"].dt.day >= 24)) | ((tr["date"].dt.month == 1) & (tr["date"].dt.day <= 6))
tr["year"], tr["month"] = tr["date"].dt.year, tr["date"].dt.month
tr["week"] = tr["date"].dt.isocalendar().week.astype(int).to_numpy()
tr["dgrp"] = np.where(tr["dow"] < 5, "workday", tr["dow"].astype(str))  # pool Mon-Fri inside a week
s2_base = tr[(tr["hol"] < 0.5) & ~tr["xmas"]].dropna(subset=["y"])
WRONG_SIGN = {"winter Dec-Feb": 1, "July": -1, "August": -1}


def s2_table(df, cell, label):
    d = df.copy()
    d["y_d"] = d["y"] - d.groupby(cell)["y"].transform("mean")
    d["T_d"] = d["T"] - d.groupby(cell)["T"].transform("mean")
    rows = []
    for name, months in [("winter Dec-Feb", [12, 1, 2]), ("July", [7]), ("August", [8])]:
        for yr in (2015, 2016):
            g = d[d["month"].isin(months) & (d["year"] == yr)]
            beta = (g["y_d"] * g["T_d"]).sum() / (g["T_d"] ** 2).sum()
            impact = 100 * abs(beta) * g["T_d"].std() / tr["y"].mean()
            ok = np.sign(beta) != WRONG_SIGN[name] and impact >= 0.5
            rows.append({"control": label, "season": name, "year": yr, "beta_MW_per_C": beta,
                         "impact_%load": impact, "pass": ok})
    return pd.DataFrame(rows)


s2 = pd.concat([s2_table(s2_base, ["hour", "dow", "month", "year"], "hour x weekday x month x year"),
                s2_table(s2_base, ["hour", "dgrp", "week", "year"], "hour x day-group x week x year")])
show(s2, "S2 per season and year", csv="s2_within_month.csv")
show(s2.groupby(["control", "season"])["pass"].all().unstack(), "S2 verdict (branch passes only if both years pass)")
# Kết quả trên train: mùa đông KHÔNG đạt (2015 beta dương ở mọi cách kiểm soát; có thể vì sưởi ở Tây Ban Nha phần lớn
# dùng gas, hoặc do chất lượng trung bình 5 thành phố, dữ liệu không đủ để phân biệt). Mùa hè yếu và không ổn định.
# Hệ quả: chữ U chủ yếu đi cùng MÙA. Nhiệt độ vẫn là feature (để model tự quyết), nhưng báo cáo KHÔNG khẳng định cơ
# chế "nhiệt độ làm tăng phụ tải" ngoài phần mùa.

# ------------------------------------------------------------------ 6. automatic leakage test
# Với một số ngày D ngẫu nhiên: thay TOÀN BỘ phụ tải và nhiệt độ sau 08:00 ngày D-1 bằng giá trị nhiễu, tính lại
# feature, khẳng định feature của ngày D không đổi. Ở cấu hình "actual", nhiệt độ của chính ngày D được miễn
# (cấu hình này cố ý giả định có dự báo thời tiết hoàn hảo).
section("6. Automatic look-ahead test")
for mode in C.WEATHER_MODES:
    res = F.assert_no_lookahead(panel, mode, n_days=8)
    print(f"{mode:>7}: {len(res)} random days, max feature change = {res['max_abs_change'].max():.1f} -> OK")

panel.to_parquet(C.PROCESSED_DIR / "panel_dev.parquet", index=False)
json.dump(info, open(C.RESULTS_DIR / "data_quality.json", "w"), indent=2, default=str)
print("\nsaved", C.PROCESSED_DIR / "panel_dev.parquet", "and", C.RESULTS_DIR / "data_quality.json")
