# Dự báo phụ tải điện day-ahead cho Tây Ban Nha

Assignment môn Programming for AI & Data Science. Dự báo 24 giờ phụ tải ngày D, phát hành lúc 10:00 ngày D-1.
So sánh B0 seasonal naive, B1 Ridge theo giờ (có feature engineering) và M1 LightGBM + Optuna; giải thích bằng SHAP.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `src/config.py` | Split, mốc thông tin, seed, đường dẫn (mỗi hằng số có lý do kèm theo) |
| `src/data.py` | Làm sạch: múi giờ, DST, khử trùng weather, gai 1 giờ, thiếu dữ liệu |
| `src/features.py` | Feature + `assert_no_lookahead` (tự kiểm tra leakage) |
| `src/models.py` | B0, B1 `RidgePerHour`, M1 `LGBMModel`, tuning, SHAP |
| `src/evaluation.py` | Metric, block bootstrap theo tuần, phân tầng sai số |
| `scripts/0X_*.py` | Các bước của pipeline; phần "vì sao" nằm trong docstring/comment |
| `data/external/holidays_es.csv` | Bảng ngày lễ (sinh bởi `tools/make_holiday_table.py`) |
| `results/frozen/` | Tham số đã chốt trên validation. **Phải commit trước khi chạy test** |
| `tools/kaggle_sync.sh` | Đóng gói code vào `~/kaggle_up/p4aids-code` và upload dataset `bathanhphan/p4aids-code` |

| Script | Việc | Chạy ở đâu |
|---|---|---|
| 01_data_cleaning | Làm sạch, kiểm tra căn thời gian, EDA và S2 trên train, kiểm tra leakage | Máy cá nhân (vài giây) |
| 02_features_baselines | Feature, B0, tuning Ridge trên validation 2017 | Máy cá nhân (~15 giây) |
| 03_lightgbm_tuning | Optuna 30 trial × 2 cấu hình thời tiết, so sánh trên validation | **Kaggle** |
| 04_test_and_analysis | Test 2018 **một lần**, phân tích, SHAP | **Kaggle**, sau khi auditor kiểm tra |

Năm 2018 (test) bị cắt khỏi dữ liệu thô ở script 01-03. Chỉ script 04 nạp năm này.
Output: `results/` (bảng, tham số, dự báo), `figures/` (hình). Trên Kaggle là `/kaggle/working/`.

## Chạy trên máy

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp
kaggle datasets download -d nicholasjhana/energy-consumption-generation-prices-and-weather -p data/raw --unzip
.venv/bin/python scripts/01_data_cleaning.py
.venv/bin/python scripts/02_features_baselines.py
# smoke test 03 và dry-run 04 (2017 đóng vai test, KHÔNG nạp 2018), output ra thư mục tạm:
P4AIDS_SMOKE=1 P4AIDS_OUT=/tmp/p4aids_smoke .venv/bin/python scripts/03_lightgbm_tuning.py
P4AIDS_SMOKE=1 P4AIDS_DRYRUN=1 P4AIDS_OUT=/tmp/p4aids_smoke .venv/bin/python scripts/04_test_and_analysis.py
```

## Chạy trên Kaggle

Code được upload thành dataset private `bathanhphan/p4aids-code` bằng `tools/kaggle_sync.sh create`, các lần sau dùng
`tools/kaggle_sync.sh version "<message>"`. Trên Kaggle tạo notebook, *Add Input* hai dataset
(`nicholasjhana/energy-consumption-generation-prices-and-weather` và `bathanhphan/p4aids-code`), chọn CPU, rồi chạy
script bằng `!python -u <đường dẫn>/scripts/0X_*.py`. *Save Version → Save & Run All* để output tải về được bằng
`kaggle kernels output bathanhphan/<notebook-slug> -p <thư mục>`.

Script 04 từ chối chạy nếu tham số đến từ smoke run, hoặc nếu test đã chạy (trừ khi đặt `P4AIDS_RERUN_REASON` để sửa
một lỗi có ghi nhận).

## Tái lập

Seed 42 dùng chung cho numpy, Optuna TPE, LightGBM và bootstrap. LightGBM chạy với `deterministic=True`,
`force_row_wise=True`, 4 thread. Phiên bản lightgbm/optuna được ghi vào file JSON đã chốt.
