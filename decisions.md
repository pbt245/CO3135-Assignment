# Decision log

## [2026-11-07] Quyết định: Chọn topic và dataset (Giai đoạn 1)
- Chọn: Topic 1, dự báo phụ tải điện day-ahead của Tây Ban Nha (Demand Forecasting, thuộc danh sách gợi ý). Dataset: Kaggle `nicholasjhana/energy-consumption-generation-prices-and-weather`. Người dùng (Jenda) chốt.
- Vì sao (gắn với đặc điểm dữ liệu/bài toán):
  - Dữ liệu theo giờ có phụ tải và thời tiết 5 thành phố, nên giả thuyết chữ U nhiệt độ - phụ tải và tương tác giờ × loại ngày có thể kiểm chứng bằng thí nghiệm rẻ. Đây vẫn là giả thuyết, chưa kiểm tra trên dữ liệu.
  - Baseline chính (seasonal naive, Ridge) soi được bằng chính thí nghiệm, nên thắng hay thua đều có hướng giải thích.
  - Chạy được trên CPU, không phụ thuộc GPU quota; phù hợp 2 tuần trên Kaggle.
- Các phương án đã cân nhắc và lý do loại:
  - Topic 2, LendingClub (`wordsforthewise/lending-club`): mạnh nhất về preprocessing/FE, nhưng tốn công nhất, test có thể bị thu hẹp, và baseline trung tâm `sub_grade` là hộp đen, nên thắng hay thua đều có câu hỏi không trả lời trọn vẹn được.
  - Topic 3, sentiment PhraseBank (`ankurzing/sentiment-analysis-for-financial-news`): đề bài ghi "và/hoặc" nên không cần làm DL; cần GPU và nạp trọng số offline; người dùng của bài toán là giả định; phần cơ chế nhãn nhiễu chỉ mang tính mô tả.
  - Credit Card Fraud: feature đã qua PCA nên ít đất cho FE và khó giải thích. Kết luận chỉ áp dụng cho dataset này, không cho cả nhóm Anomaly Detection.
  - Diabetes 130-US: trùng dạng bài với Topic 2.
  - Home Credit: đa bảng, công sức rơi vào việc nối bảng. M5: quy mô và phân cấp vượt mức 2 tuần. Rossmann: lý do yếu, chỉ là slot dự báo đã dành cho Topic 1.
- Phản biện mạnh nhất và cách trả lời:
  - "TSO đã có dự báo trong file, vậy ai cần model này?" (A1). Hướng trả lời do phản biện gợi ý, chưa qua tranh luận: model không thay thế TSO; đối tượng là người chỉ có dữ liệu công khai; câu hỏi là phần sai số nào giảm được nhờ FE so với độ phức tạp của model. Dòng "Ai cần" trong thiết kế chưa được viết lại.
  - "Con số nào là năng lực thật?" (A2). File không có dự báo thời tiết, nên chỉ báo cáo cận trên và cận dưới, không có con số điểm.
  - Cột `total load forecast` chưa rõ mốc thông tin (T1.1): chỉ là tham chiếu, không kết luận thắng/thua TSO; so sánh chính dựa trên B0/B1.
- Bằng chứng (thí nghiệm/nguồn):
  - Chưa có thí nghiệm nào; chưa ai mở file dữ liệu. License và danh sách file đã tra qua Kaggle API. Số dòng, số cột mới đến từ nguồn thứ cấp.
  - Tranh luận: `debate/topic-v1.md`, `debate/topic-critique-v1.md`, `debate/topic-v2.md`, `debate/topic-critique-v2.md`, `debate/topic-viva-t1-t2.md`.
- Điều kiện còn mở:
  - Phản biện kết luận ĐẠT CÓ ĐIỀU KIỆN: phải sửa N.1 [CHẶN] (phép thử chữ U đặt sai cấu hình thời tiết), N.2, N.3 trước khi code. `topic-scout` chưa trả lời các ý N.x.
  - Người dùng chốt topic trước khi chạy notebook kiểm tra trên Kaggle. Kiểm tra này vẫn cần làm: nếu thời tiết thiếu nhiều, hoặc không thấy phi tuyến sau khi kiểm soát giờ và loại ngày, thì phải báo lại để xem xét quyết định này.
  - Chưa xác minh "GPU thì internet tắt" và giới hạn tài nguyên Kaggle. Topic 1 chạy CPU nên ít bị ảnh hưởng.

## [2026-11-08] Quyết định: Đóng Giai đoạn 2, literature review
- Chọn: dùng `debate/litreview-v2.md` (đã có mục "Sửa sau phản biện vòng 2") làm nền cho Giai đoạn 3. Lõi 17 nguồn (11 đọc toàn văn qua công cụ trích xuất, 6 chỉ abstract); tổng 33 nguồn, dự kiến cắt còn khoảng 25 khi viết báo cáo. Người dùng bảo đi tiếp sau phản biện vòng 2; `thay-phan-bien` kết luận ĐẠT CÓ ĐIỀU KIỆN, không còn ý [CHẶN].
- Vì sao (gắn với đặc điểm dữ liệu/bài toán):
  - Không có quyết định nào của Giai đoạn 3 cần biết trước hồi quy có FE hay GBM tốt hơn, vì chính thí nghiệm của bài là phép so sánh đó.
  - Tài liệu đã đọc không có so sánh đúng thiết lập của bài (quốc gia, theo giờ, day-ahead, có thời tiết, không có dự báo thời tiết).
- Các phương án đã cân nhắc và lý do loại:
  - Kết luận một bên thắng từ thứ hạng GEFCom2012: loại, vì đó là thứ hạng không có chênh lệch điểm hay CI trong phần đã đọc, dữ liệu cấp vùng của Mỹ, có phần backcast.
  - Bỏ LSTM vì "DL kém": loại lập luận này. Bằng chứng không cho phép nói vậy; lý do bỏ là phạm vi.
  - Gọi cấu hình thời tiết trễ là "không phải ex-ante": loại, vì đi ngược fpp3. Cách gọi: dự báo ex-ante với predictor trễ, không dùng dự báo thời tiết. Đây là cách ghép, không phải nguyên văn fpp3.
- Phản biện mạnh nhất và cách trả lời:
  - "Phương pháp trung tâm thiếu tài liệu đối đầu giữa hồi quy và tree" (L.1): đã thêm mục so sánh, kết luận không phân định được, chỉ được phép nói hai câu về tính tồn tại. Tiền lệ tree trực tiếp cho GBM chỉ có Rubattu, cộng random forest của Dang-Ha.
  - "Ridge không lag có tiền lệ mạnh" là suy rộng từ bài month-ahead (L.2): đổi thành phép chẩn đoán, không gọi là cách làm chuẩn.
- Bằng chứng (thí nghiệm/nguồn):
  - `debate/litreview-v1.md`, `debate/litreview-critique-v1.md`, `debate/litreview-v2.md`, `debate/litreview-critique-v2.md`.
  - Chưa có thí nghiệm. Các số trong review đọc qua công cụ trích xuất, phải mở bảng gốc đối chiếu trước khi đưa vào báo cáo, trước hết là các bảng Incremona. Phản biện chưa xác nhận độc lập phần lớn các số này.
- Điều kiện còn mở:
  - Chưa quyết: 6 nguồn lõi chỉ đọc abstract có được giữ trong lõi không. Phản biện đề nghị người dùng tự đọc ít nhất Charlton & Singleton 2014, Lloyd 2014, Bouktif 2018 trước khi nộp.
  - Base learner của Lloyd và định nghĩa gốc của Vanilla chưa xác minh. Các giới hạn đọc phải ghi vào báo cáo nằm trong `debate/litreview-v2.md`.

## [2026-11-10] Quyết định: Đóng băng cấu hình trước khi chạy test 2018
- Chọn: tham số trong `results/frozen/` (Ridge chọn trên validation 2017 ở máy cá nhân; LightGBM 30 trial Optuna trên Kaggle, LightGBM 4.6.0). Đóng băng lúc 2026-11-10 23:32 (+07), **trước** mọi lần nạp dữ liệu 2018. Script 04 kiểm tra hash với `results/frozen/SHA256SUMS` và từ chối chạy nếu khác.
  - `ridge_actual.json` 999ba364a15991dba399ebcc0db6796c84b6d7166d44f430f9c239361ba6ee9e
  - `ridge_lagged.json` 75948cf8d6af6c77e9d43026673bb7ff44ea1c0e41492151b62b987b967fa652
  - `lgbm_actual.json` 77550690f481d4aa6e7124fe9951a1516c84cc2dfba6f289697874af3a6422ac
  - `lgbm_lagged.json` b7bd4b40ffca820a54efd7dd1dd2cbe7e07ab7ac1f966d468a9c1773cd73c274
- Vì sao: thầy chấm nặng "test chỉ chạm một lần"; hash + thời điểm + commit là bằng chứng thứ tự đóng băng rồi mới test.
- Kết quả validation (MAE, MW): B0 2627; B1 1840 (actual) / 1859 (lagged); M1 1591 / 1683. M1 tốt hơn B1 rõ ở cả hai cấu hình; LightGBM thua Ridge ở ngày lễ và ngày cầu nối.
- Auditor (methodology-auditor, một lần): không có lỗi chặn. Đã sửa trước test: feature `bridge` của ngày cuối dữ liệu (2018-12-31) không thấy được lễ 2019-01-01 (feature dev không đổi, đã kiểm); metric bỏ qua NaN của cột TSO; ghi đủ phiên bản gói. Ghi chú cho báo cáo: spike cleaning dùng giờ kế tiếp; `T_ma24` ở cấu hình actual gồm phần còn lại của D-1; tuning không cân bằng (Ridge 94 fit, LightGBM 30 trial + early stopping trên cùng validation); số cây giữ cố định khi refit; `min_data_in_leaf` sát biên dưới; actual vs lagged của Ridge gần giới hạn phân giải; refit một lần thay vì hằng tháng (lệch có chủ đích so với pipeline-v1); bảng lễ thiếu 5 ngày lễ quốc gia rơi vào Chủ nhật (2015-2017, test không bị ảnh hưởng).
- Điều kiện còn mở: chạy script 04 đúng một lần trên Kaggle; sync `test_run_log.json` về repo ngay sau đó.
- Lần chạy 04 thứ nhất trên Kaggle (2026-11-10) dừng ở bước import `src/config.py` với `FileNotFoundError` vì notebook chưa gắn dataset dữ liệu. Lỗi xảy ra trước khi đọc bất kỳ file dữ liệu nào, nên năm 2018 chưa bị chạm; không có `test_run_log.json`. Chạy lại sau khi gắn dataset, không đổi code hay tham số. Log lưu ở `results/logs/04_attempt1_failed_log.txt`.
