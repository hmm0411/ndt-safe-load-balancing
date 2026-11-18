# Evaluation protocol v1 (BẢN NHÁP, 18/11/2026)

Giao thức đánh giá hệ thống, viết **trước** campaign so sánh (02–08/12). Mọi thay đổi sau khi đã
xem kết quả campaign phải ghi vào mục "Thay đổi" kèm lý do. Owner: My (viết), Vy (review);
bản nháp này do Vy soạn từ kế hoạch đã thống nhất, My chỉnh và chốt.

## 1. Câu hỏi nghiên cứu (theo đề cương, mục 4 và Nội dung 4)

| RQ | Câu hỏi | Khía cạnh đề cương |
|---|---|---|
| RQ1 | NDT có phản ánh (state fidelity) và dự đoán (predictive fidelity) trạng thái mạng đủ tin cậy cho quyết định cân bằng tải không? | Sai số mô hình |
| RQ2 | Hệ đề xuất có cân bằng tải và giữ hiệu năng tốt hơn các phương pháp đối chứng không? | Hiệu năng, cân bằng tải |
| RQ3 | Hệ có an toàn không: không migration gây hại, phát hiện lỗi, rollback đúng? | Độ an toàn |
| RQ4 | Thành phần nào tạo ra lợi ích? | Tính mới (tích hợp một vòng) |
| RQ5 | Chi phí thời gian và tài nguyên? | Chi phí xử lý |

## 2. Chỉ số (một giá trị mỗi episode; code: `src/experiments/final_evaluation/metrics.py`)

| # | Chỉ số (đề cương) | Định nghĩa |
|---|---|---|
| M1 | Thời gian phản hồi | P95 độ trễ thiết lập luồng từ probe (R95), kèm CDF |
| M2 | Thời gian quá tải | Số giây có ít nhất một controller U ≥ 1; kèm OE = Σ_c Σ_t max(0, U−1)·Δt |
| M3 | Mức mất cân bằng | Trung bình theo thời gian của độ lệch chuẩn tổng thể U giữa các controller |
| M4 | Số lần chuyển quyền | Số migration tới bước thực thi; kèm số lần qua lại A→B→A trong 60 s |
| M5 | Tỷ lệ thành công | COMMITTED / số migration tới bước thực thi |
| M6 | Số lần khôi phục | Số RESTORED; kèm tỉ lệ rollback đúng (ownership và kết nối như trước) |
| M7 | Sai số dự đoán | Dự báo: tỉ số RMSE so với persistence theo workload. What-if: RMSE U_dst, R95_dst |
| M8 | Thời gian ra quyết định | P95 mỗi vòng, theo từng bước; lead time so với h |
| S1 | Tỷ lệ migration không an toàn (DOR) | Migration COMMITTED có U_dst sau > 1 hoặc R95_dst sau > R95_max, trên số COMMITTED có trạng thái sau |
| S2 | Coverage của biên | Tỉ lệ giá trị thật ≤ dự đoán + biên (m_f, m_U, m_R) |
| S3 | Recall phát hiện lỗi | Lỗi phát hiện / lỗi đã tiêm, đối chiếu file ghi lỗi tiêm; kèm tỉ lệ báo hỏng nhầm khi chạy sạch |

Chỉ số không tính được trả về None, không bao giờ 0. U không bị chặn ở 1.

## 3. Phương pháp so sánh (cố định)

| Phương pháp | Định nghĩa |
|---|---|
| Static | Không migration |
| Threshold | U đo được ≥ ngưỡng → switch nặng nhất sang đích nhẹ nhất, có kiểm tra chỗ trống đơn giản ở đích |
| Forecast-only | Kích hoạt theo Û + m_f ≥ 1, cùng heuristic chọn; không what-if, không Safety Gate đầy đủ |
| NDT không feedback | Toàn pipeline; biên không được hiệu chỉnh bằng phản hồi |
| Full | Hệ đề xuất |

Random / Greedy chỉ là baseline phụ, không vào bảng chính.

## 4. Kịch bản

- **A. Thành phần (RQ1):** capacity; dự báo (so với persistence, hot-switch giữ riêng); what-if ở 5 mức tải
  đích; coverage của biên; Hard Gate; executor.
- **B. So sánh chính (RQ2, RQ3):** 5 phương pháp × 5 workload (ổn định, tăng dần, đột biến, dao động,
  hot-switch) × 6 seed, cấu hình **4C40S**. 2C20S: Static / Threshold / Full × 5 × 6 để xác nhận xu hướng.
- **C. Ablation (RQ4), 4C:** bỏ phần học của outcome model, biên = 0, ranking → ngẫu nhiên trong tập an
  toàn, cooldown = 0 (5 workload × 6 seed). Bỏ Hard Gate / OOD và bỏ verification chỉ chạy trong kịch bản
  lỗi tương ứng (chúng không hoạt động khi không có lỗi). "Bỏ dự báo" và "bỏ feedback" đã nằm trong
  chuỗi baseline.
- **D. Lỗi (RQ3), 4C, Full:** D1 snapshot cũ, D2 thiếu dữ liệu, D3 controller mất kết nối, D4 quá thời
  gian chờ, D5 xung đột giao dịch, D6 controller đích chết giữa giao dịch, D7 hot-spot không có đích an
  toàn (kèm quét cường độ 50–90%).
- **E. Chi phí (RQ5):** từ log của B.
- **F/G (phụ):** oracle lấy mẫu (regret); độ nhạy của biên (replay offline), h (offline), cooldown (C).

### Hành vi đúng khai báo trước (Full)

| Kịch bản | Hành vi đúng |
|---|---|
| Ổn định | Gần như 0 migration, không kém Static |
| Tăng dần | Migrate trước khi U chạm 1; M2 nhỏ hơn Threshold |
| Đột biến | Không migrate theo gai tải tự hết; S1 = 0 |
| Dao động | M4 nhỏ hơn Threshold, không qua lại |
| Hot-switch | Chuyển đúng switch nóng nếu đích còn chỗ |
| D1, D2 | Không quyết định trên snapshot không hợp lệ |
| D3 | Không chọn đích mất kết nối |
| D4, D6 | Rollback, ownership và kết nối như trước |
| D5 | Từ chối giao dịch thứ hai |
| D7 | NO_MIGRATION (Threshold sẽ migrate và làm đích quá tải) |

## 5. Quy trình

- Mỗi episode khởi động lại controller, mapping, flow table, twin, buffer; bỏ warm-up theo quy tắc khai
  báo trước (đến khi snapshot hợp lệ đầu tiên + cửa sổ twinning đầy).
- 6 seed đánh giá, khác mọi seed đã dùng để sinh dữ liệu train / validation / calibration / tune
  (forecast 101–105, outcome 9201–9205, closed loop 2C 9601). Seed đánh giá sẽ chọn và ghi ở đây.
- Xáo trộn thứ tự chạy giữa các phương pháp; ghi tải host.
- Pilot đo thời gian một episode trước khi chốt độ dài và số seed.

## 6. Thống kê

- Đơn vị phân tích là episode.
- Mỗi workload: median [min–max] và "k/6 seed thắng" — mô tả, không gọi là "có ý nghĩa thống kê".
- Kiểm định chính: Wilcoxon signed-rank ghép cặp trên 30 cặp workload × seed, Full so với từng baseline,
  hiệu chỉnh Holm cho 4 so sánh trên M2; kèm effect size (matched rank-biserial). Ablation và chỉ số phụ là
  mô tả.
- Khi báo % cải thiện luôn kèm giá trị tuyệt đối.

## 7. Điều kiện trước campaign

1. Workload gây quá tải thật, có D7.
2. Script tiêm lỗi D1–D7.
3. R95 từ probe vào Safety Gate (policy v2) nếu Day 3 đạt (`evidence/week15/probe_acceptance.md`).
4. Script campaign: reset, xáo trộn, seed.
5. Pilot thời gian episode.

## Thay đổi

(chưa có)
