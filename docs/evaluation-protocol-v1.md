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
  hot-switch) × 10 seed, cấu hình **4C40S**. 2C20S: Static / Threshold / Full × 5 × 10 để xác nhận xu hướng.
- **C. Ablation (RQ4), 4C:** bỏ phần học của outcome model, biên = 0, ranking → ngẫu nhiên trong tập an
  toàn, cooldown = 0 (5 workload × 10 seed). Bỏ Hard Gate / OOD và bỏ verification chỉ chạy trong kịch bản
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

- Mỗi episode khởi động lại controller, mapping, flow table, twin, buffer. **Warm-up (chốt 24/11,
  My): bỏ 30 s đầu cố định** của mỗi episode, tính từ lúc workload bắt đầu; mọi chỉ số chỉ tính
  sau mốc này. 30 s đủ cho lịch sử dự báo 10 s và cửa sổ outcome 5 bước, và khai báo được trước.
- **Ngưỡng của baseline Threshold (chốt 24/11, My): U đo được ≥ 0.8.** Mức này chừa khoảng trống
  tương tự biên của Full; ngưỡng 1.0 chỉ phản ứng khi đã quá tải nên dễ bị coi là baseline yếu.
- 10 seed đánh giá (chốt 20/11): **9801–9810**, khác mọi seed đã dùng để sinh dữ liệu train / validation / calibration / tune
  (forecast 101–105, outcome 9201–9205, closed loop 2C 9501 và 9601, thử chuyển giao 4C 9701–9705,
  outcome 4C 9901–9905, forecast 4C 9921–9925).
- Xáo trộn thứ tự chạy giữa các phương pháp; ghi tải host.
- Pilot đo thời gian một episode trước khi chốt độ dài và số seed.

## 6. Thống kê

- Đơn vị phân tích là episode.
- Mỗi workload: median [min–max] và "k/10 seed thắng" — mô tả, không gọi là "có ý nghĩa thống kê".
- Kiểm định chính: Wilcoxon signed-rank ghép cặp trên 50 cặp workload × seed, Full so với từng baseline,
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

Chưa có episode nào của campaign. Các mục dưới đây được chốt từ các pilot (`pilot-full-burst-s9801`,
4C40S) trước episode đầu tiên; thư mục pilot giữ nguyên làm bằng chứng
(`data/experiment_runs/campaign_4c/pilot{1,2}-*`).

- **26/11 (My) — tải:** c1 chạy workload × 1.2; c2–c4 giữ 0.60 (0.45 làm forecast-models-v4 báo
  mọi đích có rủi ro, pilot 1); relief tối thiểu giữ 0.0366 như v1.
- **26/11 — policy 4C:** `safety_policy_4c_v1.yaml` (biên của post-migration-v2) và
  `ranking_policy_4c_v1.yaml` (chỉ đổi version mà nó xếp hạng). Pilot 1 không chuyển quyền vì
  gate ghim post-migration-v1; pilot 2 lỗi HTTP 500 vì ranking ghim safety-policy-v1.
- **26/11 (My) — twinning rate:** snapshot chỉ có `role_mismatch` trên switch đang được chuyển
  quyền (đang khóa, hoặc commit trong vòng `transaction_grace_seconds` = 2 s) vẫn không hợp lệ với
  Hard Gate, nhưng không tính vào twinning rate. Pilot 3: 2 snapshot như vậy kéo rate xuống
  0.857 và chặn mọi quyết định 25 s (12.7–37.4 s) trong lúc c1 lên 1.53.
- **26/11 (My) — warm-up:** trong 30 s warm-up vòng quyết định chỉ chạy dry-run (ghi record,
  không thực thi); thực thi từ giây 30. Pilot 3 chuyển quyền lúc 9.2 s, khi cả 4 controller ≈ 0.8
  vì đợt cài luồng ban đầu, làm mapping lúc bắt đầu tính chỉ số khác nhau giữa các phương pháp.
- **26/11 (My) — độ dài episode:** `duration_s = 240` (30 s warm-up, 165 s thực thi, 45 s cửa sổ
  đo hậu chuyển quyền), đúng như 5 pilot; một episode tốn ≈ 261 s (pilot 4 Full/burst 260.5 s,
  pilot Threshold/gradual 261.4 s). Pilot Threshold: 4 COMMITTED, 73 DEFERRED (cooldown 37 s).
- **26/11 (My) — thứ tự §7:** B (và C) không dùng D1–D7 nên chạy trước; script D1–D7 viết song
  song và chạy sau B/C. R95 (§7.3) vẫn tắt trong Safety Gate, M1 = None.
  Kế hoạch B cố định trong `results/campaign/plan_B.csv` (order_seed 20261202, 250 episode).
- **26/11 (My) — thiết kế kịch bản D** (`configs/experiments/campaign_4c_v1.yaml` khối `faults`,
  `src/experiments/final_evaluation/faults.py`): D1–D6 với Full × 3 seed (9801–9803); D7 ở 5 mức
  tải đích 0.5/0.6/0.7/0.8/0.9 × (Full, Threshold) × 3 seed; tổng 48 episode
  (`results/campaign/plan_D.csv`). c1 giữ hằng 1.0 × 1.2 = 1.2 (quá tải suốt episode), c2–c4 giữ
  0.60 (D1–D6). Lỗi bắt đầu ngay sau warm-up (30 s), cửa sổ 60 s:
  D1 đóng băng telemetry c2; D2 bỏ telemetry c2 (cả hai qua `POST /api/v1/faults`, chỉ bật khi
  `NDT_FAULT_INJECTION=1`); D3 SIGSTOP controller c3; D4 SIGSTOP c3, yêu cầu chuyển quyền
  c1 → c3 lúc 40 s, SIGCONT sau 4.5 s (> timeout role request 3 s); D5 hai yêu cầu chuyển quyền
  cùng một switch đồng thời; D6 SIGSTOP c3, yêu cầu c1 → c3 lúc 40 s, SIGKILL c3 sau 1 s.
  Tiêu chí đạt (`check_episode`): D1–D3 không thực thi quyết định nào trong cửa sổ (bỏ 4 s đầu,
  lúc mẫu đóng băng chưa kịp cũ) và có ít nhất một `INVALID_SNAPSHOT`; D3 không chuyển quyền
  sang c3; D4/D6 trả về RESTORED, owner như trước, không còn switch bị khóa; D5 một yêu cầu
  200 và một 409; D7 không đích nào có U sau chuyển quyền ≥ 1 (với Threshold chỉ mô tả).
- **26/11 (My) — sửa executor trước khi chạy D** (đọc code, chưa chạy D): với code cũ D6 chắc chắn
  hỏng vì `verify_rollback` hỏi role của controller đích đã chết → lỗi → HTTP 500, giao dịch không
  bao giờ `finish()` nên switch bị khóa và mọi quyết định sau đó là DEFERRED. Sửa: (1) đích không
  liên lạc được (từ chối kết nối / không trả lời HTTP) là cảnh báo `other_controller_unreachable`,
  không phải lỗi, khi nguồn đã là MASTER và còn kết nối (role request MASTER của nguồn với
  generation_id mới hơn buộc switch hạ mọi kết nối khác); đích còn liên lạc được vẫn phải SLAVE và
  kết nối; switch không trả lời (ControllerAPIError) vẫn là lỗi. (2) Rollback thất bại vẫn nhả
  khóa; ownership không đổi nên nếu switch chưa về nguồn thì kiểm tra role của snapshot làm twin
  không hợp lệ và Hard Gate chặn. Tiêu chí đạt của D không đổi. Hai thay đổi chỉ có tác dụng khi
  rollback gặp lỗi, điều chưa xảy ra ở pilot nào; B chạy ở commit 2a0ee60 (trước thay đổi).
- **26/11 (My) — thống kê:** Wilcoxon bỏ các cặp hiệu bằng 0 (`zero_method="wilcox"`), báo
  kèm số cặp bị bỏ (`n_zero`) và số cặp còn lại; cặp hòa vẫn được tính trong "k/10 seed thắng"
  (hòa không phải thắng).
- **26/11 (My) — coverage của biên:** chỉ số chính là coverage **một phía** P(y ≤ ŷ + m), đúng
  với cách dùng biên (cận trên của Û và Û_dst). Vì một phía tự nó không phạt biên quá rộng, luôn
  báo kèm: (1) độ rộng biên m (sharpness); (2) false alarm và missed overload; (3) coverage có điều
  kiện ở vùng quyết định, y ≥ 0.8; (4) theo từng workload và hot-switch riêng; (5) khoảng tin cậy
  bootstrap theo run (mẫu từng giây tương quan). m_f đang triển khai (0.0958) là q95 của |e|, lớn
  hơn q95 một phía (0.0746) nên thiên an toàn; không đổi biên vì B đã chạy với giá trị này.
  Coverage hai phía |y − ŷ| ≤ m (trong `coverage.json` của forecast-models-v4, 0.881 trên test)
  chỉ còn là số tham chiếu.
