# Tiêu chí đạt / không đạt của probe độ trễ thiết lập luồng (Day 3)

Ghi ngày 18/11/2026, **trước khi đọc** kết quả lượt đo `probe_sweep_4c40s` / `probe_off_4c40s`
(đang chạy trên VM). Các ngưỡng nằm trong `configs/experiments/probe_acceptance_v1.yaml`;
`scripts/analysis/probe_validation.py` chấm tự động và ghi `evidence/week15/probe_validation.json`.
Sửa ngưỡng sau khi đã có `probe_sweep_summary.json` thì phải tạo v2 và ghi rõ lý do.

Phần này thuộc việc của My trong phân công tuần 16–22/11 (Day 2). Vy soạn theo đúng năm tiêu chí
trong bản phân công; các con số cụ thể là **đề xuất**, My xác nhận trước khi chạy script chấm.

## Câu hỏi

Probe một-switch (`p<i> → h<i>` qua đúng `s<i>`) có đo được độ trễ phục vụ của **từng controller**
theo tải, để dùng làm R95 trong Safety Gate (Hướng 3) không?

## Năm tiêu chí

| # | Tiêu chí | Cách đo | Ngưỡng | Vì sao |
|---|---|---|---|---|
| P1 | **Tách controller** | Khi chỉ một controller bị tăng tải (90/140 fps), P95 của ba controller kia so với mức nền của chính chúng | ≤ 1.5 × mức nền | R95 phải gắn được với đúng controller; nếu tải ở c1 làm tăng R95 của c2 thì không dùng được cho quyết định chọn đích |
| P2 | **Đơn điệu** | P95 của controller bị tăng tải ở 30 → 90 → 140 fps | tăng chặt, cả 4 controller | Một chỉ số an toàn phải tăng theo tải; R95 cũ (thời gian trong handler) không đạt điều này |
| P3 | **Thấy được bão hòa** | Ở 140 fps: số timeout, hoặc P95 so với mức nền | timeout > 0 **hoặc** ≥ 2 × mức nền; không bao giờ 0 hay thiếu | Điểm gãy của capacity-v4 là 140 fps; probe phải nhìn thấy nó |
| P4 | **Tải do probe gây ra** | Chênh lệch U của từng controller khi bật và tắt probe, mọi controller ở 30 fps | \|ΔU\| ≤ 0.02 | Probe không được làm thay đổi đại lượng mà nó đo |
| P5 | **Đủ mẫu** | Số mẫu mỗi run, mỗi controller | ≥ 95% × (10 switch × 1/s × 60 s) = 570 | Đủ để P95 ổn định (≈ 100 mẫu cho một cửa sổ 10 s) |

Mức nền của một controller = trung bình P95 của nó trong mọi run mà cả bốn controller đều ở 30 fps.
ΔU tạm tính theo thang capacity-v4 (933 pps) cho tới khi có capacity-v5; file kết quả ghi rõ thang.

## Quyết định

- Đạt cả năm → **Hướng 3**: R95 = P95 độ trễ thiết lập luồng của controller đích, đo bằng probe;
  ngưỡng R95_max tính từ capacity-v5 theo luật đã ghi trong `capacity_v5_rules.yaml`.
- Trượt bất kỳ tiêu chí nào → **Hướng 2** (giữ ràng buộc R95 tắt, ghi là hạn chế). Không nới tiêu chí sau
  khi đã thấy số.
