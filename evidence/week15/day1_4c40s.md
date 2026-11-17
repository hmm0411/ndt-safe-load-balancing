# Day 1: hạ tầng 4C40S

Chạy ngày 2026-11-17 (giờ VM 10:08 đến 10:24). Commit: 38c9f17, 5ded632, a5290fc.

## Kiểm tra hệ thống 4C
- 4 controller (OpenFlow 6653 đến 6656, REST 8081 đến 8084), mỗi controller thấy 40 switch.
- 40/40 switch có đúng 1 MASTER và 3 SLAVE.
- MASTER mỗi controller: c1 10, c2 10, c3 10, c4 10.
- init-roles trả ownership khớp configs/controllers_4c40s.yaml.
- twin/state: 4 controller, 160 dòng switch (40 switch x 4 controller), valid=true, quality_score 1.0.
- 2C không hồi quy: collect_forecast_data.sh --dry-run ra 25 run, DRY_RUN: PASS.

## Collection latency 4C
Hệ thống rảnh, không có workload. 300 snapshot hợp lệ, 1200 quan sát, 299.143 s.
Lệnh: benchmark_collection_latency.py --controllers c1,c2,c3,c4 --count 300 --timeout 600

| Controller | Quan sát | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|---|
| c1 | 300 | 468.8 | 676.3 | 698.4 |
| c2 | 300 | 300.0 | 522.0 | 558.4 |
| c3 | 300 | 556.9 | 758.8 | 786.2 |
| c4 | 300 | 568.0 | 770.0 | 797.5 |
| Gộp | 1200 | 480.3 | 743.3 | 797.5 |

Gộp: mean 472.839 ms. Không chạm 1000 ms (chu kỳ lấy mẫu và request_timeout_seconds).
Mốc 2C trong bản rà soát là P95 0.98 đến 0.99 s, chưa rõ điều kiện tải khi đo, nên chưa so sánh trực tiếp.
c2 thấp hơn ba controller còn lại; chưa rõ nguyên nhân.
Dữ liệu thô: collection_latency_4c.jsonl.gz (giải nén ra file 15 MB, 300 dòng).
sha256 của file giải nén: f4181c4bd48141b62366f8ed34c5c34356443cc4c20923b5a6ff2934d31ae2ae

## Giới hạn
- Capacity của c3 và c4 là tạm: sao từ c1 và c2 của capacity-v4 (file ngoài git, sha256 677ee0fa9d0731af5a6f6564b54e73722a1a7efca580f31bb67239ae59f081b0), chỉ để kiểm tra đường truyền. U của lần chạy này chưa dùng được; tính lại sau capacity-v5 (Day 4).
- Thời gian hồi phục cửa sổ twinning sau init-roles chưa đo: lần đọc đầu đã valid=true.
- Orchestrator cần NDT_CAPACITY_FILE (commit a5290fc) vì capacity-v4 chỉ có c1 và c2; không đặt biến thì đường dẫn như cũ.
