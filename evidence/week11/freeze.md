# Week 11: pilot and freeze of the migration-outcome collector

## Pilot
- `migration-pilot-d120` (Vy, VM): 15 MIGRATE + 2 NO_OP, phase giữa của 5 workload, cả 2 hướng, cả 2 mức tải đích.
  Chạy trên 3c71270 kèm 2 thay đổi chưa commit (tắt kiểm tra tải hot-switch cho MIGRATE, workload 120 s);
  `git diff 3c71270 4e2acf6` chỉ gồm 2 thay đổi này và test.
  Kết quả: 17/17 valid, 15 COMMITTED, Gate Day 3 PASS, không có lỗi do code (`pilot_report.md`, `pilot_validate.json`).
- `migration-pilot-late` (4e2acf6, tree sạch): 2 attempt `stable:late`, 2/2 COMMITTED, cửa sổ sau đủ,
  settle 4.2 s và 4.9 s (`pilot_late.md`).
- Attempt hot_switch c2->c1 lỗi trước khi sửa nằm trong `_invalid/` của pilot (giữ làm bằng chứng).

## Quyết định đóng băng
| Thông số | Giá trị | Căn cứ |
| --- | --- | --- |
| Cách chạy | mỗi attempt một lượt | ước lượng 340 attempt: median 12.5 h, p95 14.8 h, ngân sách 38.4 h |
| Thời lượng workload | 120 s cho cả 5 workload | 19/19 cửa sổ sau đủ, kể cả `stable:late` |
| `post_window_timeout_s` | 60 | settle max 39.6 s × 1.5 ≈ 60 |
| NO_OP | 40 | giữ theo kế hoạch |
| hot_switch_share | 0.25 / 0.40 / 0.55 | chốt ở Day 1 (#72) |
| Seeds | r01–r05 = 9201–9205 | `configs/experiments/migration_outcome.yaml` |
| Thu chính thức | chỉ trên tag `w11-collect-freeze`, tree sạch, không dùng `--duration` hay `--allow-dirty` | |

## Ghi chú cho Data Card / Tuần 12
- Sau migration, Hard Gate chặn median 10 và tối đa 36 snapshot (R7 Tuần 10: khoảng 2). Đây là chi phí của migration và liên quan cooldown của Safety Gate.
- Với workload không ổn định (burst_1, oscillating peak, gradual u60), ΔU quan sát là tác động migration cộng xu hướng workload
  (ΔU nguồn bằng 0.03–5.4 lần tải switch). stable và hot_switch khớp tải switch. Mô hình outcome phải tách phần xu hướng (NO_OP hoặc dự báo).
- Phase `early` có giai đoạn khởi động: `delta_*_mean` và `delta_*_last` khác nhau rõ (smoke: −0.007 và −0.071).
