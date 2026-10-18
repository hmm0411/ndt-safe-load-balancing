# CLAUDE.md — ndt-safe-load-balancing (KLTN SDN-NDT)

> File này nằm ở **gốc repo**. Claude Code đọc nó ở đầu mỗi session. Giữ ngắn, đúng, cập nhật khi
> repo thay đổi (cột "Trạng thái" bên dưới là thứ hay lỗi thời nhất).

## 0. Cách làm việc với người dùng

- Người dùng: **Vy** (Trần Ngọc Thúy Vy). Đồng đội: **My** (Nguyễn Lê Hạ My), chủ repo GitHub, review PR.
- **Trả lời bằng tiếng Việt.** Chỉ dùng tiếng Anh cho identifier, tên field, tên lệnh, nhãn kỹ thuật.
- Vy phụ trách: hạ tầng SDN, Central Orchestrator, Switch Migration, Safety Gate, thực thi hệ thống.
  My phụ trách phần thuật toán/mô hình. Code đụng vào phần của người kia → nói rõ trong tóm tắt.
- Trước khi sửa code: đọc file liên quan và test của nó; không đoán shape JSON, đọc `src/schemas/`.
- Kết thúc mỗi việc: liệt kê file đã đổi, lệnh đã chạy + kết quả, việc **chưa** kiểm chứng được
  (vd. cần Mininet/sudo trên VM). Không tuyên bố "đã chạy được" nếu chỉ mới chạy unit test.
- Hình cho khóa luận: đơn giản, sắc nét, kiểu học thuật (matplotlib/TikZ phẳng). Không gradient,
  không bóng đổ, không phong cách "AI-generated".

## 1. Đề tài (tóm tắt đề cương)

**A network digital twin-assisted framework for safe load balancing in multi-controller SDN.**
GVHD: ThS. Nguyễn Khánh Thuật. Thời gian 07/09/2026 – 26/12/2026; giữa kỳ ~23/11.

Vòng khép kín cần xây:

```
SDN thật (Ryu C1..Cn, OVS/Mininet)
 → (1) Telemetry 1 s + State Synchronizer → TwinState X_t, chỉ dùng khi Valid = Fresh ∧ Complete ∧ Consistent
     (+ Quality Score: bổ sung, KHÔNG thay Valid)
 → (2) Future Load Predictor: Û(t+h) + m_f ≥ 1 ⇒ rủi ro quá tải
 → (3) Candidate Generator a = (s, c_src, c_dst)  +  Post-Migration Outcome Model Ŷ(a) = F_θ(X_t, a)
       (ước lượng phân tích + học phần dư) → U_src/U_dst, R95, imbalance dự đoán
 → (4) Safety Gate: Û_post_dst + m_U ≤ U_safe_dst  và  R̂95_post_dst + m_R ≤ R95_max, + cooldown,
       xung đột giao dịch, cải thiện tối thiểu; tập hành động LUÔN có NO_MIGRATION → Ranking
 → (5) Migration Executor: Role Request/Reply + Barrier → Verification → commit hoặc rollback
 → (6) Feedback & Calibration: so X_actual(t+Δ) với dự đoán → cập nhật sai số và biên
```

Giới hạn (đừng mở rộng phạm vi nếu không được yêu cầu): tối đa **1 migration / chu kỳ**; không
controller placement, không routing; orchestrator là logic tập trung (không consensus); ML chỉ cho
dự báo tải và dự đoán hậu migration; không làm drift detection + retrain tự động; "an toàn" = an toàn
vận hành, không phải bảo mật.

Workload: stable, gradual, burst, oscillating, hot-switch. Kịch bản lỗi: snapshot cũ, thiếu dữ liệu,
controller mất kết nối, timeout, xung đột giao dịch. Baseline: static, threshold (reactive),
forecast-only, NDT không có feedback. Cấu hình: 2 controller × 20 switch và 4 × 40.

## 2. Bản đồ repo và trạng thái

| Thành phần (đề cương) | Vị trí | Trạng thái |
|---|---|---|
| Controller Ryu, role, barrier, telemetry REST | `src/controller/` | Có |
| Orchestrator FastAPI (ownership, transaction, state) | `src/orchestrator/` | Có |
| Telemetry, snapshot builder/validator | `src/telemetry/` | Có |
| TwinState + quality / twinning rate | `src/twin/state/` | Có |
| Safe capacity model | `src/twin/capacity/`, `data/benchmarks/controller_capacity.json` | Có (capacity-v3, e2-standard-8) |
| Forecasting (features, splitter, baselines, LR/RF/XGB, MLflow) | `src/twin/forecasting/`, `scripts/experiments/train_forecast_baselines.py` | Có, `forecast-models-v2` |
| Overload detector (`upper_bound = pred + margin ≥ 1`) | `src/twin/decision/overload_detector.py` | Có |
| Migration / verification / rollback | `src/execution/` | Có |
| Candidate Generator | — | **Chưa có** |
| Post-Migration Outcome Model | — (`data/outcome/` mới là thư mục rỗng) | **Chưa có** |
| Safety Gate, Ranking & Selection | — | **Chưa có** |
| Feedback & Calibration (cập nhật biên) | — | **Chưa có** |
| Baselines vòng kín, ablation, kịch bản lỗi | — | **Chưa có** |
| Dashboard | `dashboard/index.html` | Bản tĩnh |

Module mới cho các phần "Chưa có" nên đặt trong `src/twin/decision/` (candidate, gate, ranking),
`src/twin/outcome/` (outcome model), `src/twin/calibration/` (feedback) — trừ khi Vy chỉ định khác.
Schema dùng chung đặt trong `src/schemas/` và phải được cả hai người review.

Dọn repo (10/2026): đã xóa `h1`, `sh`, `configs/experiments/runners/forecast_data_runner.py` (rỗng,
đặt nhầm chỗ; bản thật ở `src/experiments/runners/`), và các package rỗng không ai import
`src/experiments/validation/`, `src/experiments/analysis/`. Đừng tạo lại; validation thật nằm trong
`src/experiments/capacity/validator.py` và `src/twin/forecasting/raw_validator.py`.

File rỗng **cố ý giữ** — không xóa:
- `__init__.py` rỗng trong các package đang được import (cần cho `from src....`).
- `.gitkeep` trong `data/**` và `logs/ci/` (giữ cấu trúc thư mục đã gitignore).
- `evidence/twin_state/git_diff_check.txt` (rỗng = `git diff --check` không lỗi, là bằng chứng).

Còn chờ Vy/My quyết, không tự xóa: `APPLY_PATCH.md`, `PATCH_NOTES.md` (trùng nội dung với README và
`docs/capacity-benchmark.md`),
`forecasting_prediction_plots.zip` (plot sinh lại được bằng `plot_forecast_baselines.py`).
Không đụng `data/benchmarks.zip` và mọi thứ trong `data/`, `evidence/`.

Nguyên tắc chung: không tạo file/package rỗng "để dành", không commit file scratch ở gốc repo
(dùng `/tmp` hoặc scratchpad), không để code trong `configs/`.

## 3. Môi trường

- VM GCP `kltn-sdn` (zone `asia-southeast1-b`), **e2-standard-8**, Ubuntu, Python 3.10.
- **VM dùng chung**: Vy và My SSH cùng một Linux user, cùng một thư mục repo. Hậu quả:
  - Trước khi chạy gì đụng tới Mininet/OVS/port, kiểm tra người kia có đang chạy không:
    `pgrep -af "ryu-manager|uvicorn|mininet|mn "; ss -ltnp | grep -E ':(6653|6654|8081|8082|9000|5000)'`
  - `sudo mn -c` / `scripts/cleanup.sh` **giết run của người kia** → hỏi trước khi chạy nếu thấy tiến trình đang sống.
  - Không `git checkout`/`reset`/`stash` khi working tree đang có thay đổi chưa rõ của ai; báo lại thay vì xử lý.
- Hai venv tách biệt:
  - `~/ryu-venv` — Ryu 4.34 (chỉ cho controller; Ryu trên py3.10 cần chuỗi cài đặt đặc biệt, **đừng nâng cấp gói trong venv này**).
  - `~/ndt-venv` — orchestrator, test, ML (`requirements-orchestrator.txt`, `-dev.txt`, `-ml.txt`).
- Luôn `export PYTHONPATH=$PWD` từ gốc repo khi chạy script (import dạng `from src....`).
- Ports: C1 OF 6653 / REST 8081; C2 OF 6654 / REST 8082; orchestrator 9000; MLflow 5000; dashboard 8080.
- Sandbox cloud của Claude Code (không phải VM) **không có** Mininet/OVS/sudo/Ryu: ở đó chỉ chạy được
  unit test, lint, typecheck, train/analysis trên dữ liệu đã có. Nói rõ khi một bước phải chạy trên VM.

## 4. Lệnh thường dùng

```bash
source ~/ndt-venv/bin/activate && export PYTHONPATH=$PWD

make ci                 # compile + ruff + mypy + unittest (chạy trước mọi commit)
python -m unittest tests.unit.test_overload_detector -v    # một file test
make smoke              # smoke 2C-4S thật — CHỈ trên VM, cần sudo
make clean-sdn          # sudo mn -c — xem cảnh báo VM dùng chung
```

CI (`.github/workflows/ci.yml`) gọi đúng các target của `make ci` → local xanh thì CI xanh.
Phạm vi: ruff `src scripts tests`, mypy `src tests/unit` (toàn bộ, kể cả `src/twin`, `src/execution`,
`src/controller`). Code mới phải qua cả hai; không thu hẹp phạm vi hay thêm `# type: ignore` để
cho qua — sửa lỗi thật. Job `repo-guards` còn chặn: commit `.env`/`mlflow.db`/`mlruns/`/`data/raw|snapshots|experiment_runs`,
file > 25 MB, artifact capacity/model JSON hỏng hoặc thiếu so với `MODEL_VERSION`, và kế hoạch 25 run
(`collect_forecast_data.sh --dry-run`) sai. Đổi `MODEL_VERSION` → phải commit đủ manifest của version mới.
Test dùng **`unittest`**, không phải pytest (~327 test).
`test_capacity_2c20s_topology` import `mininet` nên lỗi ImportError ở máy không có Mininet (sandbox);
đó là lỗi môi trường, không phải regression — trên VM phải xanh.

Chạy hệ thống thủ công (mỗi lệnh một terminal/tmux pane):

```bash
./scripts/start_c1.sh ; ./scripts/start_c2.sh ; RUN_ID=dev ./scripts/start_orchestrator.sh
sudo python3 src/experiments/topologies/smoke_2c4s.py          # hoặc runtime_2c20s.py
curl -fsS -X POST http://127.0.0.1:9000/api/v1/init-roles      # TRƯỚC khi tạo traffic
curl -fsS http://127.0.0.1:9000/api/v1/twin/state | python3 -m json.tool
curl -fsS -X POST http://127.0.0.1:9000/api/v1/migrations -H 'Content-Type: application/json' \
  -d '{"switch_id":"s1","target_controller":"c2","simulate_failure":"none"}'   # hoặc "flow_mod"
```

Migration chỉ COMMITTED khi target nhận **Packet-In mới** trong lúc VERIFYING (xóa flow priority=10 rồi ping).

Dữ liệu dự báo và huấn luyện:

```bash
scripts/experiments/collect_forecast_data.sh --dry-run                 # xem kế hoạch 25 run
scripts/experiments/collect_forecast_data.sh --single stable 1         # 1 run (trên VM)
scripts/experiments/collect_forecast_data.sh --all --skip-existing
python3 scripts/experiments/validate_forecast_collection.py
python3 scripts/experiments/build_forecast_dataset.py && python3 scripts/experiments/build_forecast_splits.py
python3 scripts/experiments/train_forecast_baselines.py --run-name <tên> --note "<mô tả>" [--no-mlflow]
python3 scripts/experiments/plot_forecast_baselines.py
```

Collection có **environment freeze**: cần git sạch + tag, fingerprint VM/phiên bản/sha capacity/config.
Đừng "sửa nhanh" để vượt qua kiểm tra này; nếu bị chặn thì báo lý do.

## 5. Quy ước dữ liệu và phiên bản

- Version tập trung ở `src/twin/forecasting/versions.py`: `DATASET_GENERATION="v2"`,
  `MODEL_VERSION="forecast-models-v2"`, dataset `forecast-dataset-v2-h3-provisional`.
  Đổi dữ liệu/mô hình theo cách không tương thích → tăng version ở đây, **không ghi đè** thư mục cũ.
- Artifact đã đóng băng — không sửa tay: `data/benchmarks/controller_capacity.json`,
  `data/benchmarks/capacity_runs.csv`, `data/models/forecasting/*/`, `evidence/`.
  `data/experiment_runs_invalid/` là bằng chứng run hỏng, giữ nguyên.
- Run id: `{workload}-r{NN}-s{seed}`, seeds 101–105 (repeat N ↔ seeds[N-1]); 180 s/run, 2 controller.
- Split cố định theo run: train repeat 1–3, validation 4, test 5; **hot-switch = unseen**, không bao giờ
  dùng cho train, chọn mô hình hay tính biên. Không xáo trộn theo dòng.
- U = processed Packet-In rate / safe capacity (c_safe ≈ 1433 pps mỗi controller). Ngưỡng quá tải U = 1.
- Horizon h = 3 bước (1 s/bước) là **provisional** (`configs/forecasting/forecast.yaml`): còn thiếu
  decision p95. Khi có decision latency đo được thì tính lại theo `estimate_horizon()`.
- Mô hình cây dự báo **ΔU = U(t+h) − U(t)**; forecast = U(t) + predict(X). `managed_switch_count` và
  `max_switch_control_load_share` là AUDIT_ONLY, không đưa vào mô hình (lý do trong `features.py`).
- Siêu tham số nằm trong `configs/forecasting/models.yaml`, không hardcode trong code.
- MLflow: store `sqlite:///mlflow.db` (đã gitignore; `docs/mlflow.md` nói "committed" là lỗi thời).

## 6. Nguyên tắc đánh giá (bắt buộc khi viết code thí nghiệm/phân tích)

Kết quả hiện tại (test, h=3): persistence RMSE 0.098, LR 0.070, RF 0.042, XGB 0.041; P95|e| persistence
0.178 → XGB ~0.043. Trên hot-switch thứ hạng đảo: MA3 0.025 < persistence 0.029 < LR < RF 0.060 < XGB 0.065.

- Luôn báo cáo **tỉ số RMSE/skill so với persistence**, theo **từng workload** + macro-average.
  Ghi rõ skill dạng RMSE hay MSE. Bỏ R² và MAPE khỏi bảng theo workload/OOD.
- RF và XGB ~ tương đương: không viết "XGB tốt nhất" nếu chưa có Diebold–Mariano (HLN, h=3) + Holm.
- Kết quả hot-switch là kết quả âm phải báo cáo thẳng, kèm cả tỉ số OOD/ID và tỉ số vs persistence.
- Biên m_f / m_U / m_R tính trên **validation**, theo kiểu split conformal: quantile finite-sample
  ⌈(n+1)(1−α)⌉/n, ưu tiên residual có dấu một phía (cận trên). Phải **đo coverage** trên test từng
  workload và trên hot-switch, kèm false alarm / missed overload.
- Dự báo chỉ được **thêm** cảnh báo, không được **xóa** cảnh báo sinh từ U đo được.
- Hướng đã định: cổng OOD → fallback persistence; biên thích nghi (ACI). Có thể thêm một GRU nhỏ làm hàng
  so sánh. Không thêm LightGBM/ARIMA/Transformer trừ khi được yêu cầu.
- Thí nghiệm vòng kín: mỗi episode reset hoàn toàn (controller, mapping, flow table, twin, buffer);
  bỏ warm-up theo quy tắc khai báo trước; xáo trộn thứ tự cấu hình; **seed đánh giá ≠ seed sinh dữ liệu
  train/val/calibration/tune**. Đơn vị phân tích là episode, không phải từng giây.
- Ablation: loại bỏ thành phần bằng **phiên bản rỗng vẫn chạy được** (forecast → persistence, margin → 0,
  gate → accept-all, ...), thứ tự cumulative theo quan hệ phụ thuộc.
- Thống kê: 5 seed → Wilcoxon hai phía không thể < 0.05; kiểm định chính gộp 25 cặp workload×seed,
  từng workload báo cáo median, min–max, "k/5 seed thắng".
- Mọi số liệu đưa vào báo cáo phải sinh ra từ script + file JSON/CSV trong repo, truy vết được.

Chi tiết lập luận: tài liệu research trong Project claude.ai "KLTN"
(`research-du-bao-tai-controller-sdn.md`, `research-chon-mo-hinh-va-ablation.md`).

## 7. Quy ước code và Git

- Python 3.10, `from __future__ import annotations`, type hints đầy đủ (mypy `check_untyped_defs`),
  ruff line-length 100. Dataclass frozen cho kết quả; validate input và raise `ValueError` rõ nghĩa
  (xem `overload_detector.py` làm mẫu). Không clamp U về [0,1].
- Mỗi module mới có test tương ứng `tests/unit/test_<module>.py` (unittest, không cần mạng/sudo).
  Logic cần SDN thật → tách phần thuần (testable) khỏi phần gọi Ryu/OVS.
- Config trong `configs/*.yaml`; không hardcode ngưỡng, port, đường dẫn.
- Branch: `main` (milestone), `dev` (tích hợp). Làm trên `feat/<tên>` hoặc `wNN/<tên>` tách từ `dev`
  mới nhất; PR vào `dev`, My review, merge khi CI xanh. **Không push thẳng `dev`/`main`, không force-push
  branch người khác đang dùng.**
- Commit nhỏ, message dạng `feat(scope): ...`, `fix(scope): ...`, `exp(scope): ...`. Không commit `.env`,
  `mlflow.db`, `mlruns/`, `data/experiment_runs/`, file lớn sinh ra.
- PR dùng `.github/pull_request_template.md`: Summary / Changes / Compatibility / Verification (đánh dấu
  đúng cái đã chạy) / Evidence.

## 8. Khi nhận yêu cầu

1. Xác định yêu cầu thuộc khối nào ở mục 1–2 và chạy ở đâu (sandbox hay VM).
2. Đọc code + test + config liên quan, nêu kế hoạch ngắn nếu thay đổi > 1 file hoặc đụng `src/schemas/`.
3. Viết code + test, chạy `make ci` (hoặc phần liên quan), sửa đến khi xanh.
4. Với việc cần VM: viết sẵn lệnh chính xác để Vy chạy và nói kết quả mong đợi để đối chiếu.
5. Tóm tắt bằng tiếng Việt: đã đổi gì, đã kiểm chứng gì, còn gì chưa kiểm chứng, bước tiếp theo.
