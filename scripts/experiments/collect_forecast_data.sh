#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiments/forecast_dataset_2c20s.yaml"

CAPACITY_RUNS="data/benchmarks/capacity_runs.csv"
CONTROLLER_CAPACITY="data/benchmarks/controller_capacity.json"
ORCHESTRATOR_URL="http://127.0.0.1:9000"

DRY_RUN=0
MODE=""
WORKLOAD=""
REPEAT=""
SKIP_EXISTING=0

usage() {
    cat <<'EOF'
Usage:
  collect_forecast_data.sh --dry-run
  collect_forecast_data.sh --dry-run --single WORKLOAD REPEAT
  collect_forecast_data.sh --single WORKLOAD REPEAT
  collect_forecast_data.sh --all
  collect_forecast_data.sh --all --skip-existing

Examples:
  collect_forecast_data.sh --dry-run
  collect_forecast_data.sh --dry-run --single stable 1
  collect_forecast_data.sh --single stable 1
  collect_forecast_data.sh --all
EOF
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;

        --single)
            [[ $# -ge 3 ]] || {
                echo "--single requires WORKLOAD REPEAT" >&2
                exit 2
            }

            MODE="single"
            WORKLOAD="$2"
            REPEAT="$3"
            shift 3
            ;;

        --all)
            MODE="all"
            shift
            ;;

        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

# --dry-run alone means show the entire official plan.
if [[ -z "$MODE" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        MODE="all"
    else
        usage
        exit 2
    fi
fi

# ----------------------------------------------------------
# Load official experiment contract from YAML.
# ----------------------------------------------------------

eval "$(
python3 - "$CONFIG" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)

experiment = config["experiment"]
validation = config["validation"]
freeze = config.get("environment_freeze", {})

duration_seconds = float(config["duration_seconds"])
sample_interval_seconds = float(config["sampling_interval_seconds"])

# Official plan size: workloads x repeats; one seed per repeat.
expected_runs = len(config["workloads"]) * int(config["repeats"])
if len(config["seeds"]) != int(config["repeats"]):
    sys.exit("config error: seeds must have exactly one entry per repeat")

# nominal samples = runs x (duration / interval) x controllers  (= 9000)
nominal_samples = (
    expected_runs
    * round(duration_seconds / sample_interval_seconds)
    * len(config["controllers"])
)

plan_contract = config.get("official_plan", {})
if plan_contract.get("expected_runs", expected_runs) != expected_runs:
    sys.exit("config error: official_plan.expected_runs != workloads x repeats")
if plan_contract.get("nominal_samples", nominal_samples) != nominal_samples:
    sys.exit("config error: official_plan.nominal_samples mismatch")

values = {
    "EXPERIMENT_TYPE": experiment["type"],
    "TOPOLOGY_NAME": experiment["topology"],
    "OUTPUT_ROOT": experiment["output_root"],
    "DURATION": duration_seconds,
    "SAMPLE_INTERVAL": config["sampling_interval_seconds"],
    "MIN_COVERAGE": validation["min_sample_coverage_ratio"],
    "MAX_GAP": validation["max_gap_seconds"],
    "MAX_SEND_ERROR": validation["max_send_error_ratio"],
    "EXPECTED_RUNS": expected_runs,
    "NOMINAL_SAMPLES": int(nominal_samples),
    "REQUIRE_CLEAN_GIT": int(freeze.get("require_clean_git", True)),
    "REQUIRE_GIT_TAG": int(freeze.get("require_git_tag", True)),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

# ----------------------------------------------------------
# Preflight static artifacts.
# ----------------------------------------------------------

for path in \
    "$CONFIG" \
    "$CAPACITY_RUNS" \
    "$CONTROLLER_CAPACITY" \
    "configs/controllers_2c20s.yaml"
do
    if [[ ! -f "$path" ]]; then
        echo "Missing required artifact: $path" >&2
        exit 1
    fi
done

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_TAG="$(git tag --points-at HEAD | head -n 1)"

# ----------------------------------------------------------
# Environment freeze.
#
# All 25 official runs must come from one frozen environment (same VM,
# CPU/RAM, Mininet/OVS/Ryu, code, topology, capacity artifact, commit/tag).
# A fingerprint is stored before the first real run; any later change aborts.
# --dry-run only reports problems, real runs refuse to start.
# ----------------------------------------------------------

FREEZE_FILE="${OUTPUT_ROOT}/forecast_dataset_environment.json"
RUNTIME_SOURCE="src/experiments/topologies/runtime_2c20s.py"

RUNTIME_SUPPORTS_GIT_TAG=0
if grep -q -- '--git-tag' "$RUNTIME_SOURCE"; then
    RUNTIME_SUPPORTS_GIT_TAG=1
fi

freeze_problem() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[FREEZE WARNING] $1" >&2
    else
        echo "[FREEZE ERROR] $1" >&2
        exit 1
    fi
}

check_git_freeze() {
    # Tracked changes only: collected data under data/ is untracked output.
    if [[ "$REQUIRE_CLEAN_GIT" -eq 1 ]] &&
       [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
        freeze_problem "git working tree is not clean"
    fi

    if [[ "$REQUIRE_GIT_TAG" -eq 1 && -z "$GIT_TAG" ]]; then
        freeze_problem "HEAD has no git tag (e.g. forecast-dataset-v1)"
    fi

    if [[ "$RUNTIME_SUPPORTS_GIT_TAG" -eq 0 ]]; then
        echo "[FREEZE WARNING] $RUNTIME_SOURCE has no --git-tag option;" \
             "metadata.json will lack git_tag" >&2
    fi
}

environment_fingerprint() {
    GIT_COMMIT="$GIT_COMMIT" GIT_TAG="$GIT_TAG" CONFIG="$CONFIG" \
    CAPACITY_ARTIFACT="$CONTROLLER_CAPACITY" \
    python3 - <<'PY'
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def sha256(path: str) -> str:
    file = Path(path)
    if not file.exists():
        return "missing"
    return hashlib.sha256(file.read_bytes()).hexdigest()


def first_line(command: list[str]) -> str:
    # Version banners go to stdout or stderr depending on the tool.
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    for line in (done.stdout + done.stderr).splitlines():
        if line.strip():
            return line.strip()
    return "unavailable"


def mem_total_kb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return 0


fingerprint = {
    "git_commit": os.environ["GIT_COMMIT"],
    "git_tag": os.environ["GIT_TAG"],
    "hostname": platform.node(),
    "kernel": platform.release(),
    "python_version": platform.python_version(),
    "cpu_count": os.cpu_count(),
    "mem_total_kb": mem_total_kb(),
    "mininet_version": first_line(["mn", "--version"]),
    "ovs_version": first_line(["ovs-vsctl", "--version"]),
    "ryu_version": first_line(["ryu-manager", "--version"]),
    "capacity_artifact_sha256": sha256(os.environ["CAPACITY_ARTIFACT"]),
    "controllers_config_sha256": sha256("configs/controllers_2c20s.yaml"),
    "topology_sha256": sha256(
        "src/experiments/topologies/capacity_2c20s.py"
    )
    + "+"
    + sha256("src/experiments/topologies/runtime_2c20s.py"),
    "experiment_config_sha256": sha256(os.environ["CONFIG"]),
}
print(json.dumps(fingerprint, sort_keys=True))
PY
}

freeze_environment() {
    local verbosity="${1:-verbose}"
    local current

    check_git_freeze
    current="$(environment_fingerprint)"

    if [[ "$verbosity" == "verbose" ]]; then
        echo "ENVIRONMENT_FINGERPRINT=$current"
    fi

    if [[ ! -f "$FREEZE_FILE" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[FREEZE] not stored yet; the first real run will store it"
        else
            mkdir -p "$OUTPUT_ROOT"
            printf '%s\n' "$current" > "$FREEZE_FILE"
            echo "[FREEZE] stored fingerprint: $FREEZE_FILE"
        fi
        return 0
    fi

    python3 - "$FREEZE_FILE" "$current" <<'PY' || freeze_problem "environment differs from the frozen fingerprint"
import json
import sys
from pathlib import Path

frozen = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
current = json.loads(sys.argv[2])
changed = sorted(
    key
    for key in frozen.keys() | current.keys()
    if frozen.get(key) != current.get(key)
)
for key in changed:
    print(f"  {key}: frozen={frozen.get(key)!r} current={current.get(key)!r}",
          file=sys.stderr)
sys.exit(1 if changed else 0)
PY
}

# ----------------------------------------------------------
# Official deterministic plan.
# ----------------------------------------------------------

mapfile -t PLAN < <(
python3 - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

from scripts.experiments.forecast_collection_plan import (
    build_plan,
)

config = yaml.safe_load(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)

for run in build_plan(config):
    print(
        f"{run.workload}|"
        f"{run.repeat_index}|"
        f"{run.seed}|"
        f"{run.run_id}"
    )
PY
)

if [[ "$MODE" == "all" && "${#PLAN[@]}" -ne "$EXPECTED_RUNS" ]]; then
    echo "Official plan has ${#PLAN[@]} runs, expected $EXPECTED_RUNS" >&2
    exit 1
fi

check_runtime_contract() {
    local source="$ROOT/src/experiments/topologies/runtime_2c20s.py"

    [[ -f "$source" ]] || {
        echo "Missing runtime entrypoint: $source" >&2
        return 1
    }

    local required=(
        "--run-id"
        "--repeat-index"
        "--seed"
        "--experiment-type"
        "--topology-name"
        "--capacity-artifact"
        "--git-commit"
        "--orchestrator-url"
    )

    local flag
    for flag in "${required[@]}"; do
        if ! grep -q -- "$flag" "$source"; then
            echo "Runtime contract missing $flag in $source" >&2
            return 1
        fi
    done

    if ! grep -q -- "--git-tag" "$source"; then
        echo "Runtime contract missing --git-tag in $source" >&2
        return 1
    fi
}

run_one() {
    local workload="$1"
    local repeat="$2"
    local seed="$3"
    local run_id="$4"

    local run_dir="${OUTPUT_ROOT}/${run_id}"

    if [[ -e "$run_dir" ]]; then
        if [[ "$SKIP_EXISTING" -eq 1 ]]; then
            echo "[SKIP] existing run: $run_id"
            return 0
        fi

        echo "ERROR: official run directory already exists:" >&2
        echo "  $run_dir" >&2
        echo "Refusing to overwrite or append official data." >&2
        exit 1
    fi

    local workload_flag

    case "$workload" in
        stable)
            workload_flag="--forecast-stable"
            ;;
        gradual)
            workload_flag="--forecast-gradual"
            ;;
        burst)
            workload_flag="--forecast-burst"
            ;;
        oscillating)
            workload_flag="--forecast-oscillating"
            ;;
        hot-switch)
            workload_flag="--forecast-hot-switch"
            ;;
        *)
            echo "Unsupported workload: $workload" >&2
            exit 1
            ;;
    esac

    local -a cmd=(
        sudo -E env
        "PYTHONPATH=$ROOT"
        "$(which python3)"
        src/experiments/topologies/runtime_2c20s.py

        "$workload_flag"

        --capacity-runs
        "$CAPACITY_RUNS"

        --controller-capacity
        "$CONTROLLER_CAPACITY"

        --output-dir
        "$OUTPUT_ROOT"

        --duration
        "$DURATION"

        --sample-interval
        "$SAMPLE_INTERVAL"

        --run-id
        "$run_id"

        --experiment-type
        "$EXPERIMENT_TYPE"

        --topology-name
        "$TOPOLOGY_NAME"

        --repeat-index
        "$repeat"

        --seed
        "$seed"

        --capacity-artifact
        "$CONTROLLER_CAPACITY"

        --git-commit
        "$GIT_COMMIT"

        --orchestrator-url
        "$ORCHESTRATOR_URL"
    )

    # Recorded in metadata.json (git_tag) once the runtime accepts the option.
    if [[ "$RUNTIME_SUPPORTS_GIT_TAG" -eq 1 ]]; then
        cmd+=(--git-tag "$GIT_TAG")
    fi

    echo
    echo "============================================================"
    echo "RUN_ID=$run_id"
    echo "WORKLOAD=$workload"
    echo "REPEAT=$repeat"
    echo "SEED=$seed"
    echo "OUTPUT=$run_dir"
    echo "CAPACITY_RUNS=$CAPACITY_RUNS"
    echo "CONTROLLER_CAPACITY=$CONTROLLER_CAPACITY"
    echo "GIT_COMMIT=$GIT_COMMIT"
    echo "============================================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'COMMAND='
        printf ' %q' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    # Refuse real collection until the runtime can persist the official metadata contract.
    if [[ "$DRY_RUN" -eq 0 ]]; then
        check_runtime_contract
    fi

    # The environment must still be the frozen one before every run.
    freeze_environment quiet

    "${cmd[@]}"

    # Validate immediately: a bad official run must stop the whole collection.
    if ! python3 scripts/experiments/validate_forecast_collection.py \
        "$run_dir" --config "$CONFIG"; then
        echo "ERROR: $run_id failed collection validation" >&2
        exit 1
    fi
}

freeze_environment verbose

matched=0

for row in "${PLAN[@]}"; do
    IFS='|' read -r workload repeat seed run_id <<< "$row"

    if [[ "$MODE" == "single" ]]; then
        if [[ "$workload" != "$WORKLOAD" ||
              "$repeat" != "$REPEAT" ]]; then
            continue
        fi
    fi

    matched=$((matched + 1))
    run_one "$workload" "$repeat" "$seed" "$run_id"
done

if [[ "$matched" -eq 0 ]]; then
    echo "No official run matched request." >&2
    exit 1
fi

echo
echo "SELECTED_RUNS=$matched"

if [[ "$MODE" == "all" ]]; then
    echo "OFFICIAL_PLAN: ${matched} runs, nominal samples=${NOMINAL_SAMPLES}"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        python3 scripts/experiments/validate_forecast_collection.py \
            --official --config "$CONFIG"
    fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY_RUN: PASS"
fi
