#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

CONFIG="configs/experiments/capacity_fine_2c20s.yaml"

RATES=(
    30
    40
    50
    60
    70
    80
    90
    100
    110
    120
    140
    160
    180
    200
    220
    240
)

CONTROLLERS=(
    c1
    c2
)

REPEATS=(
    1
    2
    3
    4
    5
)

PYTHON_BIN="$(command -v python3)"

for controller in "${CONTROLLERS[@]}"; do
    for rate in "${RATES[@]}"; do
        for repeat in "${REPEATS[@]}"; do

            printf -v rep "%02d" "$repeat"

            run_id="capacity-2c20s-fine-${controller}-${rate}-r${rep}"
            run_dir="data/experiment_runs/${run_id}"

            if [[ -f "${run_dir}/validation.json" ]]; then
                if jq -e '.valid == true' \
                    "${run_dir}/validation.json" \
                    >/dev/null
                then
                    echo "SKIP VALID: ${run_id}"
                    continue
                fi
            fi

            if [[ -d "$run_dir" ]]; then
                echo "ERROR: incomplete/existing run:"
                echo "$run_dir"
                exit 1
            fi

            echo
            echo "========================================"
            echo "controller=${controller}"
            echo "rate=${rate}"
            echo "repeat=${repeat}"
            echo "========================================"

            sudo -E env \
                PYTHONPATH="$PWD" \
                "$PYTHON_BIN" \
                -m src.experiments.topologies.capacity_smoke_2c20s \
                --config "$CONFIG" \
                --controller "$controller" \
                --rate "$rate" \
                --repeat "$repeat"
        done
    done
done

