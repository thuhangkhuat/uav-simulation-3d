#!/usr/bin/env bash
# Scenario-variant experiments for `ours` (hold_latest + speed guard, RotorPy).
#
#   bash scripts/run_scen_variants.sh check   # tests + 2 s smoke run of every variant
#   bash scripts/run_scen_variants.sh r0      # r = 0, all variants in parallel (deterministic)
#   bash scripts/run_scen_variants.sh r1      # r = 1, REPS repeats each, one at a time
#
# Five-method benchmark on one scenario (default SCEN=scen1_3d_m3), same settings:
#   bash scripts/run_scen_variants.sh bench-check  # 2 s smoke run of all five methods
#   bash scripts/run_scen_variants.sh bench-r0     # r = 0, five methods in parallel
#   bash scripts/run_scen_variants.sh bench-r1     # r = 1, BENCH_REPS benchmarks, one at a time
#
# Run r0/r1 in the background so they survive an SSH disconnect, e.g.
#   nohup bash scripts/run_scen_variants.sh r1 > logs/scen-variants-r1.log 2>&1 &
# Existing run directories are skipped, so an interrupted r1 batch can be resumed.
# Environment overrides: VARIANTS="scen1_3d_m2 ..." REPS=3 CONTROLLER=ours
#                        SCEN=scen1_3d_m3 BENCH_REPS=5 CONTROLLERS="ours apf ..."
set -u
cd "$(dirname "$0")/.."

VARIANTS=${VARIANTS:-"scen1_3d_m2 scen1_3d_m3 scen1_3d_m2_tc9"}
REPS=${REPS:-3}
CONTROLLER=${CONTROLLER:-ours}
SCEN=${SCEN:-scen1_3d_m3}
BENCH_REPS=${BENCH_REPS:-5}
CONTROLLERS=${CONTROLLERS:-"ours apf pure_mpc cbf_qp_lag impc_ob_rec"}
# Identical for every method: upstream commands held until the next one, attitude
# guard (default on) and speed guard at the plant, RotorPy.
SETTINGS=(--speed-guard on --command-execution hold_latest)
COMMON=(--controller "$CONTROLLER" "${SETTINGS[@]}")
export PYTHONWARNINGS=ignore::SyntaxWarning
mkdir -p logs runs

run_one() {   # run_one <variant> <ratio> <out>
    local variant=$1 ratio=$2 out=$3
    if [ -e "$out" ]; then echo "skip $out (exists)"; return 0; fi
    echo "$(date '+%F %T') start $out  [load: $(cut -d' ' -f1-3 /proc/loadavg)]"
    uv run uav3d run --scenario "scenarios/$variant.yaml" "${COMMON[@]}" \
        --compute-ratio "$ratio" --out "$out" > "logs/$(basename "$out").log" 2>&1
    echo "$(date '+%F %T') done  $out: $(tail -n 1 "logs/$(basename "$out").log")"
}

case "${1:-}" in
  check)
    uv run pytest -q || exit 1
    for v in $VARIANTS; do
        out=runs/check-$v
        rm -rf "$out"
        uv run uav3d run --scenario "scenarios/$v.yaml" "${COMMON[@]}" --duration 2 --out "$out" || exit 1
        test -s "$out/scenario.yaml" || { echo "missing $out/scenario.yaml"; exit 1; }
        grep -E '"scenario": |"scenario_sha256"' "$out/config.json"
    done ;;
  r0)
    for v in $VARIANTS; do run_one "$v" 0 "runs/$CONTROLLER-$v-r0" & done
    wait ;;
  r1)
    # Measured delays become mission-time delays: keep these runs alone on the CPU.
    for v in $VARIANTS; do for k in $(seq 1 "$REPS"); do
        run_one "$v" 1 "runs/$CONTROLLER-$v-r1-rep$k"
    done; done ;;
  bench-check)
    out=runs/check-bench-$SCEN
    rm -rf "$out"
    # shellcheck disable=SC2086
    uv run uav3d benchmark --scenario "scenarios/$SCEN.yaml" "${SETTINGS[@]}" \
        --controllers $CONTROLLERS --duration 2 --out "$out" || exit 1
    cat "$out/benchmark.csv" ;;
  bench-r0)
    mkdir -p "runs/bench-$SCEN-r0"
    for c in $CONTROLLERS; do
        out=runs/bench-$SCEN-r0/$c
        if [ -e "$out" ]; then echo "skip $out (exists)"; continue; fi
        ( echo "$(date '+%F %T') start $out"
          uv run uav3d run --scenario "scenarios/$SCEN.yaml" --controller "$c" "${SETTINGS[@]}" \
              --compute-ratio 0 --out "$out" > "logs/bench-$SCEN-r0-$c.log" 2>&1
          echo "$(date '+%F %T') done  $out: $(tail -n 1 "logs/bench-$SCEN-r0-$c.log")" ) &
    done
    wait ;;
  bench-r1)
    # One benchmark runs the five methods one after another; keep it alone on the CPU.
    for k in $(seq 1 "$BENCH_REPS"); do
        out=runs/bench-$SCEN-r1-rep$k
        if [ -e "$out" ]; then echo "skip $out (exists)"; continue; fi
        echo "$(date '+%F %T') start $out  [load: $(cut -d' ' -f1-3 /proc/loadavg)]"
        # shellcheck disable=SC2086
        uv run uav3d benchmark --scenario "scenarios/$SCEN.yaml" "${SETTINGS[@]}" \
            --controllers $CONTROLLERS --out "$out" > "logs/bench-$SCEN-r1-rep$k.log" 2>&1
        echo "$(date '+%F %T') done  $out (exit $?)"
    done ;;
  *)
    sed -n '2,17p' "$0"; exit 2 ;;
esac
