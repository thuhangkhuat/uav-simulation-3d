#!/usr/bin/env bash
# Scenario-variant experiments for `ours` (hold_latest + speed guard, RotorPy).
#
#   bash scripts/run_scen_variants.sh check   # tests + 2 s smoke run of every variant
#   bash scripts/run_scen_variants.sh r0      # r = 0, all variants in parallel (deterministic)
#   bash scripts/run_scen_variants.sh r1      # r = 1, REPS repeats each, one at a time
#
# Run r0/r1 in the background so they survive an SSH disconnect, e.g.
#   nohup bash scripts/run_scen_variants.sh r1 > logs/scen-variants-r1.log 2>&1 &
# Existing run directories are skipped, so an interrupted r1 batch can be resumed.
# Environment overrides: VARIANTS="scen1_3d_m2 ..." REPS=3 CONTROLLER=ours
set -u
cd "$(dirname "$0")/.."

VARIANTS=${VARIANTS:-"scen1_3d_m2 scen1_3d_m3 scen1_3d_m2_tc9"}
REPS=${REPS:-3}
CONTROLLER=${CONTROLLER:-ours}
COMMON=(--controller "$CONTROLLER" --speed-guard on --command-execution hold_latest)
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
  *)
    sed -n '2,10p' "$0"; exit 2 ;;
esac
