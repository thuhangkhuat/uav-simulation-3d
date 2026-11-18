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
#   bash scripts/run_scen_variants.sh bench-seeds  # r = 1, one benchmark per seed with the 2D
#        thesis starts: --seed k --start-jitter 20 --jitter-key $JITTER_KEY (seed 7: no jitter)
#   Nine UAVs (scenarios/scen3_3d_n9.yaml, from scripts/make_scen3_3d_n9.py):
#        SCEN=scen3_3d_n9 JITTER_KEY=scen3 SCHED=async bash scripts/run_scen_variants.sh bench-seeds
#
# Run r0/r1 in the background so they survive an SSH disconnect, e.g.
#   nohup bash scripts/run_scen_variants.sh r1 > logs/scen-variants-r1.log 2>&1 &
# Existing run directories are skipped, so an interrupted r1 batch can be resumed.
# Environment overrides: VARIANTS="scen1_3d_m2 ..." REPS=3 CONTROLLER=ours
#                        SCEN=scen1_3d_m3 BENCH_REPS=5 CONTROLLERS="ours apf ..."
#                        SEEDS="1 2 ... 10" JITTER=20 NOMINAL_SEEDS="7" RATIO=1 JITTER_KEY=scen1
#                        SCHED=parallel|parallel_own|sequential|async (scheduling for every mode)
set -u
cd "$(dirname "$0")/.."

VARIANTS=${VARIANTS:-"scen1_3d_m2 scen1_3d_m3 scen1_3d_m2_tc9"}
REPS=${REPS:-3}
CONTROLLER=${CONTROLLER:-ours}
SCEN=${SCEN:-scen1_3d_m3}
BENCH_REPS=${BENCH_REPS:-5}
CONTROLLERS=${CONTROLLERS:-"ours apf pure_mpc cbf_qp_lag impc_ob_rec"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
JITTER=${JITTER:-20}
NOMINAL_SEEDS=${NOMINAL_SEEDS:-7}       # thesis_10seeds: seed 7 keeps the nominal starts
JITTER_KEY=${JITTER_KEY:-scen1}         # upstream scenario name seeding the jitter (scen3 for scen3_3d_n9)
RATIO=${RATIO:-1}
# Identical for every method: upstream commands held until the next one, attitude
# guard (default on) and speed guard at the plant, RotorPy.
SCHED=${SCHED:-parallel}               # parallel | parallel_own | sequential | async
SETTINGS=(--speed-guard on --command-execution hold_latest --scheduling "$SCHED")
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
  bench-seeds)
    # One benchmark per seed, the same starts as upstream's 2D thesis runs of scen1.
    for k in $SEEDS; do
        tag=$([ "$SCHED" = parallel ] && echo "" || echo "-$SCHED")
        out=runs/seeds-$SCEN-r$RATIO$tag/s$k
        if [ -e "$out" ]; then echo "skip $out (exists)"; continue; fi
        jitter=$JITTER
        for n in $NOMINAL_SEEDS; do [ "$k" = "$n" ] && jitter=0; done
        mkdir -p "runs/seeds-$SCEN-r$RATIO$tag"
        echo "$(date '+%F %T') start $out (jitter $jitter)  [load: $(cut -d' ' -f1-3 /proc/loadavg)]"
        jarg=(); [ "$jitter" != 0 ] && jarg=(--start-jitter "$jitter" --jitter-key "$JITTER_KEY")
        # shellcheck disable=SC2086
        uv run uav3d benchmark --scenario "scenarios/$SCEN.yaml" "${SETTINGS[@]}" --seed "$k" \
            "${jarg[@]}" --compute-ratio "$RATIO" --controllers $CONTROLLERS --out "$out" \
            > "logs/seeds-$SCEN-r$RATIO$tag-s$k.log" 2>&1
        echo "$(date '+%F %T') done  $out (exit $?)"
    done ;;
  *)
    sed -n '2,23p' "$0"; exit 2 ;;
esac
