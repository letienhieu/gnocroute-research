#!/bin/bash
# Run verification experiments for GNN v4 (fixed vs buggy)
BOOKSIM="/home/opc/.openclaw/workspace/booksim2/src/booksim"
RESULTS="/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results_v4_verify"
mkdir -p "$RESULTS"

ALGOS=("gnn_port_score_route_4x4_v4" "dor" "planar_adapt")
TRAFFICS=("uniform" "hotspot")
FAILS=(0 7)
SEEDS=(0 1)

run_one() {
    local algo=$1 traffic=$2 fails=$3 seed=$4 label=$5
    local fname="verify_algo=${algo}_traffic=${traffic}_fails=${fails}_seed=${seed}_${label}"
    local outpath="$RESULTS/${fname}.txt"
    if [[ -f "$outpath" ]]; then
        echo "CACHED: ${fname}"
        return
    fi
    
    cfg=$(mktemp /tmp/bs2_${label}_XXXX.cfg)
    cat > "$cfg" << CFGEOF
topology = mesh;
k = 4;
n = 2;
routing_function = ${algo};
num_vcs = 4;
vc_buf_size = 4;
wait_for_tail_credit = 1;
vc_allocator = islip;
sw_allocator = islip;
alloc_iters = 1;
credit_delay = 2;
routing_delay = 1;
vc_alloc_delay = 1;
sw_alloc_delay = 1;
input_speedup = 2;
output_speedup = 1;
internal_speedup = 1.0;
traffic = ${traffic};
packet_size = 1;
sim_type = latency;
injection_rate = 0.05;
warmup_periods = 3;
sample_period = 1000;
sim_count = 100;
link_failures = ${fails};
fail_seed = ${seed};
seed = ${seed};
CFGEOF
    
    $BOOKSIM "$cfg" > "$outpath" 2>&1
    local rc=$?
    rm -f "$cfg"
    
    if [[ $rc -eq 0 ]] && grep -q "Packet latency average" "$outpath"; then
        echo "OK: ${fname}"
    else
        echo "FAIL: ${fname} (rc=$rc)"
    fi
}

echo "=== Running NEW (fixed) v4 experiments ==="
for algo in "${ALGOS[@]}"; do
    for traffic in "${TRAFFICS[@]}"; do
        for fails in "${FAILS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_one "$algo" "$traffic" "$fails" "$seed" "new"
            done
        done
    done
done

echo ""
echo "=== DONE ==="
echo "Results in: $RESULTS"
ls "$RESULTS"/*.txt 2>/dev/null | wc -l
