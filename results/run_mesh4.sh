#!/bin/bash
# Batch runner for Mesh 4x4 experiments
# Runs BookSim2 for each config, saves raw output, then aggregates to CSV

BOOKSIM="/home/opc/.openclaw/workspace/booksim2/src/booksim"
RESULTS="/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results"
mkdir -p "$RESULTS"

# Parameters
TOPOS=("mesh,4,4")
TRAFFICS=("uniform" "transpose" "hotspot")
RATES=(0.01 0.02 0.05 0.1 0.2 0.3)
ALGOS=("dor" "adaptive_xy_yx" "min_adapt" "valiant" "gnn_ppo_route_4x4")
SEEDS=(0 1 2 3 4)

TOTAL=0
DONE=0
FAIL=0

for topo_spec in "${TOPOS[@]}"; do
    IFS=',' read -r topo k n <<< "$topo_spec"
    for traffic in "${TRAFFICS[@]}"; do
        for rate in "${RATES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for algo in "${ALGOS[@]}"; do
                    # Skip GNN for non-mesh or wrong size
                    if [[ "$algo" == gnn_ppo_route_* ]]; then
                        if [[ "$topo" != "mesh" ]]; then continue; fi
                        size=${algo#gnn_ppo_route_}
                        size=${size%x*}
                        if [[ "$size" != "$k" ]]; then continue; fi
                    fi
                    
                    TOTAL=$((TOTAL + 1))
                    fname="topo=${topo}_k=${k}_n=${n}_algo=${algo}_traffic=${traffic}_inj=${rate}_seed=${seed}"
                    outpath="$RESULTS/${fname}.txt"
                    
                    if [[ -f "$outpath" ]]; then
                        DONE=$((DONE + 1))
                        continue
                    fi
                    
                    # Create temp config
                    cfg=$(mktemp /tmp/bs2_XXXX.cfg)
                    cat > "$cfg" << CFGEOF
topology = ${topo};
k = ${k};
n = ${n};
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
injection_rate = ${rate};
warmup_periods = 3;
sample_period = 1000;
sim_count = 100;
seed = ${seed};
CFGEOF
                    
                    # Run BookSim2
                    $BOOKSIM "$cfg" > "$outpath" 2>&1
                    rc=$?
                    rm -f "$cfg"
                    
                    if [[ $rc -eq 0 ]] && grep -q "Packet latency average" "$outpath"; then
                        DONE=$((DONE + 1))
                    else
                        FAIL=$((FAIL + 1))
                        echo "FAIL: ${fname}"
                    fi
                    
                    # Progress every 50
                    if [[ $((TOTAL % 50)) -eq 0 ]]; then
                        echo "[$TOTAL] done=$DONE fail=$FAIL"
                    fi
                done
            done
        done
    done
done

# Create aggregated CSV
echo "topo,k,n,algo,traffic,inj_rate,seed,latency,accepted,hops" > "$RESULTS/all_results.csv"
for f in "$RESULTS"/topo=*.txt; do
    base=$(basename "$f" .txt)
    # Parse filename
    # topo=mesh_k=4_n=4_algo=dor_traffic=uniform_inj=0.01_seed=0
    topo=$(echo "$base" | sed 's/topo=//;s/_.*//')
    k=$(echo "$base" | sed 's/.*_k=//;s/_.*//')
    n=$(echo "$base" | sed 's/.*_n=//;s/_algo.*//')
    algo=$(echo "$base" | sed 's/.*_algo=//;s/_traffic.*//')
    traffic=$(echo "$base" | sed 's/.*_traffic=//;s/_inj.*//')
    rate=$(echo "$base" | sed 's/.*_inj=//;s/_seed.*//')
    seed=$(echo "$base" | sed 's/.*_seed=//')
    
    # Parse last occurrence from output
    latency=$(grep "Packet latency average" "$f" | tail -1 | grep -oP '[\d.]+' | head -1)
    accepted=$(grep "Accepted flit rate average" "$f" | tail -1 | grep -oP '[\d.]+' | head -1)
    hops=$(grep "Hops average" "$f" | tail -1 | grep -oP '[\d.]+' | head -1)
    
    if [[ -n "$latency" ]]; then
        echo "$topo,$k,$n,$algo,$traffic,$rate,$seed,$latency,$accepted,$hops" >> "$RESULTS/all_results.csv"
    fi
done

echo ""
echo "=== DONE ==="
echo "Total: $TOTAL, Done: $DONE, Failed: $FAIL"
echo "CSV: $RESULTS/all_results.csv"
wc -l "$RESULTS/all_results.csv"
