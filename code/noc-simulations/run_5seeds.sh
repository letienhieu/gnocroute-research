#!/bin/bash
# ============================================================
# BookSim2 — Reproducible NoC Routing Experiments (5 seeds)
# Scientific reproducibility: fixed seeds, tracked outputs
# ============================================================
# Experiment design:
#   - Topology: mesh88, mesh44
#   - Routing: dor, adaptive_xy_yx, min_adapt
#   - Traffic: uniform, hotspot
#   - IR: 0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45
#   - Seeds: 5 per config (42, 123, 456, 789, 999)
#   - Total: 2 × 3 × 2 × 8 × 5 = 480 runs
# ============================================================

BOOKSIM="./src/booksim"
OUTDIR="./results_v3"
mkdir -p $OUTDIR

SUMMARY="$OUTDIR/summary_5seeds.csv"
echo "timestamp,topology,algorithm,traffic,inj_rate,seed,avg_latency,min_latency,max_latency,status" > $SUMMARY

# Experiment configs
TOPOLOGIES="mesh44 mesh88"
ALGORITHMS="dor adaptive_xy_yx min_adapt"
TRAFFICS="uniform hotspot"
INJ_RATES="0.01 0.05 0.1 0.2 0.3 0.35 0.4 0.45"
SEEDS="42 123 456 789 999"

# Print header
echo "========================================================================="
echo "GNNocRoute — Reproducible BookSim2 Experiments (5 seeds)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "BookSim: $(git describe --always 2>/dev/null || echo 'cloned')"
echo "Configs: $(echo $TOPOLOGIES | wc -w) topologies × $(echo $ALGORITHMS | wc -w) algorithms"
echo "         × $(echo $TRAFFICS | wc -w) traffics × $(echo $INJ_RATES | wc -w) IR × $(echo $SEEDS | wc -w) seeds"
echo "         = $(( $(echo $TOPOLOGIES | wc -w) * $(echo $ALGORITHMS | wc -w) * $(echo $TRAFFICS | wc -w) * $(echo $INJ_RATES | wc -w) * $(echo $SEEDS | wc -w) )) runs"
echo "========================================================================="
echo ""

TOTAL=0
PASSED=0

for topo in $TOPOLOGIES; do
    case $topo in
        mesh44) K=4; N=2; TOPO_STR="mesh" ;;
        mesh88) K=8; N=2; TOPO_STR="mesh" ;;
    esac
    
    for algo in $ALGORITHMS; do
        for traffic in $TRAFFICS; do
            for inj in $INJ_RATES; do
                for seed in $SEEDS; do
                    
                    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
                    CFG="/tmp/noc_${topo}_${algo}_${traffic}_${inj}_s${seed}.cfg"
                    
                    cat > $CFG << CFGEOF
topology = $TOPO_STR;
k = $K;
n = $N;
routing_function = $algo;
num_vcs = 4;
vc_buf_size = 8;
wait_for_tail_credit = 1;
priority = local_age;
sim_type = latency;
warmup_periods = 5;
sample_period = 2000;
sim_count = 1;
print_csv_results = 1;
traffic = $traffic;
injection_rate = $inj;
packet_size = 1;
seed = $seed;
CFGEOF
                    
                    OUTFILE="${OUTDIR}/${topo}_${algo}_${traffic}_${inj}_s${seed}.txt"
                    
                    # Run
                    $BOOKSIM $CFG 2>&1 > $OUTFILE
                    EXIT_CODE=$?
                    
                    # Extract results
                    LATENCY=$(grep "Packet latency average" $OUTFILE | head -1 | grep -oP '= \K[0-9.]+')
                    MIN_LAT=$(grep "Packet latency average" $OUTFILE | head -1 | grep -oP 'minimum = \K[0-9]+' || echo "")
                    MAX_LAT=$(grep "Packet latency average" $OUTFILE | head -1 | grep -oP 'maximum = \K[0-9]+' || echo "")
                    
                    if [ -z "$LATENCY" ]; then
                        STATUS="FAILED"
                    else
                        STATUS="OK"
                        PASSED=$((PASSED + 1))
                    fi
                    
                    TOTAL=$((TOTAL + 1))
                    
                    # Progress
                    printf "  [%3d/%3d] %s %-12s %-8s inj=%s seed=%s → %s\n" \
                        $TOTAL $(( $(echo $TOPOLOGIES | wc -w) * $(echo $ALGORITHMS | wc -w) * $(echo $TRAFFICS | wc -w) * $(echo $INJ_RATES | wc -w) * $(echo $SEEDS | wc -w) )) \
                        "$topo" "$algo" "$traffic" "$inj" "$seed" "${LATENCY:-FAILED}"
                    
                    echo "$TIMESTAMP,$topo,$algo,$traffic,$inj,$seed,${LATENCY:-NA},${MIN_LAT:-NA},${MAX_LAT:-NA},$STATUS" >> $SUMMARY
                    
                    rm -f $CFG
                done
            done
        done
    done
done

echo ""
echo "========================================================================="
echo "COMPLETE: $TOTAL runs, $PASSED passed"
echo "Output: $OUTDIR"
echo "Summary: $SUMMARY"
echo "Date: $(date)"

# Generate aggregate stats
python3 << 'PYEOF'
import csv
import numpy as np
from collections import defaultdict

agg = defaultdict(lambda: {"lats": [], "mins": [], "maxes": []})

with open("'"$SUMMARY"'") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["topology"], row["algorithm"], row["traffic"], row["inj_rate"])
        if row["avg_latency"] != "NA":
            agg[key]["lats"].append(float(row["avg_latency"]))
            if row["min_latency"] != "NA":
                agg[key]["mins"].append(float(row["min_latency"]))
            if row["max_latency"] != "NA":
                agg[key]["maxes"].append(float(row["max_latency"]))

print("\n=== AGGREGATE RESULTS (mean ± std across 5 seeds) ===")
print(f"{'Topo':8s} {'Routing':16s} {'Traffic':10s} {'IR':5s} {'Latency':12s} {'CI':10s} {'Min':8s} {'Max':8s}")
print("-" * 80)

for key in sorted(agg.keys()):
    d = agg[key]
    mean = np.mean(d["lats"])
    std = np.std(d["lats"])
    ci95 = 1.96 * std / np.sqrt(len(d["lats"]))
    mean_min = np.mean(d["mins"]) if d["mins"] else 0
    mean_max = np.mean(d["maxes"]) if d["maxes"] else 0
    print(f"{key[0]:8s} {key[1]:16s} {key[2]:10s} {key[3]:5s} {mean:8.1f}±{std:5.1f} ±{ci95:4.1f} {mean_min:6.0f} {mean_max:6.0f}")

# Save aggregate
agg_file = "'"${OUTDIR%/}"'"'/aggregate_results.csv'
with open(agg_file, 'w') as f:
    f.write("topology,algorithm,traffic,inj_rate,mean_latency,std_latency,ci95,mean_min,mean_max,n_seeds\n")
    for key in sorted(agg.keys()):
        d = agg[key]
        mean = np.mean(d["lats"])
        std = np.std(d["lats"])
        ci95 = 1.96 * std / np.sqrt(max(len(d["lats"]), 1))
        mean_min = np.mean(d["mins"]) if d["mins"] else 0
        mean_max = np.mean(d["maxes"]) if d["maxes"] else 0
        f.write(f"{key[0]},{key[1]},{key[2]},{key[3]},{mean:.2f},{std:.2f},{ci95:.2f},{mean_min:.0f},{mean_max:.0f},{len(d['lats'])}\n")

print(f"\nAggregate saved: {agg_file}")
PYEOF