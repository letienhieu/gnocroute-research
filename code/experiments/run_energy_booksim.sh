#!/bin/bash
# ============================================================
# BookSim2: Energy + PARSEC-like experiments
# ============================================================
BOOKSIM="/home/opc/.openclaw/workspace/booksim2/src/booksim"
TECHFILE="/home/opc/.openclaw/workspace/booksim2/src/power/techfile.txt"
OUTDIR="/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/experiments/results_v2"
mkdir -p $OUTDIR

SUMMARY="$OUTDIR/summary_energy.csv"
echo "topology,algorithm,traffic,inj_rate,latency,energy_pJ,status" > $SUMMARY

echo "=== ENERGY + PARSEC EXPERIMENTS ==="

# Check if techfile exists
if [ ! -f "$TECHFILE" ]; then
    echo "Warning: No techfile found. Energy results will use default."
fi

TOPOLOGIES="mesh44 mesh88"
ALGORITHMS="dor adaptive_xy_yx min_adapt"
TRAFFICS="uniform transpose hotspot"
INJ_RATES="0.01 0.02 0.05 0.1"

CONFIG_COUNT=0
for topo in $TOPOLOGIES; do
    case $topo in
        mesh44) K=4; T="mesh" ;;
        mesh88) K=8; T="mesh" ;;
    esac
    
    for algo in $ALGORITHMS; do
        for traffic in $TRAFFICS; do
            for inj in $INJ_RATES; do
                CONFIG_COUNT=$((CONFIG_COUNT + 1))
                
                CFG="/tmp/noc_energy_$$.cfg"
                cat > $CFG << CFGEOF
topology = $T;
k = $K;
n = 2;
routing_function = $algo;
num_vcs = 4;
vc_buf_size = 8;
wait_for_tail_credit = 1;
priority = local_age;
sim_type = latency;
warmup_periods = 3;
sample_period = 1000;
sim_count = 1;
print_csv_results = 1;
traffic = $traffic;
injection_rate = $inj;
packet_size = 1;
techfile = $TECHFILE;
power_output_file = $OUTDIR/power_${topo}_${algo}_${traffic}_${inj}.txt;
sim_power = 1;
trace_mode = 0;
CFGEOF
                
                printf "  [%3d] %-20s %-12s inj=%.2f ... " "$CONFIG_COUNT" "${topo}_${algo}" "${traffic}" $inj
                
                OUTPUT=$( $BOOKSIM $CFG 2>&1 )
                
                LATENCY=$(echo "$OUTPUT" | grep "Packet latency average" | head -1 | grep -oP '= \K[0-9.]+')
                ENERGY=$(echo "$OUTPUT" | grep -i "energy\|power" | head -1 | grep -oP '= \K[0-9.]+')
                
                if [ -n "$LATENCY" ]; then
                    echo "$topo,$algo,$traffic,$inj,$LATENCY,${ENERGY:-NA},OK" >> $SUMMARY
                    printf "lat=%-8s energy=%-8s ✅\n" "$LATENCY" "${ENERGY:-N/A}"
                else
                    echo "$topo,$algo,$traffic,$inj,NA,NA,SATURATED" >> $SUMMARY
                    printf "SATURATED ❌\n"
                fi
                
                rm -f $CFG
            done
        done
    done
done

echo ""
echo "=== DONE: $CONFIG_COUNT configs ==="
echo "Results: $SUMMARY"
head -10 $SUMMARY
echo "..."
tail -5 $SUMMARY
