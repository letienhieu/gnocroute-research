#!/bin/bash
# ============================================================
# BookSim2 — Comprehensive NoC Routing Experiments
# So sanh 4 routing algorithms × 3 topologies × 3 traffic patterns
# ============================================================
BOOKSIM="./src/booksim"
OUTDIR="./results_v2"
mkdir -p $OUTDIR

SUMMARY="$OUTDIR/summary.csv"
echo "topology,algorithm,traffic,inj_rate,avg_latency,status" > $SUMMARY

echo "=== COMPREHENSIVE NOC EXPERIMENTS ==="
echo "Date: $(date)"
echo ""

# Configurations
TOPOLOGIES="mesh44 mesh88 torus44"
ALGORITHMS="dor adaptive_xy_yx min_adapt valiant"
TRAFFICS="uniform transpose hotspot"
INJ_RATES="0.01 0.02 0.05 0.1 0.15 0.2 0.3 0.4 0.5"

for topo in $TOPOLOGIES; do
    case $topo in
        mesh44) K=4; N=2; TOPO_STR="mesh" ;;
        mesh88) K=8; N=2; TOPO_STR="mesh" ;;
        torus44) K=4; N=2; TOPO_STR="torus" ;;
    esac
    
    for algo in $ALGORITHMS; do
        for traffic in $TRAFFICS; do
            for inj in $INJ_RATES; do
                
                CFG="/tmp/noc_exp_${topo}_${algo}_${traffic}_${inj}.cfg"
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
warmup_periods = 3;
sample_period = 1000;
sim_count = 1;
print_csv_results = 1;
traffic = $traffic;
injection_rate = $inj;
packet_size = 1;
CFGEOF
                
                OUT="${OUTDIR}/${topo}_${algo}_${traffic}_${inj}.txt"
                
                printf "  %-20s %-10s inj=%.2f ... " "${topo} ${algo}" "${traffic}" $inj
                
                $BOOKSIM $CFG 2>&1 > $OUT
                
                # Extract latency
                LATENCY=$(grep "Packet latency average" $OUT | head -1 | grep -oP '= \K[0-9.]+')
                if [ -z "$LATENCY" ]; then
                    # Maybe saturated
                    LATENCY=$(grep -oP 'exceeded \K[0-9]+' $OUT | head -1)
                    if [ -z "$LATENCY" ]; then
                        STATUS="FAILED"
                        echo "❌ $STATUS"
                    else
                        STATUS="SATURATED"
                        echo "⚠️  SATURATED (>$LATENCY)"
                    fi
                else
                    STATUS="OK"
                    echo "✅ ${LATENCY} cyc"
                fi
                
                echo "$topo,$algo,$traffic,$inj,$LATENCY,$STATUS" >> $SUMMARY
                rm -f $CFG
            done
        done
    done
done

echo ""
echo "=== DONE ==="
echo "Total configs tested: $(grep -c '.' $SUMMARY)"
echo "Summary: $SUMMARY"
ls -la $OUTDIR/*.txt | wc -l
echo "txt files"
