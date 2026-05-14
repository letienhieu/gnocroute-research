#!/bin/bash
# NoC Routing Simulation Experiments with BookSim2
# So sanh: XY routing vs adaptive routing
# Topology: Mesh 4x4, Mesh 8x8, Torus 4x4
# Traffic: uniform, transpose, hotspot
# Injection rate: 0.01 -> 0.5

BOOKSIM="./src/booksim"
OUTDIR="./results"
mkdir -p $OUTDIR

echo "=== NOC ROUTING SIMULATION EXPERIMENTS ==="
echo "Date: $(date)"
echo ""

for topo in "mesh88" "mesh44" "torus44"; do
    case $topo in
        mesh88) K=8; N=2; TOPO="mesh" ;;
        mesh44) K=4; N=2; TOPO="mesh" ;;
        torus44) K=4; N=2; TOPO="torus" ;;
    esac
    
    for routing in "dor" "adaptive_local"; do
        for traffic in "uniform" "transpose" "hotspot"; do
            for inj in 0.02 0.05 0.1 0.15 0.2 0.3; do
                
                # Create config
                CFG="/tmp/noc_${topo}_${routing}_${traffic}_${inj}.cfg"
                cat > $CFG << EOF
topology = $TOPO;
k = $K;
n = $N;
routing_function = $routing;
num_vcs = 2;
vc_buf_size = 4;
wait_for_tail_credit = 1;
priority = local_age;
sim_type = latency;
warmup_periods = 2;
sample_period = 1000;
sim_count = 1;
print_csv_results = 1;
traffic = $traffic;
injection_rate = $inj;
packet_size = 1;
EOF
                
                OUT="${OUTDIR}/${topo}_${routing}_${traffic}_inj${inj}.txt"
                
                echo "Running: $topo $routing $traffic @ $inj"
                $BOOKSIM $CFG 2>&1 | grep -E "Packet latency average|Injected packet rate|Accepted packet rate" | head -4 > $OUT
                
                # Check if it finished
                if [ -s $OUT ]; then
                    AVG_LAT=$(grep "Packet latency average" $OUT | head -1 | grep -oP '= \K[0-9.]+')
                    echo "  Latency: $AVG_LAT cycles" >> ${OUTDIR}/summary.csv
                    echo "  $topo,$routing,$traffic,$inj,$AVG_LAT" >> ${OUTDIR}/summary.csv
                else
                    echo "  FAILED (saturation)" >> ${OUTDIR}/summary.csv
                    echo "$topo,$routing,$traffic,$inj,FAILED" >> ${OUTDIR}/summary.csv
                fi
                
                rm -f $CFG
            done
        done
    done
done

echo ""
echo "=== DONE ==="
echo "Results saved in ${OUTDIR}/"
ls -la ${OUTDIR}/
head -20 ${OUTDIR}/summary.csv
