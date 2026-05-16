#!/bin/bash
# Comprehensive benchmark: Fixed GNN-PPO vs baselines
# Tests: uniform, transpose, hotspot
# Rates: 0.05, 0.10, 0.20, 0.30, 0.40
# Algorithms: dor, adaptive_xy_yx, min_adapt, valiant, gnn_ppo_route_4x4

BOOKSIM="/home/opc/.openclaw/workspace/booksim2/src/booksim"
CONFIG_TEMPLATE="/tmp/bench_${$}.cfg"
RESULTS="/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/benchmark_results.csv"

echo "algo,traffic,inj,latency,hops" > $RESULTS

for algo in dor adaptive_xy_yx min_adapt valiant gnn_ppo_route_4x4; do
  for traffic in uniform transpose hotspot; do
    for inj in 0.05 0.10 0.20 0.30 0.40; do
      cat > $CONFIG_TEMPLATE << EOFCFG
topology = mesh;
k = 4;
n = 2;
routing_function = ${algo};
traffic = ${traffic};
injection_rate = ${inj};
warmup_periods = 500;
sample_period = 20000;
sim_count = 5;
sim_type = latency;
num_vcs = 4;
vc_buf_size = 8;
EOFCFG
      
      output=$(timeout 30 $BOOKSIM $CONFIG_TEMPLATE 2>&1)
      lat=$(echo "$output" | grep "Packet latency average" | tail -1 | awk '{print $4}')
      hops=$(echo "$output" | grep "Hops average" | tail -1 | awk '{print $4}')
      
      if [ ! -z "$lat" ]; then
        echo "${algo},${traffic},${inj},${lat},${hops}" >> $RESULTS
        echo "OK: ${algo} ${traffic} @${inj} → lat=${lat}"
      else
        # Check if saturated
        if echo "$output" | grep -q "SIMULATION\|exceeded\|Simulation"; then
          echo "${algo},${traffic},${inj},SAT,SAT" >> $RESULTS
          echo "SAT: ${algo} ${traffic} @${inj}"
        else
          echo "${algo},${traffic},${inj},ERR,ERR" >> $RESULTS
          echo "ERR: ${algo} ${traffic} @${inj}"
        fi
      fi
      
      rm -f $CONFIG_TEMPLATE
    done
  done
done

echo ""
echo "=== RESULTS ==="
cat $RESULTS | column -t -s,
