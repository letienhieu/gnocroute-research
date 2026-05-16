#!/bin/bash
# Quick test: compare v4 vs v5 vs DOR vs Planar Adapt on 8x8
# 3 traffics (uniform, transpose, hotspot) @ 0.05 injection rate, seed=0

BS=/home/opc/.openclaw/workspace/booksim2/src/booksim
OUTDIR=/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results_8x8_v5
mkdir -p "$OUTDIR"

ALGOS=(
  "dor"
  "adaptive_xy_yx"
  "gnn_port_score_route_8x8_v4_mesh"
  "gnn_port_score_route_8x8_v5_mesh"
)

TRAFFICS=("uniform" "transpose" "hotspot")
RATES=(0.05)

for algo in "${ALGOS[@]}"; do
  for traffic in "${TRAFFICS[@]}"; do
    for rate in "${RATES[@]}"; do
      seed=0
      output="${OUTDIR}/${algo}_${traffic}_r${rate}_s${seed}.txt"
      if [ -f "$output" ]; then
        echo "SKIP: $output already exists"
        continue
      fi
      
      echo "RUN: algo=$algo traffic=$traffic rate=$rate seed=$seed"
      
      $BS \
        topology=mesh \
        k=8 n=2 \
        routing_function="$algo" \
        traffic="$traffic" \
        injection_rate="$rate" \
        num_vcs=4 vc_buf_size=4 \
        wait_for_tail_credit=1 \
        packet_size=1 \
        sim_type=latency \
        warmup_periods=3 \
        sample_period=1000 \
        sim_count=1 \
        seed=$seed \
        max_samples=50 \
        2>/dev/null | tee "$output"
      
      echo ""
    done
  done
done

echo "=== DONE ==="
ls -la "$OUTDIR"
