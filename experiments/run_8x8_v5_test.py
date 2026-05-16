#!/usr/bin/env python3
"""Run 8x8 v5 test: compare v4 vs v5 vs DOR vs adaptive_xy_yx at 0.05 rate."""

import subprocess, os, re, time
from pathlib import Path

BOOKSIM = "/home/opc/.openclaw/workspace/booksim2/src/booksim"
OUTDIR = Path("/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results_8x8_v5")
OUTDIR.mkdir(parents=True, exist_ok=True)

CFG_TEMPLATE = """topology = mesh;
k = 8;
n = 2;
routing_function = {algo};
num_vcs = 4;
vc_buf_size = 4;
wait_for_tail_credit = 1;
traffic = {traffic};
packet_size = 1;
sim_type = latency;
injection_rate = {rate};
warmup_periods = 3;
sample_period = 1000;
sim_count = 1;
seed = {seed};
max_samples = 50;
"""

ALGOS = [
    "dor",
    "adaptive_xy_yx",
    "gnn_port_score_route_8x8_v4",
    "gnn_port_score_route_8x8_v5",
]
TRAFFICS = ["uniform", "transpose", "hotspot"]
RATES = [0.05]
SEEDS = [0]

results = []

for algo in ALGOS:
    for traffic in TRAFFICS:
        for rate in RATES:
            for seed in SEEDS:
                fname = f"{algo}_{traffic}_r{rate}_s{seed}.txt"
                outpath = OUTDIR / fname
                
                # Build hotspot with center node
                t = traffic
                if traffic == "hotspot":
                    t = "hotspot(35,10)"  # center of 8x8 near (3,4)
                
                cfg = CFG_TEMPLATE.format(algo=algo, traffic=t, rate=rate, seed=seed)
                
                with open("/tmp/bs_8x8_v5_test.cfg", "w") as f:
                    f.write(cfg)
                
                print(f"  Running: {algo:45s} {traffic:12s} rate={rate} seed={seed}", end=" ", flush=True)
                t0 = time.time()
                
                r = subprocess.run(
                    [BOOKSIM, "/tmp/bs_8x8_v5_test.cfg"],
                    capture_output=True, text=True, timeout=120
                )
                
                dt = time.time() - t0
                output = r.stdout + r.stderr
                
                # Write full output
                with open(outpath, "w") as f:
                    f.write(f"Algorithm: {algo}\n")
                    f.write(f"Traffic: {traffic}\n")
                    f.write(f"Rate: {rate}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Elapsed: {dt:.0f}s\n")
                    f.write(f"\nBEGIN Configuration File: /tmp/bs_8x8_v5_test.cfg\n")
                    f.write(cfg)
                    f.write(f"END Configuration File: /tmp/bs_8x8_v5_test.cfg\n")
                    f.write(output)
                
                # Parse latency
                lat_match = re.findall(r'Packet latency average = ([\d.]+)', output)
                latency = float(lat_match[-1]) if lat_match else -1
                
                results.append({
                    'algo': algo, 'traffic': traffic, 'rate': rate, 'seed': seed,
                    'latency': latency, 'elapsed': dt,
                })
                
                print(f"→ latency={latency:.1f} ({dt:.1f}s)")

# Summary table
print("\n" + "="*90)
print(f"{'Algorithm':45s} {'Traffic':12s} {'Rate':6s} {'Latency':10s} {'Time':8s}")
print("="*90)
for r in results:
    print(f"{r['algo']:45s} {r['traffic']:12s} {r['rate']:6.2f} {r['latency']:10.1f} {r['elapsed']:7.1f}s")
print("="*90)

# Comparison summary
print("\n=== COMPARISON (seed=0, rate=0.05) ===")
for traffic in TRAFFICS:
    print(f"\n--- {traffic.upper()} ---")
    for r in results:
        if r['traffic'] != traffic: continue
        pct_str = ""
        if r['algo'] != 'dor':
            dor_lat = [x['latency'] for x in results if x['traffic']==traffic and x['algo']=='dor'][0]
            if dor_lat > 0:
                pct = (dor_lat - r['latency']) / dor_lat * 100
                pct_str = f" ({pct:+.1f}% vs DOR)"
        print(f"  {r['algo']:45s} → {r['latency']:8.1f} cycles{pct_str}")
