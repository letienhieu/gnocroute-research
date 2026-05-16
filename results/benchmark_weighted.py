#!/usr/bin/env python3
"""Benchmark GNN-Weighted Adaptive Routing against baselines."""

import subprocess, tempfile, re, os, csv, sys, time

BOOKSIM = '/home/opc/.openclaw/workspace/booksim2/src/booksim'
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'weighted_results.csv')

def run_booksim(routing, traffic, rate, seed=42):
    cfg = f"""topology = mesh;
k = 4;
n = 2;
routing_function = {routing};
traffic = {traffic};
injection_rate = {rate};
warmup_periods = 1000;
sample_period = 20000;
sim_count = 3;
sim_type = latency;
num_vcs = 4;
vc_buf_size = 8;
seed = {seed};
"""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    f.write(cfg)
    cfg_path = f.name
    f.close()
    try:
        out = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=120)
        stdout = out.stdout
        os.unlink(cfg_path)
        lat_match = re.findall(r'Packet latency average = ([0-9.]+)', stdout)
        return float(lat_match[-1]) if lat_match else None
    except subprocess.TimeoutExpired:
        os.unlink(cfg_path)
        return None
    except Exception as e:
        os.unlink(cfg_path)
        return None

# Test configs
algos = [
    ('dor', 'XY (DOR)'),
    ('adaptive_xy_yx', 'Adaptive XY-YX'),
    ('gnn_weighted_route_4x4', 'GNN-Weighted'),
]
traffics = ['uniform', 'transpose', 'hotspot']
rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
seeds = [42, 43, 44]

# Write header
with open(RESULTS_FILE, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['algorithm', 'traffic', 'injection_rate', 'seed', 'latency'])

total = len(algos) * len(traffics) * len(rates) * len(seeds)
done = 0
t0 = time.time()

print(f"Benchmark: {total} simulations")
print("=" * 60)

for routing, aname in algos:
    for traffic in traffics:
        for rate in rates:
            for seed in seeds:
                done += 1
                lat = run_booksim(routing, traffic, rate, seed)
                with open(RESULTS_FILE, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([aname, traffic, rate, seed, lat if lat else 'FAIL'])
                
                elapsed = time.time() - t0
                status = f"{lat:.2f}" if lat else "FAIL"
                pct = done / total * 100
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(f"[{pct:4.0f}%] {aname:12s} {traffic:10s} @{rate:.2f} s{seed}: {status:>8} | "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

print(f"\nDone! Results saved to {RESULTS_FILE}")
print(f"Total time: {time.time()-t0:.0f}s")

# Print summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

with open(RESULTS_FILE, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for traffic in traffics:
    print(f"\n=== {traffic.upper()} ===")
    print(f"{'Rate':>6}", end='')
    for _, aname in algos:
        print(f" {aname:>18}", end='')
    print()
    
    for rate in rates:
        print(f"{rate:>6.2f}", end='')
        for _, aname in algos:
            lats = [float(r['latency']) for r in rows 
                    if r['algorithm'] == aname and r['traffic'] == traffic 
                    and float(r['injection_rate']) == rate and r['latency'] != 'FAIL']
            if lats:
                avg = sum(lats) / len(lats)
                print(f" {avg:>18.2f}", end='')
            else:
                print(f" {'FAIL':>18}", end='')
        print()
