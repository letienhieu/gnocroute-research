#!/usr/bin/env python3
"""Final BookSim2 benchmark comparing GNN-PPO routing variants vs baselines."""
import subprocess, tempfile, re, json, os, time

BOOKSIM = '/home/opc/.openclaw/workspace/booksim2/src/booksim'
RESULTS = os.path.join(os.path.dirname(__file__), 'final_benchmark_results.json')

def run_once(algo, traffic, inj):
    """Run BookSim2 once and return latency."""
    cfg = f"""topology = mesh;
k = 4;
n = 2;
routing_function = {algo};
traffic = {traffic};
injection_rate = {inj};
warmup_periods = 500;
sample_period = 10000;
sim_count = 5;
sim_type = latency;
num_vcs = 4;
vc_buf_size = 8;
"""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    f.write(cfg)
    cfg_path = f.name
    f.close()
    
    try:
        out = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=30)
        stdout = out.stdout
        lat = re.search(r'Packet latency average\s*=\s*([0-9.]+)', stdout)
        if lat:
            result = float(lat.group(1))
        else:
            result = None  # saturated
        os.unlink(cfg_path)
        return result
    except:
        os.unlink(cfg_path)
        return None

# All test configurations
tests = []
for algo in ['dor', 'adaptive_xy_yx', 'min_adapt', 'gnn_ppo_route_4x4']:
    for traffic in ['uniform', 'transpose', 'hotspot']:
        for inj in [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]:
            tests.append((algo, traffic, inj))

print(f"Total benchmarks: {len(tests)}")
results = []
t0 = time.time()

for i, (algo, traffic, inj) in enumerate(tests):
    lat = run_once(algo, traffic, inj)
    results.append({
        'algo': algo, 'traffic': traffic, 'inj': inj, 'latency': lat
    })
    
    pct = (i+1)/len(tests)*100
    elapsed = time.time() - t0
    eta = elapsed/(i+1)*(len(tests)-i-1) if i > 0 else 0
    
    symbol = '✅' if lat else '⚠️'
    val = f"{lat:.1f}" if lat else "SAT"
    print(f"{symbol} [{i+1}/{len(tests)}] {elapsed:.0f}s | {algo:20s} {traffic:12s} @{inj:.2f} → {val} (ETA: {eta:.0f}s)")

# Save
with open(RESULTS, 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print(f"\n{'='*110}")
print(f"{'Algorithm':20s} {'Traffic':12s} {'0.02':>8} {'0.05':>8} {'0.10':>8} {'0.20':>8} {'0.30':>8} {'0.40':>8}")
print(f"{'='*110}")
for algo in ['dor', 'adaptive_xy_yx', 'min_adapt', 'gnn_ppo_route_4x4']:
    for traffic in ['uniform', 'transpose', 'hotspot']:
        row = [r for r in results if r['algo']==algo and r['traffic']==traffic]
        row.sort(key=lambda r: r['inj'])
        vals = ' '.join(f"{r['latency']:>8.1f}" if r['latency'] else f"{'SAT':>8}" for r in row)
        print(f"{algo:20s} {traffic:12s} {vals}")
    print()

# Improvement analysis
print(f"\n{'='*60}")
print(f"GNN-PPO vs XY (dor) - Latency Reduction %")
print(f"{'='*60}")
for traffic in ['uniform', 'transpose', 'hotspot']:
    for inj in [0.05, 0.10, 0.20, 0.30]:
        xy = next((r['latency'] for r in results if r['algo']=='dor' and r['traffic']==traffic and r['inj']==inj), None)
        gn = next((r['latency'] for r in results if r['algo']=='gnn_ppo_route_4x4' and r['traffic']==traffic and r['inj']==inj), None)
        ad = next((r['latency'] for r in results if r['algo']=='adaptive_xy_yx' and r['traffic']==traffic and r['inj']==inj), None)
        
        line = f"  {traffic:12s} @{inj:.2f}: XY={xy:.1f}"
        if gn:
            impr = (xy - gn) / xy * 100 if xy else 0
            line += f" GNN={gn:.1f} ({impr:+.1f}%)"
        else:
            line += f" GNN=SAT"
        if ad:
            aimpr = (xy - ad) / xy * 100 if xy else 0
            line += f" Adapt={ad:.1f} ({aimpr:+.1f}%)"
        print(line)

print(f"\nResults saved to {RESULTS}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
