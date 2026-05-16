#!/usr/bin/env python3
"""Clean comprehensive benchmark: GNN-PPO (fixed) vs baselines on BookSim2"""
import subprocess, tempfile, re, json, os

BOOKSIM = '/home/opc/.openclaw/workspace/booksim2/src/booksim'
RESULTS = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')

configs = []
for algo in ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant', 'gnn_ppo_route_4x4']:
    for traffic in ['uniform', 'transpose', 'hotspot']:
        for inj in [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]:
            configs.append((algo, traffic, inj))

results = []
for i, (algo, traffic, inj) in enumerate(configs):
    cfg_content = f"""topology = mesh;
k = 4;
n = 2;
routing_function = {algo};
traffic = {traffic};
injection_rate = {inj};
warmup_periods = 500;
sample_period = 20000;
sim_count = 5;
sim_type = latency;
num_vcs = 4;
vc_buf_size = 8;
"""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    f.write(cfg_content)
    cfg_path = f.name
    f.close()
    
    try:
        out = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=30)
        stdout = out.stdout
        lat_match = re.search(r'Packet latency average\s*=\s*([0-9.]+)', stdout)
        hops_match = re.search(r'Hops average\s*=\s*([0-9.]+)', stdout)
        
        lat = float(lat_match.group(1)) if lat_match else None
        hops = float(hops_match.group(1)) if hops_match else None
        
        status = 'OK' if lat else 'SAT'
        if lat and lat > 500: status = 'NEAR_SAT'
        
        row = {'algo': algo, 'traffic': traffic, 'inj': inj, 'latency': lat, 'hops': hops, 'status': status}
        results.append(row)
        
        symbol = '✅' if status == 'OK' else '⚠️'
        print(f"{symbol} [{i+1}/{len(configs)}] {algo:20s} {traffic:12s} @{inj:.2f} → lat={lat} hops={hops}")
        
    except subprocess.TimeoutExpired:
        results.append({'algo': algo, 'traffic': traffic, 'inj': inj, 'latency': None, 'hops': None, 'status': 'TIMEOUT'})
        print(f"⏰ [{i+1}/{len(configs)}] {algo:20s} {traffic:12s} @{inj:.2f} → TIMEOUT")
    except Exception as e:
        results.append({'algo': algo, 'traffic': traffic, 'inj': inj, 'latency': None, 'hops': None, 'status': f'ERROR: {e}'})
        print(f"❌ [{i+1}/{len(configs)}] {algo:20s} {traffic:12s} @{inj:.2f} → {e}")
    finally:
        os.unlink(cfg_path)

# Save
with open(RESULTS, 'w') as f:
    json.dump(results, f, indent=2)

# Summary table
print(f"\n{'='*100}")
print(f"{'Algorithm':20s} {'Traffic':12s} {'@0.02':>8} {'@0.05':>8} {'@0.10':>8} {'@0.20':>8} {'@0.30':>8} {'@0.40':>8}")
print(f"{'='*100}")
for traffic in ['uniform', 'transpose', 'hotspot']:
    for algo in ['dor', 'adaptive_xy_yx', 'min_adapt', 'gnn_ppo_route_4x4']:
        row = [r for r in results if r['algo']==algo and r['traffic']==traffic]
        row.sort(key=lambda r: r['inj'])
        vals = ' '.join(f"{r['latency']:>8.1f}" if r['latency'] and r['latency']<500 else f"{'SAT':>8}" for r in row)
        print(f"{algo:20s} {traffic:12s} {vals}")
    print()

# Improvement comparison
print(f"\n{'='*60}")
print("GNN-PPO vs XY Improvement (%):")
print(f"{'='*60}")
for traffic in ['uniform', 'transpose', 'hotspot']:
    for inj in [0.05, 0.10, 0.20, 0.30]:
        xy = next((r['latency'] for r in results if r['algo']=='dor' and r['traffic']==traffic and r['inj']==inj), None)
        gn = next((r['latency'] for r in results if r['algo']=='gnn_ppo_route_4x4' and r['traffic']==traffic and r['inj']==inj), None)
        if xy and gn and xy < 500 and gn < 500:
            impr = (xy - gn) / xy * 100
            print(f"  {traffic:12s} @{inj:.2f}: XY={xy:.1f} GNN={gn:.1f} → {'+' if impr>=0 else ''}{impr:.1f}%")
        else:
            print(f"  {traffic:12s} @{inj:.2f}: XY={xy} GNN={gn}")

print(f"\nResults saved to {RESULTS}")
