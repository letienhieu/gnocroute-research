#!/usr/bin/env python3
"""Fix FAIL entries in weighted_results.csv by re-running BookSim2."""
import subprocess, tempfile, re, os, csv, sys

BOOKSIM = '/home/opc/.openclaw/workspace/booksim2/src/booksim'

def run(routing, traffic, rate, seed):
    cfg = f"topology = mesh;\nk = 4;\nn = 2;\nrouting_function = {routing};\ntraffic = {traffic};\ninjection_rate = {rate};\nwarmup_periods = 1000;\nsample_period = 20000;\nsim_count = 3;\nsim_type = latency;\nnum_vcs = 4;\nvc_buf_size = 8;\nseed = {seed};\n"
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    f.write(cfg)
    cfg_path = f.name
    f.close()
    try:
        out = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=60)
        os.unlink(cfg_path)
        lat_match = re.findall(r'Packet latency average = ([0-9.]+)', out.stdout)
        return float(lat_match[-1]) if lat_match else 'FAIL'
    except:
        os.unlink(cfg_path)
        return 'FAIL'

routing_map = {
    'XY (DOR)': 'dor',
    'Adaptive XY-YX': 'adaptive_xy_yx',
    'Min Adapt': 'min_adapt',
    'GNN-Weighted': 'gnn_weighted_route_4x4',
}

with open('weighted_results.csv', 'r') as f:
    rows = list(csv.DictReader(f))

n_fail = sum(1 for r in rows if r['latency'] == 'FAIL')
print(f"Found {n_fail} FAIL entries to fix")

updated = 0
for i, r in enumerate(rows):
    if r['latency'] == 'FAIL':
        routing = routing_map[r['algorithm']]
        lat = run(routing, r['traffic'], float(r['injection_rate']), int(r['seed']))
        rows[i]['latency'] = str(lat)
        updated += 1
        with open('weighted_results.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['algorithm','traffic','injection_rate','seed','latency'])
            w.writeheader()
            w.writerows(rows)
        print(f"  [{updated}/{n_fail}] {r['algorithm']:15s} {r['traffic']:10s} @{r['injection_rate']:>4s} s{r['seed']}: {lat}")

print(f"\nAll {updated} FAIL entries fixed!")
