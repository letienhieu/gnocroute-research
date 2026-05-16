#!/usr/bin/env python3
"""
Analyze GNNocRoute-DRL BookSim2 results.
Reads all_results.csv, computes stats, generates tables and figures.
"""
import csv, statistics, math, sys, os
from collections import defaultdict

RESULTS = "/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results"
CSV_PATH = os.path.join(RESULTS, "all_results.csv")
OUT_DIR = "/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/latex/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    data = []
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return data
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('latency') and row['latency'].strip():
                data.append({
                    'topo': row['topo'],
                    'k': int(row['k']),
                    'n': int(row['n']),
                    'algo': row['algo'],
                    'traffic': row['traffic'],
                    'rate': float(row['inj_rate']),
                    'seed': int(row['seed']),
                    'latency': float(row['latency']),
                    'accepted': float(row['accepted']) if row.get('accepted') else 0,
                    'hops': float(row['hops']) if row.get('hops') else 0,
                })
    return data

def mean_ci(values):
    """Return (mean, half_width_95)"""
    n = len(values)
    if n < 2:
        return (statistics.mean(values) if values else 0, 0)
    m = statistics.mean(values)
    if n < 30:
        # Use t-distribution for small samples
        t = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
             11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,20:2.086,25:2.060,30:2.042}.get(n, 1.96)
    else:
        t = 1.96
    s = statistics.stdev(values) if n > 1 else 0
    return (m, t * s / math.sqrt(n))

def compute_aggregated(data):
    """Aggregate across seeds: mean ± CI per (topo, k, n, algo, traffic, rate)."""
    grouped = defaultdict(list)
    for d in data:
        key = (d['topo'], d['k'], d['n'], d['algo'], d['traffic'], d['rate'])
        grouped[key].append(d['latency'])
    
    agg = {}
    for key, vals in grouped.items():
        mean, ci = mean_ci(vals)
        agg[key] = {'mean': mean, 'ci': ci, 'n': len(vals)}
    return agg

def generate_latex_table(data):
    """Generate the main comparison table (Mesh 4x4, aggregated across rates)."""
    agg = compute_aggregated(data)
    
    # Filter: Mesh 4x4
    mesh4 = {k: v for k, v in agg.items() if k[0] == 'mesh' and k[1] == 4}
    
    # Group by (algo, traffic) across all rates
    algo_traffic = defaultdict(list)
    for (topo, k, n, algo, traffic, rate), val in mesh4.items():
        algo_traffic[(algo, traffic)].append(val['mean'])
    
    # Compute average across rates per (algo, traffic)
    results = {}
    for (algo, traffic), lats in algo_traffic.items():
        if (algo, traffic) not in results:
            results[(algo, traffic)] = []
        results[(algo, traffic)] = statistics.mean(lats) if lats else 0
    
    # Pretty algo names
    ALGO_NAMES = {
        'dor': 'XY (DOR)',
        'adaptive_xy_yx': 'Adaptive XY/YX',
        'min_adapt': 'Minimal Adaptive',
        'valiant': 'Valiant',
        'gnn_ppo_route_4x4': 'GNNocRoute-DRL',
    }
    
    print("=" * 80)
    print("TABLE: Average Packet Latency (cycles) on Mesh 4×4")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Uniform':>10} {'Transpose':>12} {'Hotspot':>10}")
    print("-" * 60)
    
    algos_order = ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant', 'gnn_ppo_route_4x4']
    for algo in algos_order:
        row = [ALGO_NAMES.get(algo, algo)]
        for traffic in ['uniform', 'transpose', 'hotspot']:
            key = (algo, traffic)
            if key in results:
                row.append(f"{results[key]:.1f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<25} {row[1]:>10} {row[2]:>12} {row[3]:>10}")
    
    # Compute improvement
    print("\n--- Improvement vs XY (%) ---")
    xy_vals = {t: results.get(('dor', t), 0) for t in ['uniform', 'transpose', 'hotspot']}
    for algo in algos_order[1:]:
        imprs = []
        for t in ['uniform', 'transpose', 'hotspot']:
            key = (algo, t)
            if key in results and xy_vals[t] > 0:
                impr = (xy_vals[t] - results[key]) / xy_vals[t] * 100
                imprs.append(f"{impr:.1f}%")
            else:
                imprs.append("N/A")
        print(f"{ALGO_NAMES.get(algo, algo):<25} {'/'.join(imprs)}")

def check_progress(data):
    """Report what's missing."""
    algos = set(d['algo'] for d in data)
    topos = set((d['topo'], d['k'], d['n']) for d in data)
    traffics = set(d['traffic'] for d in data)
    rates = sorted(set(d['rate'] for d in data))
    
    print(f"\nProgress report:")
    print(f"  Total observations: {len(data)}")
    print(f"  Topologies: {topos}")
    print(f"  Algorithms: {algos}")
    print(f"  Traffics: {traffics}")
    print(f"  Rates: {rates}")
    
    # Check Mesh 4x4 completeness
    for algo in ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant', 'gnn_ppo_route_4x4']:
        expected = 3 * 6 * 5  # traffics × rates × seeds
        found = sum(1 for d in data if d['algo'] == algo and d['topo'] == 'mesh' and d['k'] == 4)
        print(f"  Mesh 4x4 {algo}: {found}/{expected}")

def main():
    data = load_data()
    if not data:
        print("No data loaded.")
        return
    
    check_progress(data)
    generate_latex_table(data)

if __name__ == "__main__":
    main()
