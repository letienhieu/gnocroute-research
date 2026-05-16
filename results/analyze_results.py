#!/usr/bin/env python3
"""
GNNocRoute-DRL: Comprehensive analysis for JSA Q1 paper.
Computes comparison tables and generates figures.
"""
import csv, math, statistics, os, re
from collections import defaultdict

RESULTS = "/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results"
OUT_DIR = "/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/latex/figures"
os.makedirs(OUT_DIR, exist_ok=True)

ALGO_NAMES = {
    'dor': 'XY (DOR)',
    'adaptive_xy_yx': 'Adaptive XY/YX',
    'min_adapt': 'Minimal Adaptive',
    'valiant': 'Valiant',
    'gnn_ppo_route_4x4': 'GNNocRoute-DRL',
}
ALGO_ORDER = ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant', 'gnn_ppo_route_4x4']

def load_data():
    """Load data, only using sim_count=100 (reliable) results."""
    data = []
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith('.txt'): continue
        fpath = os.path.join(RESULTS, f)
        text = open(fpath).read()
        if '(100 samples)' not in text:
            continue  # skip sim_count=1 runs
        m = re.match(r'topo=(\w+)_k=(\d+)_n=(\d+)_algo=(.+?)_traffic=(\w+)_inj=([\d.]+)_seed=(\d+)', f)
        if not m: continue
        lat = re.findall(r'Packet latency average = ([\d.]+)', text)
        acc = re.findall(r'Accepted flit rate average = ([\d.]+)', text)
        hop = re.findall(r'Hops average = ([\d.]+)', text)
        if lat:
            data.append({
                'topo': m.group(1), 'k': int(m.group(2)), 'n': int(m.group(3)),
                'algo': m.group(4), 'traffic': m.group(5),
                'rate': float(m.group(6)), 'seed': int(m.group(7)),
                'latency': float(lat[-1]),
                'accepted': float(acc[-1]) if acc else 0,
                'hops': float(hop[-1]) if hop else 0,
            })
    return data

def mean_ci(values):
    n = len(values)
    if n < 2: return (values[0] if values else 0, 0)
    m = statistics.mean(values)
    t_table = {2:4.303,3:3.182,4:2.776,5:2.571}
    t = min((t_table.get(n, 1.96), 1.96))
    s = statistics.stdev(values) if n > 1 else 0
    return (m, t * s / math.sqrt(n))

def compute_comparison(data):
    """Compute mean latency per (algo, traffic, rate) across seeds."""
    grouped = defaultdict(list)
    for d in data:
        if d['topo'] != 'mesh' or d['k'] != 4: continue
        key = (d['algo'], d['traffic'], d['rate'])
        grouped[key].append(d['latency'])
    agg = {}
    for key, vals in grouped.items():
        agg[key] = {'mean': statistics.mean(vals), 'n': len(vals)}
    return agg

def print_table_table(agg):
    """Print the main comparison table (one per injection rate)."""
    print("=" * 80)
    print("COMPARISON TABLE: Average packet latency (cycles) on Mesh 4×4")
    print("=" * 80)
    
    rates = sorted(set(k[2] for k in agg.keys()))
    
    for rate in rates:
        print(f"\n--- Injection rate = {rate} ---")
        print(f"{'Algorithm':<25} {'Uniform':>10} {'Transpose':>12} {'Hotspot':>10}")
        print("-" * 60)
        for algo in ALGO_ORDER:
            row = [ALGO_NAMES.get(algo, algo)]
            for t in ['uniform', 'transpose', 'hotspot']:
                key = (algo, t, rate)
                if key in agg:
                    row.append(f"{agg[key]['mean']:.1f}")
                else:
                    row.append("N/A")
            print(f"{row[0]:<25} {row[1]:>10} {row[2]:>12} {row[3]:>10}")
    
    # Improvement at rate=0.1
    print("\n--- Improvement vs XY at rate=0.1 ---")
    xy_01 = {}
    for t in ['uniform', 'transpose', 'hotspot']:
        key = ('dor', t, 0.1)
        xy_01[t] = agg[key]['mean'] if key in agg else 0
    
    for algo in ALGO_ORDER[1:]:
        imprs = []
        for t in ['uniform', 'transpose', 'hotspot']:
            key = (algo, t, 0.1)
            if key in agg and xy_01[t] > 0:
                pct = (xy_01[t] - agg[key]['mean']) / xy_01[t] * 100
                imprs.append(f"{pct:.1f}%")
            else:
                imprs.append("N/A")
        print(f"{ALGO_NAMES.get(algo, algo):<25} {' / '.join(imprs)}")

def print_gnn_analysis(agg):
    """Print GNN-PPO specific analysis."""
    print("\n" + "=" * 80)
    print("GNNocRoute-DRL ANALYSIS")
    print("=" * 80)
    
    # Compare GNN vs best baseline at each rate
    print(f"\n{'Rate':>6} {'Traffic':<12} {'GNN-PPO':>10} {'Best Baseline':>14} {'Delta %':>10}")
    print("-" * 60)
    rates = sorted(set(k[2] for k in agg.keys()))
    for r in rates:
        for t in ['uniform', 'transpose', 'hotspot']:
            gnn_key = ('gnn_ppo_route_4x4', t, r)
            if gnn_key not in agg: continue
            gnn = agg[gnn_key]['mean']
            # Find best baseline
            best = min((agg.get((a, t, r), {'mean': float('inf')})['mean'] for a in ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant'] if (a, t, r) in agg))
            delta = (best - gnn) / best * 100
            best_name = min((agg.get((a, t, r), {'mean': float('inf')}) for a in ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant'] if (a, t, r) in agg), key=lambda x: x['mean'])['mean'] if any((a, t, r) in agg for a in ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant']) else 0
            actual_best_algo = [a for a in ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant'] if (a, t, r) in agg]
            if actual_best_algo:
                best_val = min(agg[(a, t, r)]['mean'] for a in actual_best_algo)
                delta = (best_val - gnn) / best_val * 100
            else:
                delta = 0
            print(f"{r:>6.2f} {t:<12} {gnn:>10.1f} {best_val if actual_best_algo else 0:>14.1f} {delta:>+9.1f}%")

def print_coverage(data):
    """Print data coverage report."""
    print("\n" + "=" * 80)
    print("DATA COVERAGE (sim_count=100 only)")
    print("=" * 80)
    
    algos = set(d['algo'] for d in data)
    for algo in sorted(algos):
        count = sum(1 for d in data if d['algo'] == algo and d['topo'] == 'mesh'
                    and d['k'] == 4 and d['n'] == 4)
        expected = 3 * 6 * 5  # traffics × rates × seeds
        print(f"  Mesh 4x4 {algo}: {count}/{expected}")

def main():
    data = load_data()
    print(f"Loaded {len(data)} valid results (sim_count=100)")
    print_coverage(data)
    agg = compute_comparison(data)
    print_table_table(agg)
    print_gnn_analysis(agg)

if __name__ == "__main__":
    main()
