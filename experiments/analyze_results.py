#!/usr/bin/env python3
"""
Analyze GNNocRoute-DRL experiment results.
Processes both Hướng A (clean mesh) and Hướng B (faulty mesh) results.
"""
import csv, re, sys, os
from collections import defaultdict
from pathlib import Path

def load_csv(path):
    """Load CSV results."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def print_comparison_table(results, group_by, value_key='latency', title="Results"):
    """Print comparison table grouped by specified field."""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r[g] for g in group_by)
        try:
            val = float(r[value_key])
        except:
            continue
        groups[key].append(val)
    
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    
    # Sort by group
    for key in sorted(groups.keys()):
        vals = groups[key]
        avg = sum(vals) / len(vals)
        key_str = " | ".join(str(k) for k in key)
        print(f"  {key_str:50s}  avg_{value_key}={avg:.4f}  (n={len(vals)})")
    print()

def analyze_mesh4(results):
    """Analyze clean mesh 4x4 results (Hướng A)."""
    print("\n" + "="*70)
    print("  HƯỚNG A: Clean Mesh 4x4 Results")
    print("="*70)
    
    # Summary by algo and traffic
    print("\n--- Average Latency by Algorithm and Traffic Pattern ---")
    print(f"{'Algorithm':30s} {'Traffic':12s} {'Rate':8s} {'Avg Lat':10s}")
    print("-" * 60)
    
    data = defaultdict(list)
    for r in results:
        if r['traffic'] == 'traffic': continue  # header
        key = (r['algo'], r['traffic'], float(r['inj_rate']))
        data[key].append(float(r['latency']))
    
    for key in sorted(data.keys()):
        algo, traffic, rate = key
        vals = data[key]
        avg = sum(vals)/len(vals)
        print(f"  {algo:30s} {traffic:12s} {rate:<8.2f} {avg:<10.2f}")
    
    # Best algorithm per pattern
    print("\n\n--- Best Algorithm per (Traffic, Rate) ---")
    best = {}
    for (algo, traffic, rate), vals in data.items():
        avg = sum(vals)/len(vals)
        tr_key = (traffic, rate)
        if tr_key not in best or avg < best[tr_key][1]:
            best[tr_key] = (algo, avg)
    
    print(f"{'Traffic':12s} {'Rate':8s} {'Best Algo':30s} {'Latency':10s}")
    print("-" * 60)
    for key in sorted(best.keys()):
        traffic, rate = key
        algo, avg = best[key]
        print(f"  {traffic:12s} {rate:<8.2f} {algo:30s} {avg:<10.2f}")

def analyze_faulty(results):
    """Analyze faulty mesh results (Hướng B)."""
    print("\n" + "="*70)
    print("  HƯỚNG B: Faulty Mesh Results")
    print("="*70)
    
    if not results:
        print("  No faulty results yet.")
        return
    
    data = defaultdict(list)
    for r in results:
        try:
            key = (r['algo'], r['traffic'], float(r['inj_rate']), int(r['link_failures']))
            data[key].append(float(r['latency']))
        except:
            continue
    
    # For each fault level and traffic, compare algorithms
    print(f"\n--- Latency vs Fault Level (averaged over seeds) ---")
    print(f"{'Traffic':12s} {'Fails':6s} {'dor':12s} {'planar_adapt':16s} {'gnn_port_score':16s}")
    print("-" * 62)
    
    for traffic in ['uniform', 'transpose', 'hotspot']:
        for fails in [0, 1, 2, 4, 7]:
            row = f"  {traffic:12s} {fails:<6d}"
            for algo in ['dor', 'planar_adapt', 'gnn_port_score_route_4x4']:
                key = (algo, traffic, 0.05, fails)  # rate = 0.05
                if key in data:
                    vals = data[key]
                    avg = sum(vals)/len(vals)
                    row += f" {avg:<12.2f}"
                else:
                    row += " N/A          "
            print(row)

def main():
    # Hướng A: Clean mesh
    csv_a = Path("/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results/all_results.csv")
    if csv_a.exists():
        results_a = load_csv(csv_a)
        analyze_mesh4(results_a)
    
    # Hướng B: Faulty mesh
    csv_b = Path("/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results_faulty/faulty_results.csv")
    if csv_b.exists():
        results_b = load_csv(csv_b)
        analyze_faulty(results_b)

if __name__ == "__main__":
    main()
