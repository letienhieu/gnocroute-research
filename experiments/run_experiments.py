#!/usr/bin/env python3
"""
GNNocRoute-DRL: Batch simulation runner for BookSim2.
Runs all routing algorithms across topologies, traffic patterns, and injection rates.
"""
import subprocess, os, csv, time, sys, re, argparse
from pathlib import Path

BOOKSIM = "/home/opc/.openclaw/workspace/booksim2/src/booksim"
RESULTS_DIR = "/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config template ──
def make_cfg(topo, k, n, algo, traffic, inj_rate, seed):
    if topo == "torus":
        routing = algo  # torus already in name for some
    elif topo == "mesh":
        # BookSim2 appends "_mesh" to routing_function name for mesh topology
        if algo in ("dor", "adaptive_xy_yx", "min_adapt", "valiant", "gnn_ppo_route_4x4", "gnn_ppo_route_8x8", "gnn_ppo_route_16x16"):
            pass  # these already have correct names
        routing = algo
    
    lines = f"""topology = {topo};
k = {k};
n = {n};
routing_function = {algo};
num_vcs = 4;
vc_buf_size = 4;
wait_for_tail_credit = 1;
vc_allocator = islip;
sw_allocator = islip;
alloc_iters = 1;
credit_delay = 2;
routing_delay = 1;
vc_alloc_delay = 1;
sw_alloc_delay = 1;
input_speedup = 2;
output_speedup = 1;
internal_speedup = 1.0;
traffic = {traffic};
packet_size = 1;
sim_type = latency;
injection_rate = {inj_rate};
warmup_periods = 3;
sample_period = 1000;
sim_count = 1;
seed = {seed};
"""
    return lines

# ── Parsing functions ──
def parse_output(text):
    lat = re.search(r'Packet latency average = (\S+)', text)
    acc = re.search(r'Accepted flit rate average = (\S+)', text)
    hops = re.search(r'Hops average = (\S+)', text)
    inj = re.search(r'Injected flit rate average = (\S+)', text)
    return {
        'latency': float(lat.group(1)) if lat else None,
        'accepted': float(acc.group(1)) if acc else None,
        'hops': float(hops.group(1)) if hops else None,
        'injected': float(inj.group(1)) if inj else None,
    }

# ── Experiment definitions ──
def get_experiments():
    """Generate all experiment configs."""
    topos = [
        ("mesh", 4, 4),
        ("mesh", 8, 8),
        ("torus", 4, 4),
    ]
    traffics = ["uniform", "transpose", "hotspot"]
    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    seeds = list(range(0, 5))
    
    # Baseline algorithms
    baselines = ["dor", "adaptive_xy_yx", "min_adapt", "valiant"]
    # GNN-PPO algorithms (per topology)
    gnn_map = {
        (4, 4): "gnn_ppo_route_4x4",   # Mesh 4x4
        (8, 8): "gnn_ppo_route_8x8",   # Mesh 8x8
    }
    # Note: gnn_ppo_route_16x16 is for Mesh 16x16, not Torus. Torus doesn't have GNN routing.
    
    experiments = []
    
    for topo_name, k, n in topos:
        for traffic in traffics:
            for rate in rates:
                for seed in seeds:
                    # Baselines
                    for algo in baselines:
                        experiments.append((topo_name, k, n, algo, traffic, rate, seed))
                    # GNN-PPO (only for mesh topologies that have routing tables)
                    if topo_name == "mesh" and (k, n) in gnn_map:
                        experiments.append((topo_name, k, n, gnn_map[(k, n)], traffic, rate, seed))
                    
    return experiments

def run_single(topo, k, n, algo, traffic, rate, seed, force=False):
    """Run one BookSim2 simulation."""
    cfg_text = make_cfg(topo, k, n, algo, traffic, rate, seed)
    outfile = f"topo={topo}_k={k}_n={n}_algo={algo}_traffic={traffic}_inj={rate}_seed={seed}"
    outpath = os.path.join(RESULTS_DIR, outfile + ".txt")
    
    if os.path.exists(outpath) and not force:
        with open(outpath) as f:
            text = f.read()
        return parse_output(text)
    
    # Write temp config
    cfg_path = f"/tmp/bs2_{int(time.time())}_{os.getpid()}.cfg"
    with open(cfg_path, "w") as f:
        f.write(cfg_text)
    
    try:
        result = subprocess.run(
            [BOOKSIM, cfg_path],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        with open(outpath, "w") as f:
            f.write(output)
        parsed = parse_output(output)
        os.remove(cfg_path)
        return parsed
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        if os.path.exists(cfg_path):
            os.remove(cfg_path)
        return None

def main():
    parser = argparse.ArgumentParser(description="Run GNNocRoute BookSim2 batch experiments")
    parser.add_argument("--algo", help="Filter: specific algorithm")
    parser.add_argument("--topo", help="Filter: mesh or torus")
    parser.add_argument("--traffic", help="Filter: uniform, transpose, hotspot")
    parser.add_argument("--rate", type=float, help="Filter: specific injection rate")
    parser.add_argument("--limit", type=int, default=None, help="Max experiments to run")
    parser.add_argument("--force", action="store_true", help="Re-run existing")
    parser.add_argument("--dry", action="store_true", help="Just count, don't run")
    args = parser.parse_args()
    
    experiments = get_experiments()
    
    # Apply filters
    if args.algo:
        experiments = [e for e in experiments if e[3] == args.algo]
    if args.topo:
        experiments = [e for e in experiments if e[0] == args.topo]
    if args.traffic:
        experiments = [e for e in experiments if e[4] == args.traffic]
    if args.rate is not None:
        experiments = [e for e in experiments if abs(e[5] - args.rate) < 0.001]
    if args.limit:
        experiments = experiments[:args.limit]
    
    # Check existing
    existing = 0
    for e in experiments:
        outfile = f"topo={e[0]}_k={e[1]}_n={e[2]}_algo={e[3]}_traffic={e[4]}_inj={e[5]}_seed={e[6]}"
        outpath = os.path.join(RESULTS_DIR, outfile + ".txt")
        if os.path.exists(outpath):
            existing += 1
    
    print(f"Total experiments: {len(experiments)}")
    print(f"Already completed: {existing}")
    print(f"Remaining: {len(experiments) - existing}")
    
    if args.dry:
        return
    
    # Run experiments
    completed = 0
    errors = 0
    skipped = 0
    results = []
    
    for i, (topo, k, n, algo, traffic, rate, seed) in enumerate(experiments):
        label = f"[{i+1}/{len(experiments)}] {topo}{k}x{n} {algo} {traffic} @{rate} seed={seed}"
        
        outfile = f"topo={topo}_k={k}_n={n}_algo={algo}_traffic={traffic}_inj={rate}_seed={seed}"
        outpath = os.path.join(RESULTS_DIR, outfile + ".txt")
        
        if os.path.exists(outpath) and not args.force:
            skipped += 1
            with open(outpath) as f:
                parsed = parse_output(f.read())
            if parsed and parsed['latency'] is not None:
                results.append((topo, k, n, algo, traffic, rate, seed, parsed))
            continue
        
        print(f"  Running: {label}", flush=True)
        t0 = time.time()
        parsed = run_single(topo, k, n, algo, traffic, rate, seed, force=args.force)
        elapsed = time.time() - t0
        
        if parsed and parsed['latency'] is not None:
            completed += 1
            results.append((topo, k, n, algo, traffic, rate, seed, parsed))
            print(f"    ✓ lat={parsed['latency']:.1f} acc={parsed['accepted']:.4f} ({elapsed:.1f}s)")
        else:
            errors += 1
            print(f"    ✗ ERROR ({elapsed:.1f}s)")
    
    # Write aggregated CSV
    csv_path = os.path.join(RESULTS_DIR, "all_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topology", "k", "n", "algorithm", "traffic", "injection_rate", "seed", "latency", "accepted_flit_rate", "hops"])
        for topo, k, n, algo, traffic, rate, seed, parsed in results:
            writer.writerow([
                topo, k, n, algo, traffic, rate, seed,
                parsed['latency'], parsed['accepted'], parsed['hops']
            ])
    
    print(f"\n{'='*60}")
    print(f"Done! {completed} new, {skipped} cached, {errors} errors")
    print(f"Total results in CSV: {len(results)}")
    print(f"CSV: {csv_path}")

if __name__ == "__main__":
    main()
