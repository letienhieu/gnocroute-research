#!/usr/bin/env python3
"""
Faulty Topology Experiments for GNNocRoute-DRL (Hướng B)
========================================================
Compare DOR vs planar_adapt vs GNN port-score on meshes with link failures.

Link failures via BookSim2's link_failures + fail_seed parameters.
Uses fixed routing functions (no fault-awareness modification needed for comparison).

Author: Ngoc Anh for Thay Hieu
Date: 16/05/2026
"""

import subprocess, os, csv, sys, re, time, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BOOKSIM = "/home/opc/.openclaw/workspace/booksim2/src/booksim"
RESULTS = Path("/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results_faulty")
RESULTS.mkdir(parents=True, exist_ok=True)

# Config template with link_failures parameter
CFG = """topology = mesh;
k = 4;
n = 2;
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
injection_rate = {rate};
warmup_periods = 3;
sample_period = 500;
sim_count = 50;
seed = {seed};
link_failures = {fails};
fail_seed = {fail_seed};
"""

def run_one(topo, k, n, algo, traffic, rate, seed, fails, fail_seed):
    """Run a single BookSim2 simulation."""
    fname = f"topo={topo}_k={k}_n={n}_algo={algo}_traffic={traffic}_inj={rate}_seed={seed}_fails={fails}_fseed={fail_seed}"
    outpath = RESULTS / f"{fname}.txt"
    if outpath.exists():
        return fname, "cached"
    
    cfg = CFG.format(topo=topo, k=k, n=n, algo=algo, traffic=traffic, rate=rate, seed=seed, fails=fails, fail_seed=fail_seed)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False, dir='/tmp') as f:
        f.write(cfg)
        cfg_path = f.name
    try:
        r = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=30)
        outpath.write_text(r.stdout + r.stderr)
        os.unlink(cfg_path)
        if "Packet latency average" in (r.stdout + r.stderr):
            return fname, "ok"
        else:
            return fname, "no_output"
    except subprocess.TimeoutExpired:
        if os.path.exists(cfg_path): os.unlink(cfg_path)
        return fname, "timeout"
    except Exception as e:
        if os.path.exists(cfg_path): os.unlink(cfg_path)
        return fname, f"error:{e}"

def make_csv():
    """Parse all result files into CSV."""
    rows = []
    for f in sorted(RESULTS.glob("topo=*.txt")):
        base = f.stem
        try:
            parts = base.split("_")
            topo = parts[0].split("=")[1]
            k = int(parts[1].split("=")[1])
            n = int(parts[2].split("=")[1])
            algo = parts[3].split("=")[1]
            traffic = parts[4].split("=")[1]
            rate = float(parts[5].split("=")[1])
            seed = int(parts[6].split("=")[1])
            fails = int(parts[7].split("=")[1])
            fseed = int(parts[8].split("=")[1])
        except:
            print(f"  SKIP (parse fail): {base}", file=sys.stderr)
            continue
        
        text = f.read_text()
        lat = re.findall(r'Packet latency average = ([\d.]+)', text)
        acc = re.findall(r'Accepted flit rate average = ([\d.]+)', text)
        hop = re.findall(r'Hops average = ([\d.]+)', text)
        inj = re.findall(r'Injected flit rate average = ([\d.]+)', text)
        
        if lat:
            rows.append({
                'topo': topo, 'k': k, 'n': n, 'algo': algo,
                'traffic': traffic, 'inj_rate': rate, 'seed': seed,
                'link_failures': fails, 'fail_seed': fseed,
                'latency': float(lat[-1]), 'accepted': float(acc[-1]) if acc else 0,
                'hops': float(hop[-1]) if hop else 0, 'injected': float(inj[-1]) if inj else 0
            })
        else:
            print(f"  SKIP (no data): {base}", file=sys.stderr)
    
    csv_path = RESULTS / "faulty_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=['topo','k','n','algo','traffic','inj_rate','seed','link_failures','fail_seed','latency','accepted','hops','injected'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path} ({len(rows)} rows)")

def generate_experiments():
    """Generate all faulty topology experiments."""
    algos = ['dor', 'planar_adapt', 'gnn_port_score_route_4x4']
    traffics = ['uniform', 'transpose', 'hotspot']
    
    # Low injection rate to avoid saturation effect
    rates = [0.01, 0.05, 0.1]
    
    # Seeds for BookSim2 randomness (routing decisions within simulation)
    seeds = list(range(3))
    
    # Fault configurations
    fault_rates = [0, 1, 2, 4, 7]  # 0%, ~2%, ~4%, ~8%, ~15% of 48 channels
    fail_seeds = [101, 202, 303]  # Different random fault patterns
    
    for algo in algos:
        for traffic in traffics:
            for rate in rates:
                for seed in seeds:
                    for fails in fault_rates:
                        for fseed in fail_seeds:
                            yield ('mesh', 4, 2, algo, traffic, rate, seed, fails, fseed)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--check", action="store_true", help="Just count")
    parser.add_argument("--csv", action="store_true", help="Just rebuild CSV")
    args = parser.parse_args()
    
    if args.csv:
        make_csv()
        return
    
    experiments = list(generate_experiments())
    total = len(experiments)
    
    # Count existing
    existing_count = 0
    for e in experiments:
        fname = f"topo={e[0]}_k={e[1]}_n={e[2]}_algo={e[3]}_traffic={e[4]}_inj={e[5]}_seed={e[6]}_fails={e[7]}_fseed={e[8]}"
        if (RESULTS / f"{fname}.txt").exists():
            existing_count += 1
    remaining = total - existing_count
    
    print(f"Total experiments: {total}")
    print(f"  Algos: {3} (dor, planar_adapt, gnn_port_score)")
    print(f"  Traffic: {3} (uniform, transpose, hotspot)")
    print(f"  Rates: {3} (0.01, 0.05, 0.1)")
    print(f"  Seeds: {3}")
    print(f"  Fault levels: {5} (0,1,2,4,7 failures)")
    print(f"  Fault seeds: {3}")
    print(f"Already cached: {existing_count}")
    print(f"Need to run:    {remaining}")
    
    if args.check:
        return
    if remaining == 0:
        make_csv()
        return
    
    todo = [e for e in experiments if not (RESULTS / f"topo={e[0]}_k={e[1]}_n={e[2]}_algo={e[3]}_traffic={e[4]}_inj={e[5]}_seed={e[6]}_fails={e[7]}_fseed={e[8]}.txt").exists()]
    
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut_map = {pool.submit(run_one, *e): e for e in todo}
        for i, future in enumerate(as_completed(fut_map)):
            fname, status = future.result()
            if status == "ok" or status == "cached":
                ok_count += 1
            else:
                fail_count += 1
                print(f"  FAIL: {fname} -> {status}")
                sys.stdout.flush()
            
            if (i+1) % 10 == 0 or (i+1) == len(todo):
                elapsed = time.time() - t0
                rate_val = (i+1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(todo)}] ok={ok_count} fail={fail_count} ({rate_val:.1f} runs/s)")
                sys.stdout.flush()
    
    elapsed = time.time() - t0
    print(f"\nCompleted {ok_count} runs in {elapsed:.0f}s ({ok_count/elapsed:.1f} runs/s)")
    if fail_count > 0:
        print(f"Failed: {fail_count}")
    
    make_csv()

if __name__ == "__main__":
    main()
