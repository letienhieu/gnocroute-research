#!/usr/bin/env python3
"""
Efficient parallel BookSim2 batch runner for GNNocRoute experiments.
"""
import subprocess, os, csv, sys, re, time, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BOOKSIM = "/home/opc/.openclaw/workspace/booksim2/src/booksim"
RESULTS = Path("/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results")
RESULTS.mkdir(parents=True, exist_ok=True)

CFG = """topology = {topo};
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
injection_rate = {rate};
warmup_periods = 3;
sample_period = 1000;
sim_count = 100;
seed = {seed};
"""

def run_one(topo, k, n, algo, traffic, rate, seed):
    fname = f"topo={topo}_k={k}_n={n}_algo={algo}_traffic={traffic}_inj={rate}_seed={seed}"
    outpath = RESULTS / f"{fname}.txt"
    if outpath.exists():
        return fname, "cached"
    
    cfg = CFG.format(topo=topo, k=k, n=n, algo=algo, traffic=traffic, rate=rate, seed=seed)
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
        # filename format: topo=mesh_k=4_n=4_algo=dor_traffic=uniform_inj=0.01_seed=0
        try:
            parts = base.split("_")
            topo = parts[0].split("=")[1]
            k = int(parts[1].split("=")[1])
            n = int(parts[2].split("=")[1])
            algo = parts[3].split("=")[1]
            traffic = parts[4].split("=")[1]
            rate = float(parts[5].split("=")[1])
            seed = int(parts[6].split("=")[1])
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
                'latency': float(lat[-1]), 'accepted': float(acc[-1]) if acc else 0,
                'hops': float(hop[-1]) if hop else 0, 'injected': float(inj[-1]) if inj else 0
            })
        else:
            print(f"  SKIP (no data): {base}", file=sys.stderr)
    
    csv_path = RESULTS / "all_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=['topo','k','n','algo','traffic','inj_rate','seed','latency','accepted','hops','injected'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path} ({len(rows)} rows)")

def generate_experiments(limit_topology=None):
    """Yield (topo,k,n,algo,traffic,rate,seed) tuples."""
    if limit_topology == 'mesh4':
        topos = [('mesh',4,4)]
        gnn_algos = {('mesh',4,4): 'gnn_ppo_route_4x4'}
    elif limit_topology == 'mesh8':
        topos = [('mesh',8,8)]
        gnn_algos = {('mesh',8,8): 'gnn_ppo_route_8x8'}
    else:
        topos = [('mesh',4,4), ('mesh',8,8), ('torus',4,4)]
        gnn_algos = {('mesh',4,4): 'gnn_ppo_route_4x4', ('mesh',8,8): 'gnn_ppo_route_8x8'}
    
    baselines = ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant']
    traffics = ['uniform', 'transpose', 'hotspot']
    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    seeds = list(range(5))
    
    for topo, k, n in topos:
        for algo in baselines:
            for traffic in traffics:
                for rate in rates:
                    for seed in seeds:
                        yield (topo, k, n, algo, traffic, rate, seed)
        if (topo, k, n) in gnn_algos:
            algo = gnn_algos[(topo, k, n)]
            for traffic in traffics:
                for rate in rates:
                    for seed in seeds:
                        yield (topo, k, n, algo, traffic, rate, seed)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo", choices=['mesh4','mesh8'], help="Limit to specific topology")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--check", action="store_true", help="Just count")
    parser.add_argument("--csv", action="store_true", help="Just rebuild CSV from existing files")
    args = parser.parse_args()
    
    if args.csv:
        make_csv()
        return
    
    experiments = list(generate_experiments(args.topo))
    total = len(experiments)
    existing = sum(1 for e in experiments if (RESULTS / f"topo={e[0]}_k={e[1]}_n={e[2]}_algo={e[3]}_traffic={e[4]}_inj={e[5]}_seed={e[6]}.txt").exists())
    remaining = total - existing
    
    print(f"Total experiments: {total}")
    print(f"Already cached:    {existing}")
    print(f"Need to run:       {remaining}")
    print(f"Parallel workers:  {args.workers}")
    if args.check:
        return
    if remaining == 0:
        print("All done! Rebuilding CSV...")
        make_csv()
        return
    
    # Submit only what's needed
    todo = [e for e in experiments if not (RESULTS / f"topo={e[0]}_k={e[1]}_n={e[2]}_algo={e[3]}_traffic={e[4]}_inj={e[5]}_seed={e[6]}.txt").exists()]
    
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
            
            if (i+1) % 25 == 0:
                elapsed = time.time() - t0
                rate_val = (i+1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(todo)}] ok={ok_count} fail={fail_count} ({rate_val:.1f} runs/s)")
    
    elapsed = time.time() - t0
    print(f"\nCompleted {ok_count} runs in {elapsed:.0f}s ({ok_count/elapsed:.1f} runs/s)")
    if fail_count > 0:
        print(f"Failed: {fail_count}")
    
    make_csv()

if __name__ == "__main__":
    main()
