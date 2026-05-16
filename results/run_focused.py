#!/usr/bin/env python3
"""
Focused batch runner — runs ONLY remaining experiments for Mesh 4x4.
Uses temp files, 4 parallel workers, minimal overhead.
"""
import subprocess, os, sys, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BOOKSIM = "/home/opc/.openclaw/workspace/booksim2/src/booksim"
RESULTS = Path("/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results")
RESULTS.mkdir(parents=True, exist_ok=True)

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
sample_period = 1000;
sim_count = 100;
seed = {seed};
"""

ALGOS = ['dor', 'adaptive_xy_yx', 'min_adapt', 'valiant', 'gnn_ppo_route_4x4']
TRAFFICS = ['uniform', 'transpose', 'hotspot']
RATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
SEEDS = [0, 1, 2, 3, 4]

def run_one(algo, traffic, rate, seed):
    fname = f"topo=mesh_k=4_n=4_algo={algo}_traffic={traffic}_inj={rate}_seed={seed}"
    outpath = RESULTS / f"{fname}.txt"
    if outpath.exists():
        return fname, "cached"
    
    cfg = CFG.format(algo=algo, traffic=traffic, rate=rate, seed=seed)
    cfg_path = f"/tmp/bs_{os.getpid()}_{int(time.time()*1e6)%1000000}.cfg"
    with open(cfg_path, "w") as f: f.write(cfg)
    try:
        r = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=30)
        outpath.write_text(r.stdout + r.stderr)
        ok = "Packet latency average" in (r.stdout + r.stderr)
        return fname, "ok" if ok else "bad"
    except subprocess.TimeoutExpired:
        return fname, "timeout"
    except Exception as e:
        return fname, f"err:{e}"
    finally:
        if os.path.exists(cfg_path): os.unlink(cfg_path)

def parse_fname(fname):
    """Parse filename like: topo=mesh_k=4_n=4_algo=gnn_ppo_route_4x4_traffic=uniform_inj=0.01_seed=0"""
    import re
    m = re.match(r'topo=(\w+)_k=(\d+)_n=(\d+)_algo=(.+?)_traffic=(\w+)_inj=([\d.]+)_seed=(\d+)', fname)
    if not m:
        return None
    return {'topo': m.group(1), 'k': int(m.group(2)), 'n': int(m.group(3)),
            'algo': m.group(4), 'traffic': m.group(5),
            'rate': float(m.group(6)), 'seed': int(m.group(7))}

def rebuild_csv():
    """Rebuild CSV from all result files."""
    rows = []
    for f in sorted(RESULTS.glob("topo=*.txt")):
        base = f.stem
        p = parse_fname(base)
        if not p:
            continue
        text = f.read_text()
        lat = re.findall(r'Packet latency average = ([\d.]+)', text)
        acc = re.findall(r'Accepted flit rate average = ([\d.]+)', text)
        hop = re.findall(r'Hops average = ([\d.]+)', text)
        if lat:
            rows.append(f"{p['topo']},{p['k']},{p['n']},{p['algo']},{p['traffic']},{p['rate']},{p['seed']},{lat[-1]},{acc[-1] if acc else ''},{hop[-1] if hop else ''}")
    csv_path = RESULTS / "all_results.csv"
    with open(csv_path, "w") as f:
        f.write("topo,k,n,algo,traffic,inj_rate,seed,latency,accepted,hops\n")
        f.write("\n".join(rows))
    print(f"CSV: {csv_path} ({len(rows)} rows)")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--csv", action="store_true", help="Just rebuild CSV")
    parser.add_argument("--gnn", action="store_true", help="Only GNN-PPO experiments")
    args = parser.parse_args()
    
    if args.csv:
        rebuild_csv()
        return
    
    # Build task list
    tasks = []
    for algo in ALGOS:
        if args.gnn and algo != 'gnn_ppo_route_4x4':
            continue
        for traffic in TRAFFICS:
            for rate in RATES:
                for seed in SEEDS:
                    tasks.append((algo, traffic, rate, seed))
    
    existing = sum(1 for t in tasks if (RESULTS / f"topo=mesh_k=4_n=4_algo={t[0]}_traffic={t[1]}_inj={t[2]}_seed={t[3]}.txt").exists())
    remaining = len(tasks) - existing
    
    print(f"Total tasks: {len(tasks)} ({args.gnn and 'GNN-only' or 'all 5 algos'})")
    print(f"Existing:    {existing}")
    print(f"Remaining:   {remaining}")
    
    if remaining == 0:
        print("All done!")
        rebuild_csv()
        return
    
    t0 = time.time()
    ok = 0
    fail = 0
    
    todo = [t for t in tasks if not (RESULTS / f"topo=mesh_k=4_n=4_algo={t[0]}_traffic={t[1]}_inj={t[2]}_seed={t[3]}.txt").exists()]
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut_map = {pool.submit(run_one, *t): t for t in todo}
        for i, future in enumerate(as_completed(fut_map)):
            fname, status = future.result()
            if status == "ok" or status == "cached":
                ok += 1
            else:
                fail += 1
                print(f"  FAIL: {fname} -> {status}")
            
            if (i+1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(todo)}] ok={ok} fail={fail} ({ok/elapsed:.1f} runs/s)")
    
    elapsed = time.time() - t0
    print(f"\nDone: {ok} ok, {fail} fail in {elapsed:.0f}s ({ok/elapsed:.1f} runs/s)")
    rebuild_csv()

if __name__ == "__main__":
    main()
