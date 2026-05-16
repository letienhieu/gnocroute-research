#!/usr/bin/env python3
"""
GNNocRoute batch simulation runner — efficient, non-spammy, properly parses final stats.
"""
from __future__ import annotations
import subprocess, os, csv, sys, re, time
from pathlib import Path

BOOKSIM = "/home/opc/.openclaw/workspace/booksim2/src/booksim"
RESULTS = Path("/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results")
RESULTS.mkdir(parents=True, exist_ok=True)

CFG_TEMPLATE = """topology = {topo};
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

def parse_final(output: str) -> dict:
    """Extract final overall stats from BookSim2 output."""
    res = {}
    # Find the last occurrence of each stat (the final overall one)
    lat_matches = [m for m in re.finditer(r'Packet latency average = ([\d.]+)', output)]
    acc_matches = [m for m in re.finditer(r'Accepted flit rate average = ([\d.]+)', output)]
    hop_matches = [m for m in re.finditer(r'Hops average = ([\d.]+)', output)]
    inj_matches = [m for m in re.finditer(r'Injected flit rate average = ([\d.]+)', output)]
    if lat_matches: res['latency'] = float(lat_matches[-1].group(1))
    if acc_matches: res['accepted'] = float(acc_matches[-1].group(1))
    if hop_matches: res['hops'] = float(hop_matches[-1].group(1))
    if inj_matches: res['injected'] = float(inj_matches[-1].group(1))
    return res

def run_one(topo, k, n, algo, traffic, rate, seed, force=False):
    fname = f"topo={topo}_k={k}_n={n}_algo={algo}_traffic={traffic}_inj={rate}_seed={seed}"
    outpath = RESULTS / f"{fname}.txt"
    if outpath.exists() and not force:
        return parse_final(outpath.read_text())
    
    cfg = CFG_TEMPLATE.format(topo=topo, k=k, n=n, algo=algo, traffic=traffic, rate=rate, seed=seed)
    cfg_path = f"/tmp/bs2_{os.getpid()}_{int(time.time())}.cfg"
    with open(cfg_path, "w") as f: f.write(cfg)
    try:
        r = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=120)
        out = r.stdout + r.stderr
        outpath.write_text(out)
        return parse_final(out)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {fname}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR {fname}: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(cfg_path): os.remove(cfg_path)

def batch_run(later: str | None = None):
    """Run ALL experiments. If later is 'mesh4', only Mesh 4x4."""
    topos = [("mesh",4,4)]
    if not later:
        topos.extend([("mesh",8,8), ("torus",4,4)])
    
    traffics = ["uniform", "transpose", "hotspot"]
    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    seeds = list(range(5))
    bl = ["dor", "adaptive_xy_yx", "min_adapt", "valiant"]
    gnn = {"mesh": {"gnn_ppo_route_4x4": (4,4), "gnn_ppo_route_8x8": (8,8)}}
    
    all_results = []
    total, done, fail = 0, 0, 0
    
    for topo, k, n in topos:
        algos = bl.copy()
        if topo == "mesh":
            for aname, (ak, an) in gnn["mesh"].items():
                if k == ak and n == an:
                    algos.append(aname)
        for t in traffics:
            for rate in rates:
                for seed in seeds:
                    for algo in algos:
                        total += 1
                        fname = f"topo={topo}_k={k}_n={n}_algo={algo}_traffic={t}_inj={rate}_seed={seed}"
                        outpath = RESULTS / f"{fname}.txt"
                        
                        if outpath.exists():
                            parsed = parse_final(outpath.read_text())
                            done += 1
                        else:
                            parsed = run_one(topo, k, n, algo, t, rate, seed)
                            if parsed:
                                done += 1
                            else:
                                fail += 1
                        
                        if parsed:
                            all_results.append((topo, k, n, algo, t, rate, seed, parsed))
                        
                        if total % 50 == 0:
                            print(f"  [{total}] done={done} fail={fail}", flush=True)
    
    # Write CSV
    csv_path = RESULTS / "all_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topo","k","n","algo","traffic","inj_rate","seed","latency","accepted","hops"])
        for r in all_results:
            w.writerow([r[0],r[1],r[2],r[3],r[4],r[5],r[6],
                        r[7].get('latency',''), r[7].get('accepted',''), r[7].get('hops','')])
    
    print(f"\n{'='*60}")
    print(f"Total: {total}, Done: {done}, Failed: {fail}")
    print(f"CSV: {csv_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--later", choices=["mesh4"], help="Run just Mesh 4x4 first")
    p.add_argument("--algo", help="Specific algo only")
    p.add_argument("--check", action="store_true", help="Count only, don't run")
    args = p.parse_args()
    batch_run(later=args.later)
