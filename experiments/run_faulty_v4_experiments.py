#!/usr/bin/env python3
"""Faulty Topology Experiments comparing GNN v4 vs DOR vs planar_adapt."""
import subprocess, os, csv, sys, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BOOKSIM = "/home/opc/.openclaw/workspace/booksim2/src/booksim"
RESULTS = Path("/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments/results_faulty_v4")
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
sample_period = 500;
sim_count = 50;
seed = {seed};
link_failures = {fails};
fail_seed = {fail_seed};
"""

def run_one(algo, traffic, rate, seed, fails, fail_seed):
    fname = f"algo={algo}_traffic={traffic}_inj={rate}_seed={seed}_fails={fails}_fseed={fail_seed}"
    outpath = RESULTS / f"{fname}.txt"
    if outpath.exists():
        return fname, "cached"
    cfg = CFG.format(algo=algo, traffic=traffic, rate=rate, seed=seed, fails=fails, fail_seed=fail_seed)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False, dir='/tmp') as f:
        f.write(cfg)
        cfg_path = f.name
    try:
        r = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=60)
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

def parse_csv():
    rows = []
    for f in sorted(RESULTS.glob("algo=*.txt")):
        base = f.stem
        text = f.read_text()
        
        lat_match = re.findall(r'Packet latency average = ([\d.]+)', text)
        acc_match = re.findall(r'Accepted flit rate average= ([\d.]+)', text)
        hop_match = re.findall(r'Hops average = ([\d.]+)', text)
        fail_match = re.findall(r'failure at node', text)
        
        if lat_match:
            parts = base.split("_")
            p = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    p[k] = v
            
            rows.append({
                'algo': p.get('algo', '?'), 'traffic': p.get('traffic', '?'),
                'inj_rate': float(p.get('inj', 0)), 'seed': int(p.get('seed', 0)),
                'link_failures': int(p.get('fails', 0)), 'fail_seed': int(p.get('fseed', 0)),
                'latency': float(lat_match[-1]), 'accepted': float(acc_match[-1]) if acc_match else 0,
                'hops': float(hop_match[-1]) if hop_match else 0,
                'n_actual_fails': len(fail_match),
                'unstable': 'unstable' in text
            })
        else:
            print(f"  SKIP (no data): {base}", file=sys.stderr)
    
    csv_path = RESULTS / "faulty_v4_results.csv"
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=['algo','traffic','inj_rate','seed','link_failures','fail_seed','latency','accepted','hops','n_actual_fails','unstable'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path} ({len(rows)} rows)")

def generate():
    algos = ['dor', 'planar_adapt', 'gnn_port_score_route_4x4_v4']
    traffics = ['uniform', 'transpose', 'hotspot']
    rates = [0.01, 0.05]
    seeds = [0, 1]
    fault_levels = [0, 2, 4, 7]
    fail_seeds = [101]
    for algo in algos:
        for traffic in traffics:
            for rate in rates:
                for seed in seeds:
                    for fails in fault_levels:
                        for fseed in fail_seeds:
                            yield (algo, traffic, rate, seed, fails, fseed)

def main():
    experiments = list(generate())
    total = len(experiments)
    existing = sum(1 for e in experiments if (RESULTS / f"algo={e[0]}_traffic={e[1]}_inj={e[2]}_seed={e[3]}_fails={e[4]}_fseed={e[5]}.txt").exists())
    
    print(f"Total: {total}, Cached: {existing}, Need: {total-existing}")
    print(f"  Algorithms: {len(set(e[0] for e in experiments))}")
    print(f"  Traffic: {len(set(e[1] for e in experiments))}")
    print(f"  Rates: {len(set(e[2] for e in experiments))}")
    print(f"  Seeds: {len(set(e[3] for e in experiments))}")
    print(f"  Fault levels: {len(set(e[4] for e in experiments))}")
    
    if total - existing == 0:
        parse_csv()
        return
    
    todo = [e for e in experiments if not (RESULTS / f"algo={e[0]}_traffic={e[1]}_inj={e[2]}_seed={e[3]}_fails={e[4]}_fseed={e[5]}.txt").exists()]
    
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=4) as pool:
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
                print(f"  [{i+1}/{len(todo)}] ok={ok_count} fail={fail_count} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()
    
    print(f"\nCompleted {ok_count} runs in {time.time()-t0:.0f}s, fail={fail_count}")
    parse_csv()

if __name__ == "__main__":
    main()
