#!/usr/bin/env python3
"""
Mesh 8x8 Injection Sweep — 3 algorithms × 3 traffics × 4 rates × 5 seeds = 180 experiments

Algorithms: gnn_port_score_route_8x8_v4, dor, planar_adapt
Traffic:    uniform, transpose, hotspot
Rates:      0.01, 0.05, 0.1, 0.2
Seeds:      0, 1, 2, 3, 4
"""

import subprocess, os, re, time, csv, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BOOKSIM = os.path.expanduser("~/.openclaw/workspace/booksim2/src/booksim")
RESULTS_DIR = os.path.expanduser("~/.openclaw/workspace/papers/gnocroute-research/experiments/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGOS = ["gnn_port_score_route_8x8_v4", "dor", "planar_adapt"]
TRAFFICS = ["uniform", "transpose", "hotspot"]
RATES = [0.01, 0.05, 0.1, 0.2]
SEEDS = list(range(5))

CFG_TEMPLATE = """topology = mesh;
k = 8; n = 2;
routing_function = {algo};
num_vcs = 4; vc_buf_size = 4;
wait_for_tail_credit = 1;
vc_allocator = islip; sw_allocator = islip;
alloc_iters = 1;
credit_delay = 2; routing_delay = 1;
vc_alloc_delay = 1; sw_alloc_delay = 1;
input_speedup = 2; output_speedup = 1; internal_speedup = 1.0;
traffic = {traffic};
packet_size = 1; sim_type = latency;
injection_rate = {rate};
warmup_periods = 3; sample_period = 1000; sim_count = 100;
seed = {seed};
"""

def filename(algo, traffic, rate, seed):
    """Generate the output filename consistently."""
    return f"topo=mesh_k=8_n=2_algo={algo}_traffic={traffic}_inj={rate}_seed={seed}.txt"

def need_experiment(algo, traffic, rate, seed):
    """Check if this experiment already exists with proper results."""
    fpath = os.path.join(RESULTS_DIR, filename(algo, traffic, rate, seed))
    if not os.path.exists(fpath):
        return True
    # Check it has the Overall Traffic Statistics line
    with open(fpath) as f:
        content = f.read()
    if "Overall Traffic Statistics" in content:
        return False
    return True

def run_one(algo, traffic, rate, seed):
    """Run a single BookSim2 experiment and return the result dict."""
    fpath = os.path.join(RESULTS_DIR, filename(algo, traffic, rate, seed))
    
    # Write config to temp file
    cfg = CFG_TEMPLATE.format(algo=algo, traffic=traffic, rate=rate, seed=seed)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        f.write(cfg)
        cfg_path = f.name
    
    try:
        result = subprocess.run(
            [BOOKSIM, cfg_path],
            capture_output=True, text=True, timeout=180,
            cwd=os.path.dirname(BOOKSIM)
        )
        output = result.stdout
        
        # Save output
        with open(fpath, 'w') as f:
            f.write(output)
        
        # Parse latency from "====== Overall Traffic Statistics =====" section
        lines = output.split('\n')
        overall_section = False
        packet_latency_avg = None
        
        for line in lines:
            if "Overall Traffic Statistics" in line:
                overall_section = True
            if overall_section and "Packet latency average" in line:
                # Parse: Packet latency average = 33.3939 (100 samples)
                m = re.search(r'=\s*([\d.]+)', line)
                if m:
                    packet_latency_avg = float(m.group(1))
                break
        
        if packet_latency_avg is None:
            # Fallback: try last occurrence
            for line in reversed(lines):
                if "Packet latency average" in line:
                    m = re.search(r'=\s*([\d.]+)', line)
                    if m:
                        packet_latency_avg = float(m.group(1))
                    break
        
        return {
            'algo': algo,
            'traffic': traffic,
            'rate': rate,
            'seed': seed,
            'latency': packet_latency_avg or -1,
            'success': packet_latency_avg is not None
        }
    except subprocess.TimeoutExpired:
        with open(fpath, 'w') as f:
            f.write("TIMEOUT\n")
        return {'algo': algo, 'traffic': traffic, 'rate': rate, 'seed': seed, 'latency': -1, 'success': False}
    except Exception as e:
        with open(fpath, 'w') as f:
            f.write(f"ERROR: {e}\n")
        return {'algo': algo, 'traffic': traffic, 'rate': rate, 'seed': seed, 'latency': -1, 'success': False}
    finally:
        try:
            os.unlink(cfg_path)
        except:
            pass

def main():
    # Build experiment list, skip existing
    experiments = []
    for algo in ALGOS:
        for traffic in TRAFFICS:
            for rate in RATES:
                for seed in SEEDS:
                    if need_experiment(algo, traffic, rate, seed):
                        experiments.append((algo, traffic, rate, seed))
    
    total = len(experiments)
    completed_total = 3 * 3 * 4 * 5 - total
    print(f"Total possible: 180 experiments")
    print(f"Already completed: {completed_total}")
    print(f"Need to run: {total}")
    print()
    
    if total == 0:
        print("All experiments already complete. Skipping to analysis.")
        return []
    
    # Run with thread pool
    results = []
    done_count = 0
    lock = threading.Lock()
    
    def run_with_progress(algo, traffic, rate, seed):
        nonlocal done_count
        result = run_one(algo, traffic, rate, seed)
        with lock:
            done_count += 1
            status = "OK" if result['success'] else "FAIL"
            print(f"[{done_count}/{total}] {algo[:6]}.. | {traffic:10} | rate={rate:.2f} | seed={seed} | latency={result['latency']:.2f} [{status}]")
            # Flush periodically
            if done_count % 10 == 0:
                print(f"--- Progress: {done_count}/{total} ({100*done_count/total:.0f}%) ---", flush=True)
        return result
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_with_progress, algo, traffic, rate, seed): (algo, traffic, rate, seed)
            for algo, traffic, rate, seed in experiments
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                algo, traffic, rate, seed = futures[future]
                print(f"ERROR: {algo} {traffic} rate={rate} seed={seed}: {e}")
                results.append({'algo': algo, 'traffic': traffic, 'rate': rate, 'seed': seed, 'latency': -1, 'success': False})
    
    return results

def analyze(all_results):
    """Aggregate results: mean ± std over seeds."""
    from statistics import mean, stdev
    
    print("\n" + "="*100)
    print("MESH 8x8 — LATENCY RESULTS (mean ± std, 5 seeds)")
    print("="*100)
    
    header = f"{'Algo':30s} {'Traffic':12s} {'Rate':6s} {'Latency':12s} {'Std':10s} {'Min Seed':10s} {'Max Seed':10s}"
    print(header)
    print("-"*100)
    
    # Group by algo, traffic, rate
    groups = {}
    for r in all_results:
        key = (r['algo'], r['traffic'], r['rate'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r['latency'])
    
    for (algo, traffic, rate), latencies in sorted(groups.items()):
        valid = [l for l in latencies if l > 0]
        if not valid:
            continue
        avg = mean(valid)
        sd = stdev(valid) if len(valid) > 1 else 0
        algo_short = algo[:28]
        print(f"{algo_short:30s} {traffic:12s} {rate:<6.2f} {avg:<12.2f} {sd:<10.2f} {min(valid):<10.2f} {max(valid):<10.2f}")
    
    print()
    
    # Also by traffic grouping
    for traffic in TRAFFICS:
        print(f"\n--- Traffic: {traffic} ---")
        print(f"{'Algo':30s} {'Rate':6s} {'Latency':12s} {'Std':10s}")
        print("-"*60)
        for algo in ALGOS:
            line = f"{algo[:28]:30s}"
            for rate in RATES:
                key = (algo, traffic, rate)
                if key in groups and groups[key]:
                    valid = [l for l in groups[key] if l > 0]
                    if valid:
                        avg = mean(valid)
                        sd = stdev(valid) if len(valid) > 1 else 0
                        line += f"  {rate:.2f}: {avg:.2f}±{sd:.2f}"
            print(line)
    
    # Write CSV
    csv_path = os.path.join(RESULTS_DIR, "all_results_8x8.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algo', 'traffic', 'rate', 'seed', 'latency'])
        for r in all_results:
            writer.writerow([r['algo'], r['traffic'], r['rate'], r['seed'], r['latency']])
    
    print(f"\nResults saved to: {csv_path}")

if __name__ == '__main__':
    results = main()
    if results:
        analyze(results)
    else:
        # All done — read existing and analyze
        print("Reading existing results for analysis...")
        # Load existing files
        all_results = []
        for algo in ALGOS:
            for traffic in TRAFFICS:
                for rate in RATES:
                    for seed in SEEDS:
                        fpath = os.path.join(RESULTS_DIR, filename(algo, traffic, rate, seed))
                        if os.path.exists(fpath):
                            with open(fpath) as f:
                                content = f.read()
                            lat = None
                            for line in content.split('\n'):
                                if "Overall Traffic Statistics" in line:
                                    break
                            # Find the last Packet latency average
                            lines = content.split('\n')
                            overall_seen = False
                            for line in lines:
                                if "Overall Traffic Statistics" in line:
                                    overall_seen = True
                                if overall_seen and "Packet latency average" in line:
                                    m = re.search(r'=\s*([\d.]+)', line)
                                    if m:
                                        lat = float(m.group(1))
                                    break
                            if lat is None:
                                for line in reversed(lines):
                                    if "Packet latency average" in line:
                                        m = re.search(r'=\s*([\d.]+)', line)
                                        if m:
                                            lat = float(m.group(1))
                                        break
                            all_results.append({
                                'algo': algo,
                                'traffic': traffic,
                                'rate': rate,
                                'seed': seed,
                                'latency': lat or -1
                            })
        analyze(all_results)
