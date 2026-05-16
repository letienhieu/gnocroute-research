#!/usr/bin/env python3
"""
Benchmark Mesh 8x8: GNN-Weighted vs. Baselines
================================================
Algos: dor, adaptive_xy_yx, min_adapt, gnn_weighted_route_8x8
Traffics: uniform, transpose, hotspot
Rates: 0.01, 0.05, 0.1, 0.2, 0.3, 0.5
Seeds: 42, 43, 44

Uses config FILE (not stdin) for BookSim2.
"""
import subprocess, os, csv, time, json, sys

BOOKSIM = '/home/opc/.openclaw/workspace/booksim2/src/booksim'
EXP_DIR = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments'
OUTPUT_DIR = f'{EXP_DIR}/results_8x8'
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGOS = ['dor', 'adaptive_xy_yx', 'min_adapt', 'gnn_weighted_route_8x8']
TRAFFICS = ['uniform', 'transpose', 'hotspot']
RATES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
SEEDS = [42, 43, 44]

def make_config(algo, traffic, rate, seed):
    if traffic == 'hotspot':
        return f"""topology = mesh;
k = 8;
n = 2;
routing_function = {algo};
num_vcs = 4;
vc_buf_size = 4;
wait_for_tail_credit = 1;
traffic = hotspot(35,10);
packet_size = 1;
sim_type = latency;
injection_rate = {rate};
warmup_periods = 3;
sample_period = 1000;
sim_count = 1;
seed = {seed};
max_samples = 50;
"""
    return f"""topology = mesh;
k = 8;
n = 2;
routing_function = {algo};
num_vcs = 4;
vc_buf_size = 4;
wait_for_tail_credit = 1;
traffic = {traffic};
packet_size = 1;
sim_type = latency;
injection_rate = {rate};
warmup_periods = 3;
sample_period = 1000;
sim_count = 1;
seed = {seed};
max_samples = 50;
"""

def run_sim(algo, traffic, rate, seed, timeout=600):
    cfg = make_config(algo, traffic, rate, seed)
    tf = f'/tmp/bs_{algo}_{traffic}_r{rate}_s{seed}.cfg'
    with open(tf, 'w') as f:
        f.write(cfg)
    try:
        r = subprocess.run([BOOKSIM, tf],
                          capture_output=True, timeout=timeout)
        out = r.stdout.decode()
        lat = None
        for line in out.split('\n'):
            if 'Packet latency average' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    val_str = parts[-1].strip()
                    # Remove trailing (N samples) if present
                    if '(' in val_str:
                        val_str = val_str.split('(')[0].strip()
                    try:
                        lat = float(val_str)
                    except:
                        pass
        try: os.remove(tf)
        except: pass
        return lat, out, r.stderr.decode()
    except subprocess.TimeoutExpired:
        return None, '', 'TIMEOUT'
    except Exception as e:
        try: os.remove(tf)
        except: pass
        return None, '', str(e)

def main():
    all_results = []
    total = len(ALGOS) * len(TRAFFICS) * len(RATES) * len(SEEDS)
    done = 0
    failed = 0

    print(f"Mesh 8x8 Benchmark: {total} configs\n")

    for algo in ALGOS:
        for traffic in TRAFFICS:
            for rate in RATES:
                for seed in SEEDS:
                    done += 1
                    key = f"{algo}_{traffic}_r{rate}_s{seed}"
                    outfile = f"{OUTPUT_DIR}/{key}.txt"

                    # Check cache
                    if os.path.exists(outfile) and os.path.getsize(outfile) > 10:
                        with open(outfile) as f:
                            c = f.read()
                        for line in c.split('\n'):
                            if 'Latency:' in line:
                                try:
                                    lat = float(line.split(':')[1].strip())
                                    all_results.append({
                                        'algorithm': algo, 'traffic': traffic,
                                        'rate': rate, 'seed': seed, 'latency': lat
                                    })
                                    print(f"[{done}/{total}] {key}: {lat:.1f} (cached)")
                                except: pass
                        # If we found cached result, skip running
                        if len(all_results) > 0 and all_results[-1].get('algorithm') == algo \
                           and all_results[-1].get('rate') == rate \
                           and all_results[-1].get('seed') == seed:
                            continue

                    print(f"[{done}/{total}] {key}... ", end='', flush=True)
                    t0 = time.time()
                    lat, stdout, stderr = run_sim(algo, traffic, rate, seed)
                    elapsed = time.time() - t0

                    with open(outfile, 'w') as f:
                        f.write(f"Algorithm: {algo}\nTraffic: {traffic}\nRate: {rate}\nSeed: {seed}\nLatency: {lat}\nElapsed: {elapsed:.0f}s\n\n{stdout}\n")
                        if stderr:
                            f.write(f"\nSTDERR:\n{stderr}\n")

                    if lat is not None:
                        print(f"{lat:.1f} ({elapsed:.0f}s)")
                    else:
                        print(f"FAIL ({elapsed:.0f}s)")
                        failed += 1

                    all_results.append({
                        'algorithm': algo, 'traffic': traffic,
                        'rate': rate, 'seed': seed, 'latency': lat
                    })

    # Save CSV
    csv_path = f'{EXP_DIR}/scalability_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['algorithm', 'traffic', 'rate', 'seed', 'latency'])
        w.writeheader()
        w.writerows(all_results)

    # Summary
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY - Mesh 8x8")
    print(f"File: {csv_path}")
    print(f"{'='*80}")
    print(f"{'Algo':20s} {'Traffic':12s}", end='')
    for r in RATES:
        print(f" {'r='+str(r):>10s}", end='')
    print()
    print('-' * 80)
    for algo in ALGOS:
        for traffic in TRAFFICS:
            print(f"{algo:20s} {traffic:12s}", end='')
            for rate in RATES:
                vals = [r['latency'] for r in all_results
                        if r['algorithm'] == algo and r['traffic'] == traffic
                        and r['rate'] == rate and r['latency'] is not None]
                if vals:
                    avg = sum(vals) / len(vals)
                    print(f" {avg:>10.1f}", end='')
                else:
                    print(f" {'FAIL':>10s}", end='')
            print()
    print()

    # Save JSON
    with open(f'{EXP_DIR}/scalability_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Total configs: {done}")
    print(f"Valid results: {sum(1 for r in all_results if r['latency'] is not None)}")
    print(f"Failed: {failed}")

if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"Total time: {time.time()-t0:.0f}s")
