#!/usr/bin/env python3
"""
Update routing table in BookSim2 header and recompile + test.
Usage: python3 update_and_test_routing_table.py <table_json> [label]

table_json: JSON file with 16x16 array of 0/1 values
label: optional label for reporting
"""

import sys, os, json, time, subprocess, tempfile, re
import numpy as np

BOOKSIM_DIR = '/home/opc/.openclaw/workspace/booksim2/src'
HEADER_4X4 = os.path.join(BOOKSIM_DIR, 'gnn_ppo_route_4x4.h')
BOOKSIM = os.path.join(BOOKSIM_DIR, 'booksim')


def update_header(table, header_path):
    """Update routing table in header file."""
    N = table.shape[0]
    
    with open(header_path, 'r') as f:
        content = f.read()
    
    # Build table string
    rows = []
    for i in range(N):
        vals = ",".join(str(int(round(v))) for v in table[i])
        comma = "," if i < N-1 else " "
        rows.append(f"  {{{vals}}}{comma}")
    
    table_str = "static const int gnn_route_table_4x4[{}][{}]={{\n".format(N, N)
    table_str += "\n".join(rows)
    table_str += "\n};"
    
    # Replace old table
    import re
    pattern = r'static const int gnn_route_table_4x4\[\d+\]\[\d+\]=\{[^}]*\};'
    new_content = re.sub(pattern, table_str, content, flags=re.DOTALL)
    
    with open(header_path, 'w') as f:
        f.write(new_content)
    
    print(f"[UPDATE] Header updated: {header_path}")


def recompile():
    """Recompile BookSim2."""
    print(f"[BUILD] Recompiling BookSim2...")
    result = subprocess.run(
        ['make', '-C', BOOKSIM_DIR, 'clean', 'all'],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"[BUILD] FAILED:\n{result.stderr}")
        return False
    print(f"[BUILD] OK")
    return True


def run_booksim_test(traffic='uniform', inj_rate=0.1):
    """Run single BookSim2 simulation and return latency."""
    cfg = f"""topology = mesh;
k = 4;
n = 2;
routing_function = gnn_ppo_route_4x4;
traffic = {traffic};
injection_rate = {inj_rate};
warmup_periods = 500;
sample_period = 10000;
sim_count = 5;
sim_type = latency;
num_vcs = 4;
vc_buf_size = 8;
"""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    f.write(cfg)
    cfg_path = f.name
    f.close()
    
    try:
        out = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=30)
        stdout = out.stdout
        lat = re.search(r'Packet latency average\s*=\s*([0-9.]+)', stdout)
        result = float(lat.group(1)) if lat else None
        os.unlink(cfg_path)
        return result
    except:
        os.unlink(cfg_path)
        return None


def benchmark_table(label="table"):
    """Run comprehensive benchmark on current binary."""
    tests = [
        ('uniform', 0.05), ('uniform', 0.10), ('uniform', 0.20), ('uniform', 0.30),
        ('transpose', 0.05), ('transpose', 0.10), ('transpose', 0.20), ('transpose', 0.30),
        ('hotspot', 0.05), ('hotspot', 0.10), ('hotspot', 0.20),
    ]
    
    results = {}
    print(f"\n{'='*50}")
    print(f"Benchmark: {label}")
    print(f"{'='*50}")
    
    for traffic, inj in tests:
        lat = run_booksim_test(traffic, inj)
        key = f"{traffic}_{inj:.2f}"
        results[key] = lat
        status = f"{lat:.1f} cycles" if lat else "SAT"
        print(f"  {traffic:12s} @{inj:.2f}: {status}")
    
    return results


# ============================================================
def handcrafted_optimal_table():
    """Create a hand-optimized routing table based on mesh analysis.
    
    Strategy:
    - For transpose (i,j)→(j,i): XY if i<j, YX if i>j (avoids diagonal)
    - For hotspot (to center node 10): prefer XY to minimize vertical contention
    - For uniform: any is fine
    
    Compromise: XY-heavy with selective YX for transpose pairs that benefit
    """
    G = 4
    table = np.zeros((16, 16), dtype=np.int32)
    
    for src in range(16):
        for dst in range(16):
            if src == dst:
                table[src, dst] = 0  # Self: XY (irrelevant)
                continue
            
            sx, sy = src % G, src // G
            dx, dy = dst % G, dst // G
            
            # Transpose pair: (i,j)→(j,i)
            is_transpose_pair = (sx == dy and sy == dx)
            
            if is_transpose_pair:
                # For transpose: choose direction that gets away from diagonal first
                if sx < sy:
                    table[src, dst] = 0  # XY: horizontal first (away from congested column)
                else:
                    table[src, dst] = 1  # YX: vertical first (away from congested row)
            else:
                # Default: prefer XY (good for hotspot)
                table[src, dst] = 0
    
    return table


# ============================================================
def gnn_generated_table(train_epochs=300):
    """Train GNN and return the generated table."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'train_routing_table',
        '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/code/train_routing_table.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"[GNN] Training for {train_epochs} epochs...")
    t0 = time.time()
    table, models = mod.train_routing_table(epochs=train_epochs)
    elapsed = time.time() - t0
    print(f"[GNN] Training done in {elapsed:.1f}s")
    return table


# ============================================================
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'handcrafted'
    
    if mode == 'handcrafted':
        table = handcrafted_optimal_table()
        print("\nHand-optimized routing table:")
        for i in range(16):
            row = " ".join(str(int(v)) for v in table[i])
            print(f"  {i:2d}: {row}")
        
        update_header(table, HEADER_4X4)
        if recompile():
            benchmark_table("Hand-optimized")
    
    elif mode == 'gnn':
        table = gnn_generated_table(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
        print(f"\nGNN-generated routing table ({table.mean()*100:.1f}% YX):")
        for i in range(16):
            row = " ".join(str(int(v)) for v in table[i])
            print(f"  {i:2d}: {row}")
        
        update_header(table, HEADER_4X4)
        if recompile():
            benchmark_table("GNN-generated")
    
    elif mode == 'all_yx':
        table = np.ones((16, 16), dtype=np.int32)
        np.fill_diagonal(table, 0)
        update_header(table, HEADER_4X4)
        if recompile():
            benchmark_table("Always YX")
    
    elif mode == 'bench_current':
        # Just benchmark the current compiled binary
        benchmark_table("Current binary")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python3 update_and_test_routing_table.py [handcrafted|gnn|all_yx|bench_current]")
