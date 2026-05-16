#!/usr/bin/env python3
"""
Generate Optimal Port Labels for 8x8 Mesh Training
===================================================
For each (src, dst, traffic_pattern), compute:
- Which ports are minimal toward dst (max 2)
- If multiple minimal ports: pick the one leading to neighbor with 
  lower congestion (fewest active shortest paths or lower degree)
- Priority heuristic: prefer E/W (horizontal) over N/S (vertical)

Output: labels[64][64][4] — one-hot optimal port for each (cur,dst) pair
Also exports labeled training data for fine-tuning.

Author: Ngoc Anh for Thay Hieu
Date: 17/05/2026
"""

import numpy as np
import time, os, sys, json
from collections import defaultdict

# ============================================================
# MESH TOPOLOGY HELPERS
# ============================================================

def get_minimal_ports(cur, dst, G=8):
    """Return list of minimal ports for (cur, dst) on GxG mesh."""
    cx, cy = cur % G, cur // G
    dx, dy = dst % G, dst // G
    
    minimal = []
    if dx > cx: minimal.append(0)  # E
    if dx < cx: minimal.append(1)  # W
    if dy > cy: minimal.append(2)  # S
    if dy < cy: minimal.append(3)  # N
    
    return minimal


def get_next_node(cur, port, G=8):
    """Get neighbor node reached by taking port from cur."""
    x, y = cur % G, cur // G
    if port == 0:  # E
        return cur + 1 if x < G - 1 else -1
    elif port == 1:  # W
        return cur - 1 if x > 0 else -1
    elif port == 2:  # S
        return cur + G if y < G - 1 else -1
    elif port == 3:  # N
        return cur - G if y > 0 else -1
    return -1


def port_direction_str(port):
    return ['E', 'W', 'S', 'N'][port]


def get_node_degree(node, G=8):
    """Degree of a node in mesh (2, 3, or 4)."""
    x, y = node % G, node // G
    deg = 0
    if x > 0: deg += 1
    if x < G - 1: deg += 1
    if y > 0: deg += 1
    if y < G - 1: deg += 1
    return deg


# ============================================================
# TRAFFIC PATTERN GENERATORS
# ============================================================

def gen_uniform(N=64):
    """Uniform random traffic."""
    T = np.ones((N, N)) / (N - 1.0)
    np.fill_diagonal(T, 0)
    return T


def gen_transpose(G=8):
    """Transpose traffic: (x,y) -> (y,x)."""
    N = G * G
    T = np.zeros((N, N))
    for y in range(G):
        for x in range(G):
            s = y * G + x
            d = x * G + y
            if s != d:
                T[s, d] = 1.0 / (N - G)  # G nodes map to diagonal
    return T


def gen_hotspot(center=28, hot_rate=0.15, N=64):
    """Hotspot traffic: hot_rate fraction to one node."""
    T = np.ones((N, N)) * (1.0 - hot_rate) / (N - 2.0)
    T[:, center] = hot_rate
    np.fill_diagonal(T, 0)
    return T


def gen_bit_reversal(G=8):
    """Bit reversal traffic for G=8: (x,y) -> (G-1-x, G-1-y)."""
    N = G * G
    T = np.zeros((N, N))
    for y in range(G):
        for x in range(G):
            s = y * G + x
            d = (G - 1 - y) * G + (G - 1 - x)
            if s != d:
                T[s, d] = 1.0 / (N - 1.0)
    np.fill_diagonal(T, 0)
    return T


def gen_shuffle(G=8, seed=42):
    """Shuffle permutation traffic."""
    N = G * G
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    T = np.zeros((N, N))
    for s in range(N):
        d = perm[s]
        if s != d:
            T[s, d] = 1.0 / sum(1 for s2 in range(N) if perm[s2] != s2)
    return T


# ============================================================
# OPTIMAL PORT LABEL COMPUTATION
# ============================================================

def compute_port_loads(traffic_matrix, G=8, N=64):
    """
    Compute link load for each port under adaptive minimal routing.
    Uses congestion-aware path selection.
    
    Returns:
        L_adapt: load on each port at each node
    """
    L_adapt = np.zeros((N, 4))
    
    for src in range(N):
        for dst in range(N):
            if src == dst:
                continue
            rate = traffic_matrix[src, dst]
            if rate == 0:
                continue
            
            # Simulate adaptive minimal routing
            cur = src
            max_hops = 30
            hops = 0
            
            while cur != dst and hops < max_hops:
                hops += 1
                min_ports = get_minimal_ports(cur, dst, G)
                
                if not min_ports:
                    break
                
                # Pick port with lowest current load (congestion-aware)
                best_port = min(min_ports, key=lambda p: L_adapt[cur, p])
                L_adapt[cur, best_port] += rate
                cur = get_next_node(cur, best_port, G)
                if cur < 0:
                    break
    
    return L_adapt


def compute_optimal_labels_from_loads(L_adapt, G=8, N=64):
    """
    For each (cur, dst) pair, determine optimal port label.
    
    Strategy:
    1. Find minimal ports toward dst.
    2. If only one minimal port: that's optimal.
    3. If multiple (intersection/diagonal): pick the one leading to 
       a neighbor with lower load. If equal, prefer E/W (horizontal)
       first for deadlock freedom + path diversity.
    
    Returns:
        labels[cur][dst]: optimal port (0-3), or -1 if cur == dst
    """
    labels = np.full((N, N), -1, dtype=np.int32)
    
    for cur in range(N):
        for dst in range(N):
            if cur == dst:
                continue
            
            min_ports = get_minimal_ports(cur, dst, G)
            
            if len(min_ports) == 1:
                labels[cur, dst] = min_ports[0]
            elif len(min_ports) >= 2:
                # Find which port leads to lower loaded neighbor
                best_port = min_ports[0]
                best_load = L_adapt[get_next_node(cur, best_port, G) if get_next_node(cur, best_port, G) >= 0 else cur].sum()
                
                for p in min_ports[1:]:
                    neighbor = get_next_node(cur, p, G)
                    if neighbor < 0:
                        continue
                    neighbor_load = L_adapt[neighbor].sum()
                    
                    if neighbor_load < best_load:
                        best_load = neighbor_load
                        best_port = p
                    elif neighbor_load == best_load:
                        # Tiebreaker: prefer E/W (horizontal) over N/S (vertical)
                        # E(0) and W(1) have priority over S(2) and N(3)
                        if best_port in [2, 3] and p in [0, 1]:
                            best_port = p
                        # Among horizontals, prefer E(0) over W(1)
                        elif p == 0 and best_port == 1:
                            best_port = p
                        # Among verticals, prefer S(2) over N(3)
                        elif p == 2 and best_port == 3:
                            best_port = p
                
                labels[cur, dst] = best_port
            else:
                # Shouldn't happen for non-diagonal pairs
                labels[cur, dst] = 0  # default to E
    
    return labels


def compute_optimal_labels_for_dataset(traffic_patterns, G=8, N=64):
    """
    Compute optimal port labels for all traffic patterns.
    
    Returns:
        all_labels: dict {pattern_name: labels[64][64]} 
                    where each label is optimal port (0-3)
    """
    all_labels = {}
    
    for name, T in traffic_patterns:
        print(f"  Computing optimal labels for {name}...", end=" ")
        t0 = time.time()
        L_adapt = compute_port_loads(T, G, N)
        labels = compute_optimal_labels_from_loads(L_adapt, G, N)
        dt = time.time() - t0
        print(f"done ({dt:.1f}s)")
        all_labels[name] = labels
    
    return all_labels


# ============================================================
# COMPUTE GNN TARGET SCORES (SOFT LABELS)
# ============================================================

def compute_target_scores(traffic_matrix, G=8, N=64):
    """
    Compute soft target scores for each (cur, dst, port).
    
    Score for port p at (cur, dst):
    - If p is minimal: score = exp(-load/L_avg) normalized among minimal ports
    - If p is not minimal: score = 0 (or very small)
    - Normalized so sum over ports = 1
    
    This gives a probability distribution over ports for KL-div training.
    """
    L_adapt = compute_port_loads(traffic_matrix, G, N)
    
    # Compute average load across all ports
    avg_load = L_adapt[L_adapt > 0].mean() if L_adapt.sum() > 0 else 0.01
    
    target_scores = np.zeros((N, N, 4))
    
    for cur in range(N):
        for dst in range(N):
            if cur == dst:
                continue
            
            min_ports = get_minimal_ports(cur, dst, G)
            if not min_ports:
                continue
            
            # Compute scores for minimal ports
            scores = np.zeros(4)
            for p in min_ports:
                load = L_adapt[cur, p]
                # Higher load → lower score (congestion penalty)
                # Use exponential decay: score = exp(-load / avg_load)
                if avg_load > 0:
                    scores[p] = np.exp(-load / avg_load)
                else:
                    scores[p] = 1.0
            
            # Normalize to sum=1
            total = scores.sum()
            if total > 0:
                scores /= total
            else:
                # Equal distribution among minimal ports
                for p in min_ports:
                    scores[p] = 1.0 / len(min_ports)
            
            target_scores[cur, dst] = scores
    
    return target_scores, L_adapt


# ============================================================
# MAIN
# ============================================================

def main():
    G = 8
    N = G * G
    
    out_dir = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments'
    
    print("=" * 70)
    print("8x8 Training Data Generation")
    print("=" * 70)
    print(f"Mesh: {G}x{G}, Nodes: {N}")
    print()
    
    t0 = time.time()
    
    # 1. Generate traffic patterns for training
    print("[1/4] Generating traffic patterns...")
    traffic_patterns = [
        ("uniform", gen_uniform(N)),
        ("transpose", gen_transpose(G)),
        ("hotspot_c28", gen_hotspot(center=28, hot_rate=0.15, N=N)),  # center hotspot
        ("hotspot_c0", gen_hotspot(center=0, hot_rate=0.15, N=N)),    # corner hotspot
        ("bit_reversal", gen_bit_reversal(G)),
        ("shuffle", gen_shuffle(G, seed=42)),
        ("transpose2", gen_transpose(G) * 0.5 + gen_uniform(N) * 0.5),  # mixed
        ("hotspot_light", gen_hotspot(center=35, hot_rate=0.08, N=N)),  # 8,3*8+3=27... node 35 is slightly center
    ]
    print(f"  {len(traffic_patterns)} patterns generated")
    
    # 2. Compute optimal port labels + target scores
    print("\n[2/4] Computing optimal port labels for each pattern...")
    all_labels = compute_optimal_labels_for_dataset(traffic_patterns, G, N)
    
    # 3. Validate labels
    print("\n[3/4] Validating labels...")
    for name, labels in all_labels.items():
        valid = 0
        total = 0
        for cur in range(N):
            for dst in range(N):
                if cur == dst: continue
                total += 1
                min_ports = get_minimal_ports(cur, dst, G)
                if labels[cur, dst] in min_ports:
                    valid += 1
        print(f"  {name}: {valid}/{total} minimal port labels valid ({valid/total*100:.1f}%)")
    
    # 4. Save training data
    print("\n[4/4] Saving training data...")
    
    # Save as numpy dict
    train_data = {
        'G': G,
        'N': N,
        'labels': np.stack([all_labels[name] for name, _ in traffic_patterns], axis=0),
        'pattern_names': [name for name, _ in traffic_patterns],
    }
    
    # Also compute and save soft targets for fine-tuning
    target_data = {}
    for name, T in traffic_patterns:
        print(f"  Computing targets for {name}...", end=" ")
        t1 = time.time()
        targets, L_adapt = compute_target_scores(T, G, N)
        target_data[name] = targets
        print(f"done ({time.time()-t1:.1f}s)")
    
    # Save targets
    targets_array = np.stack([target_data[name] for name, _ in traffic_patterns], axis=0)
    np.save(f'{out_dir}/gnn_8x8_targets_v5.npy', targets_array)
    np.save(f'{out_dir}/gnn_8x8_labels_v5.npy', train_data['labels'])
    
    # Save metadata
    metadata = {
        'G': G,
        'N': N,
        'pattern_names': train_data['pattern_names'],
        'num_patterns': len(traffic_patterns),
        'total_pairs': N * (N - 1),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'target_shape': list(targets_array.shape),
    }
    with open(f'{out_dir}/gnn_8x8_training_metadata_v5.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    total_time = time.time() - t0
    print(f"\n  Targets shape: {targets_array.shape}")
    print(f"  Labels shape: {train_data['labels'].shape}")
    print(f"  Metadata saved to gnn_8x8_training_metadata_v5.json")
    print(f"\nTotal time: {total_time:.1f}s")
    print("8x8 Training data generation complete!")
    
    # Summary stats
    print(f"\n{'='*70}")
    print("Training Data Summary")
    print(f"{'='*70}")
    for name, labels in all_labels.items():
        port_dist = [np.sum(labels == p) / (N*(N-1)) * 100 for p in range(4)]
        print(f"  {name:20s}: E={port_dist[0]:.1f}% W={port_dist[1]:.1f}% S={port_dist[2]:.1f}% N={port_dist[3]:.1f}%")
    
    return train_data, target_data


if __name__ == '__main__':
    main()
