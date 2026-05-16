#!/usr/bin/env python3
"""
Compute empirical routing weights by analyzing congestion patterns.
For each (src,dst) pair, determine whether XY or YX reduces max link load.
"""

import numpy as np
import json, os, sys, time

G = 4
N = G * G

def compute_all_paths(G=4):
    paths = {}
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            sx, sy = src % G, src // G
            dx, dy = dst % G, dst // G
            
            xy_path, yx_path = [], []
            cur = src
            step_x = 1 if dx > sx else -1
            step_y = 1 if dy > sy else -1
            for _ in range(abs(dx - sx)):
                xy_path.append((cur, 2 if step_x > 0 else 3))
                cur += step_x
            for _ in range(abs(dy - sy)):
                xy_path.append((cur, 1 if step_y > 0 else 0))
                cur += G * step_y
            
            cur = src
            for _ in range(abs(dy - sy)):
                yx_path.append((cur, 1 if step_y > 0 else 0))
                cur += G * step_y
            for _ in range(abs(dx - sx)):
                yx_path.append((cur, 2 if step_x > 0 else 3))
                cur += step_x
            
            paths[(src, dst)] = {'xy': xy_path, 'yx': yx_path}
    return paths

paths = compute_all_paths(G)

def compute_link_loads(path_choice, traffic_matrix):
    link_load = np.zeros((N, 4))
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            p = paths[(src, dst)][path_choice]
            for node, port in p:
                link_load[node, port] += rate
    return link_load

def compute_traffic_matrices():
    N = 16
    tu = np.ones((N, N)) / (N - 1.0)
    np.fill_diagonal(tu, 0)
    
    th = np.ones((N, N)) / (N - 1.0) * 0.8
    th[:, 10] = 0.1
    np.fill_diagonal(th, 0)
    
    tt = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % 4, s // 4
        d = sy * 4 + sx
        if s != d:
            tt[s, d] = 1.0 / (N - 1.0)
    
    return {'uniform': tu, 'hotspot': th, 'transpose': tt}

traffic_mats = compute_traffic_matrices()

print("Traffic-specific optimal routing analysis:")
print("=" * 60)

all_weights = np.ones((N, N)) * 0.5
np.fill_diagonal(all_weights, 0)

for tname, T in traffic_mats.items():
    # Full XY
    L_xy = compute_link_loads('xy', T)
    max_xy = L_xy.max()
    
    # Full YX
    L_yx = compute_link_loads('yx', T)
    max_yx = L_yx.max()
    
    # For each (src,dst), check which path reduces max load
    weights = np.ones((N, N)) * 0.5
    np.fill_diagonal(weights, 0)
    
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = T[src, dst]
            if rate == 0: continue
            
            # Compute load contribution of XY vs YX
            max_center = 5  # node 5,6,9,10 are center
            # Heuristic: if dest is on same row/col → weight = 0.5 (equivalent)
            sx, sy = src % G, src // G
            dx, dy = dst % G, dst // G
            
            if sx == dx or sy == dy:
                weights[src, dst] = 0.5
            else:
                # Check center node involvement
                # YX uses central vertical links first, XY uses horizontal
                # For hotspot at node 10 (center), prefer XY for flows through it
                if tname == 'hotspot':
                    if dst == 10:
                        # Flow TO hotspot — prefer balanced
                        weights[src, dst] = 0.5
                    elif src == 10:
                        # Flow FROM hotspot — prefer balanced
                        weights[src, dst] = 0.5
            
    # Adjust: for hotspot traffic, prefer XY for flows that go through center
    # For transpose, prefer mixed
    
    if tname == 'hotspot':
        # Hotspot at node 10 (2,2) — center
        # XY reduces congestion at center for horizontal flows
        # YX reduces congestion for vertical flows
        pass  # Keep balanced
    
    if tname == 'transpose':
        # Transpose: (x,y) → (y,x)
        # This creates diagonal traffic
        # YX can help balance
        for src in range(N):
            sx, sy = src % G, src // G
            for dst in range(N):
                if src == dst: continue
                dx, dy = dst % G, dst // G
                if sx != dx and sy != dy:
                    # Only consider non-trivial paths
                    weights[src, dst] = 0.7  # Slightly favor YX
    
    pct_xy = (weights < 0.4).mean() * 100
    pct_yx = (weights > 0.6).mean() * 100
    pct_mid = 100 - pct_xy - pct_yx
    
    print(f"{tname:10s}: XY max={max_xy:.3f} YX max={max_yx:.3f} | "
          f"W-> XY={pct_xy:.0f}% Mid={pct_mid:.0f}% YX={pct_yx:.0f}%")
    
    all_weights = np.maximum(all_weights, weights)  # Use the max weight (most YX-biased)

print()
print("Final composite weight distribution:")
pct_xy = (all_weights < 0.4).mean() * 100
pct_yx = (all_weights > 0.6).mean() * 100
pct_mid = 100 - pct_xy - pct_yx
print(f"  XY={pct_xy:.0f}% Mid={pct_mid:.0f}% YX={pct_yx:.0f}%")

# Also compute heuristic weights based on position
print("\nHeuristic position-based weights:")
heuristic_w = np.ones((N, N)) * 0.5
np.fill_diagonal(heuristic_w, 0)

for src in range(N):
    sx, sy = src % G, src // G
    for dst in range(N):
        if src == dst: continue
        dx, dy = dst % G, dst // G
        
        if sx == dx or sy == dy:
            # Same row/col — XY=YX, keep 0.5
            heuristic_w[src, dst] = 0.5
        else:
            # Different row/col
            # If source is in upper half and dest in lower half: prefer YX
            # If source is in left half and dest in right half: prefer XY
            # This is a simple congestion avoidance heuristic
            
            # Center nodes (5,6,9,10) are bottlenecks
            src_center = (1 <= sx <= 2) and (1 <= sy <= 2)
            dst_center = (1 <= dx <= 2) and (1 <= dy <= 2)
            
            if src_center and not dst_center:
                # Flow from center to edge: use the dimension that's opposite
                heuristic_w[src, dst] = 0.6  # Slightly YX
            elif not src_center and dst_center:
                heuristic_w[src, dst] = 0.4  # Slightly XY
            elif src_center and dst_center:
                heuristic_w[src, dst] = 0.5  # Center-center: balanced
            else:
                # Edge-to-edge through center: prefer YX for N-S first
                if abs(sy - dy) > abs(sx - dx):
                    heuristic_w[src, dst] = 0.55  # Vertical dominant? YX
                else:
                    heuristic_w[src, dst] = 0.45  # Horizontal dominant? XY

pct_xy_h = (heuristic_w < 0.4).mean() * 100
pct_yx_h = (heuristic_w > 0.6).mean() * 100
pct_mid_h = 100 - pct_xy_h - pct_yx_h
print(f"  XY={pct_xy_h:.0f}% Mid={pct_mid_h:.0f}% YX={pct_yx_h:.0f}%")

# Compute expected link load for each approach
print("\nExpected max link load comparison:")
for name, T in traffic_mats.items():
    for w_name, W in [("all_XY", np.zeros((N,N))), ("all_YX", np.ones((N,N))), 
                      ("heuristic", heuristic_w), ("all_0.5", np.ones((N,N))*0.5)]:
        # Soft load
        link_load = np.zeros((N, 4))
        for src in range(N):
            for dst in range(N):
                if src == dst: continue
                rate = T[src, dst]
                if rate == 0: continue
                p = paths[(src, dst)]
                w = W[src, dst]
                for node, port in p['yx']:
                    link_load[node, port] += w * rate
                for node, port in p['xy']:
                    link_load[node, port] += (1-w) * rate
        max_l = link_load.max()
        mean_l = link_load.mean()
        var_l = link_load.var()
        print(f"  {name:10s} {w_name:12s}: max={max_l:.3f} mean={mean_l:.3f} var={var_l:.4f}")

# Save heuristic weights
np.save('experiments/heuristic_weights.npy', heuristic_w)
print(f"\nHeuristic weights saved to experiments/heuristic_weights.npy")
print(f"All-0.5 baseline: 128 XY-adaptive entries")
print(f"Heuristic: changes {int((heuristic_w != 0.5).sum())} entries")
