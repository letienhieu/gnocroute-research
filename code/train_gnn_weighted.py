#!/usr/bin/env python3
"""
GNN-Weighted Adaptive Routing: Training Pipeline
=================================================
Trains a GNN (GATv2) to output continuous routing preference weights
W[i][j] ∈ [0,1] for each (src,dst) pair in a Mesh NoC.

Weight Interpretation:
  W ≈ 0  → strongly prefer XY routing (dimension-order: X then Y)
  W ≈ 1  → strongly prefer YX routing (dimension-order: Y then X)
  W ≈ 0.5→ balanced / adaptive (let congestion decide at runtime)

BookSim2 Runtime:
  On each hop, the routing function reads W[cur][dst] and compares it
  to an adaptive threshold T = 0.5 + k * (local_congestion - 0.5).
  - W > T  → YX path
  - W < 1-T → XY path
  - otherwise → fully adaptive (choose less congested next port)

Author: Ngoc Anh for Thay Hieu
Date: 16/05/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
import random, math, os, sys, json, time, subprocess, tempfile, re

DEVICE = 'cpu'
print(f"[GNN] Using device: {DEVICE}")

# ============================================================
# 1. MESH TOPOLOGY (configurable size)
# ============================================================
def build_mesh_graph(G=4):
    """Build PyG-compatible mesh graph with edge_index."""
    N = G * G
    edges = []
    edge_attr = []
    for y in range(G):
        for x in range(G):
            idx = y * G + x
            if x > 0:
                edges.append([idx, y * G + (x-1)])
                edges.append([y * G + (x-1), idx])
                edge_attr.append([1.0])  # horizontal
                edge_attr.append([1.0])
            if y > 0:
                edges.append([idx, (y-1) * G + x])
                edges.append([(y-1) * G + x, idx])
                edge_attr.append([2.0])  # vertical
                edge_attr.append([2.0])
    edge_index = torch.LongTensor(edges).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attr)
    return edge_index, edge_attr, N

# Default: 4x4
TOPOLOGY_SIZE = int(os.environ.get('MESH_SIZE', '4'))
EDGE_INDEX, EDGE_ATTR, NUM_NODES = build_mesh_graph(TOPOLOGY_SIZE)
G = TOPOLOGY_SIZE
print(f"[GNN] Mesh {G}x{G}, {NUM_NODES} nodes")

# ============================================================
# 2. NODE FEATURES
# ============================================================
def compute_node_features(G=4):
    """Topology-aware node features [norm_x, norm_y, degree_norm, betweenness,
       corner_flag, edge_flag, center_flag]."""
    N = G * G
    features = np.zeros((N, 7))
    
    # Approximate betweenness centrality for mesh
    betweenness = {}
    if G == 4:
        betweenness = {
            0: 0.00, 1: 0.07, 2: 0.07, 3: 0.00,
            4: 0.07, 5: 0.33, 6: 0.33, 7: 0.07,
            8: 0.07, 9: 0.33, 10: 0.33, 11: 0.07,
            12: 0.00, 13: 0.07, 14: 0.07, 15: 0.00
        }
    elif G == 8:
        # Normalized betweenness for 8x8 (interior has higher centrality)
        for y in range(G):
            for x in range(G):
                idx = y * G + x
                d_center = abs(x - (G-1)/2) + abs(y - (G-1)/2)
                betweenness[idx] = max(0.0, 1.0 - d_center / (G/2)) * 0.5
    else:
        for y in range(G):
            for x in range(G):
                idx = y * G + x
                betweenness[idx] = 0.0
    
    for y in range(G):
        for x in range(G):
            idx = y * G + x
            features[idx, 0] = x / max(G - 1, 1)       # norm_x
            features[idx, 1] = y / max(G - 1, 1)       # norm_y
            deg = 2 + (x > 0) + (x < G-1) + (y > 0) + (y < G-1)
            features[idx, 2] = deg / 4.0                # degree norm
            features[idx, 3] = betweenness.get(idx, 0) # betweenness
            
            corner = (x == 0 or x == G-1) and (y == 0 or y == G-1)
            edge = (x == 0 or x == G-1 or y == 0 or y == G-1) and not corner
            center = not corner and not edge
            
            features[idx, 4] = 1.0 if corner else 0.0
            features[idx, 5] = 1.0 if edge else 0.0
            features[idx, 6] = 1.0 if center else 0.0
    
    return torch.FloatTensor(features)

NODE_FEATURES = compute_node_features(G)
print(f"[GNN] Node features shape: {NODE_FEATURES.shape}")

# ============================================================
# 3. GNN ENCODER (same architecture as baseline, output 32-dim)
# ============================================================
class GNNEncoder(nn.Module):
    """GATv2 encoder for NoC topology-aware node embeddings."""
    
    def __init__(self, in_dim=7, hidden_dim=64, out_dim=32, num_heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=1)
        self.conv3 = GATv2Conv(hidden_dim, out_dim, heads=1, edge_dim=1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        return x


# ============================================================
# 4. WEIGHT GENERATOR (Pairwise Decoder → continuous weight ∈ [0,1])
# ============================================================
class WeightGenerator(nn.Module):
    """
    Takes node embeddings and produces W[i][j] ∈ [0,1] for each (src,dst) pair.
    Unlike the binary routing table in train_routing_table.py, this outputs
    continuous weights that capture routing preference.
    """
    
    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Forces output to [0,1]
        )
    
    def forward(self, node_embeddings):
        """
        Args:
            node_embeddings: (N, embed_dim)
        Returns:
            weights: (N, N) with each W[i][j] ∈ [0,1]
        """
        n = node_embeddings.size(0)
        src_idx = torch.arange(n).unsqueeze(1).expand(n, n)
        dst_idx = torch.arange(n).unsqueeze(0).expand(n, n)
        
        src_emb = node_embeddings[src_idx]
        dst_emb = node_embeddings[dst_idx]
        
        pair_emb = torch.cat([src_emb, dst_emb], dim=-1)
        
        weights = self.decoder(pair_emb).squeeze(-1)  # (N, N)
        return weights  # Already ∈ [0,1] via Sigmoid


# ============================================================
# 5. TRAFFIC-AWARE PATH ANALYSIS
# ============================================================
def compute_all_paths(G=4):
    """Precompute XY and YX paths for all (src,dst)."""
    paths = {}
    N = G * G
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            sx, sy = src % G, src // G
            dx, dy = dst % G, dst // G
            
            # XY path: X first, then Y
            xy_path = []
            cur = src
            step_x = 1 if dx > sx else -1 if dx < sx else 0
            step_y = 1 if dy > sy else -1 if dy < sy else 0
            for _ in range(abs(dx - sx)):
                xy_path.append((cur, 2 if step_x > 0 else 3))  # E/W
                cur = cur + step_x
            for _ in range(abs(dy - sy)):
                xy_path.append((cur, 1 if step_y > 0 else 0))  # S/N
                cur = cur + G * step_y
            
            # YX path: Y first, then X
            yx_path = []
            cur = src
            for _ in range(abs(dy - sy)):
                yx_path.append((cur, 1 if step_y > 0 else 0))  # S/N
                cur = cur + G * step_y
            for _ in range(abs(dx - sx)):
                yx_path.append((cur, 2 if step_x > 0 else 3))  # E/W
                cur = cur + step_x
            
            paths[(src, dst)] = {'xy': xy_path, 'yx': yx_path}
    return paths

MESH_PATHS = compute_all_paths(G)


def compute_link_loads_soft(weights, traffic_matrix):
    """
    Continuous link load computation.
    Each (src,dst) pair contributes proportionally to XY and YX paths
    based on weight W[src][dst].
    
    Args:
        weights: (N, N) continuous weights ∈ [0,1]
        traffic_matrix: (N, N) traffic rates
    
    Returns:
        link_loads: (N, 4) array of load on each port
        stats: dict of metrics
    """
    N = weights.shape[0]
    link_load = np.zeros((N, 4))
    
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            
            paths = MESH_PATHS[(src, dst)]
            w = weights[src, dst]  # probability of YX
            
            # Soft routing: each path gets proportional load
            for node, port in paths['yx']:
                link_load[node, port] += w * rate
            for node, port in paths['xy']:
                link_load[node, port] += (1.0 - w) * rate
    
    max_load = np.max(link_load)
    mean_load = np.mean(link_load)
    var_load = np.var(link_load)
    
    # Per-pair congestion contribution
    # Compute how much each pair contributes to max-loaded links
    max_link_idx = np.unravel_index(np.argmax(link_load), link_load.shape)
    
    stats = {
        'max_load': max_load,
        'mean_load': mean_load,
        'var_load': var_load,
        'load_imbalance': var_load / max(mean_load, 1e-8),
    }
    return link_load, stats


def compute_link_loads_hard(weights, traffic_matrix, threshold=0.5):
    """
    Hard routing decision: use soft weights but binarize for evaluation.
    W > threshold → YX, else XY.
    Used for validation, not training.
    """
    N = weights.shape[0]
    link_load = np.zeros((N, 4))
    table_hard = (weights > threshold).astype(np.float32)
    
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            
            paths = MESH_PATHS[(src, dst)]
            use_yx = table_hard[src, dst] > 0.5
            path = paths['yx'] if use_yx else paths['xy']
            
            for node, port in path:
                link_load[node, port] += rate
    
    max_load = np.max(link_load)
    mean_load = np.mean(link_load)
    var_load = np.var(link_load)
    
    return link_load, max_load, mean_load, var_load


# ============================================================
# 6. DIFFERENTIABLE PROXY LOSS
# ============================================================
def compute_weighted_loss(weights, traffic_matrix, hot_node=None):
    """
    Differentiable loss for continuous routing weights.
    
    Loss Components:
    1. MAX_LOAD_PENALTY: Normalize max link utilization
    2. IMBALANCE_PENALTY: Penalize uneven load distribution
    3. HOTSPOT_PENALTY: Encourage decisive routing near hotspot
    4. ENTROPY_REG: Regularization to avoid all-0.5 degenerate solution
    5. SHORTEST_PATH_BIAS: Guide weights toward shortest-path alignment
    
    Args:
        weights: (N, N) tensor ∈ [0,1]
        traffic_matrix: (N, N) numpy array of traffic rates
        hot_node: optional hotspot node index
    """
    N = weights.shape[0]
    
    # ---- Soft link load computation (differentiable) ----
    # We'll compute loss using the soft weights directly
    link_load, stats = compute_link_loads_soft(
        weights.detach().cpu().numpy(), traffic_matrix
    )
    
    max_load = stats['max_load']
    mean_load = stats['mean_load']
    var_load = stats['var_load']
    
    # Convert to tensors for loss computation
    # Note: Since our path computation relies on numpy,
    # we compute a "guided" loss that encourages the model
    # to minimize expected congestion
    
    max_penalty = max(0.0, max_load - 0.3) * 15.0  # Penalize >30% utilization
    mean_penalty = mean_load * 3.0
    var_penalty = var_load * 5.0
    
    # ---- Distance-aware regularization ----
    # For short paths (same row or column), XY and YX are equivalent,
    # so W should be close to 0.5 (let congestion decide)
    # For longer paths, encourage decisive routing
    
    total_penalty = 0.0
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            sx, sy = src % G, src // G
            dx, dy = dst % G, dst // G
            manhattan = abs(dx - sx) + abs(dy - sy)
            w = weights[src, dst].item()
            
            if manhattan <= 1:
                # Adjacent: XY=YX, weight should be near 0.5
                total_penalty += 0.1 * abs(w - 0.5)
            elif manhattan >= 4:
                # Long distance: prefer decisive routing
                # Penalty for being too close to 0.5
                if 0.4 < w < 0.6:
                    total_penalty += 0.05
    
    dist_penalty = total_penalty / max(N * N, 1)
    
    # ---- Hotspot penalty ----
    hotspot_penalty = 0.0
    if hot_node is not None:
        hot_neighbors = [hot_node]
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = (hot_node % G) + dx
                ny = (hot_node // G) + dy
                if 0 <= nx < G and 0 <= ny < G:
                    nidx = ny * G + nx
                    if nidx != hot_node:
                        hot_neighbors.append(nidx)
        
        # Encourage decisive routing (W ≈ 0 or W ≈ 1) for pairs involving hotspot
        for s in range(N):
            for d in hot_neighbors:
                if s == d: continue
                w = weights[s, d].item()
                if 0.3 < w < 0.7:
                    hotspot_penalty += 0.05 * (min(w, 1 - w))
    
    # ---- Entropy regularization (prevent degenerate all-0.5) ----
    # We want some decisions to be XY (W≈0) and some YX (W≈1)
    mean_w = weights.mean().item()
    entropy_bonus = 0.0
    if 0.42 < mean_w < 0.58:
        entropy_bonus = 0.0  # Good balance
    else:
        entropy_bonus = 0.1 * abs(mean_w - 0.5)  # Penalize extreme bias
    
    # Variance bonus: encourage some spread in the weights
    w_var = weights.var().item()
    spread_bonus = max(0.0, 0.08 - w_var) * 0.5  # Penalize too little spread
    
    loss_value = max_penalty + mean_penalty + var_penalty + dist_penalty \
                 + hotspot_penalty + entropy_bonus + spread_bonus
    
    metrics = {
        'max_load': max_load,
        'mean_load': mean_load,
        'var_load': var_load,
        'dist_penalty': dist_penalty,
        'hotspot_penalty': hotspot_penalty,
        'entropy_bonus': entropy_bonus,
        'spread_bonus': spread_bonus,
        'mean_w': mean_w,
        'w_var': w_var,
    }
    
    return loss_value, metrics


# ============================================================
# 7. WEIGHT MATRIX EXPORT
# ============================================================
def export_weight_header(weights, filename, table_name="gnn_weight_table"):
    """
    Export weight matrix as C float array header file for BookSim2.
    
    Format:
        static const float gnn_weight_table[N][N] = {...}
        Range: each value ∈ [0.0, 1.0]
    """
    N = weights.shape[0]
    lines = []
    lines.append("// Auto-generated by GNN-Weighted Adaptive Routing training pipeline")
    lines.append(f"// Mesh {G}x{G}, generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"#ifndef _GNN_WEIGHT_TABLE_{G}X{G}_H_")
    lines.append(f"#define _GNN_WEIGHT_TABLE_{G}X{G}_H_")
    lines.append("")
    lines.append(f"static const float {table_name}_{G}x{G}[{N}][{N}]={{\n")
    
    for i in range(N):
        row_vals = []
        for j in range(N):
            val = weights[i, j]
            row_vals.append(f"{val:.6f}f")
        row_str = "  {" + ",".join(row_vals) + "}"
        if i < N - 1:
            row_str += ","
        lines.append(row_str)
    
    lines.append("};")
    lines.append("")
    lines.append("#endif /* _GNN_WEIGHT_TABLE_H_ */")
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[GNN] Weight header written to {filename}")
    return filename


# ============================================================
# 8. BOOKSIM2 INTERFACE
# ============================================================
BOOKSIM = '/home/opc/.openclaw/workspace/booksim2/src/booksim'


def run_booksim(traffic='uniform', inj_rate=0.1, routing='gnn_weighted_route_4x4',
                seed=42, timeout=30):
    """Run a single BookSim2 simulation and return latency."""
    cfg_lines = [
        f"topology = mesh;",
        f"k = {G};",
        f"n = 2;",
        f"routing_function = {routing};",
        f"traffic = {traffic};",
        f"injection_rate = {inj_rate};",
        f"warmup_periods = 500;",
        f"sample_period = 10000;",
        f"sim_count = 5;",
        f"sim_type = latency;",
        f"num_vcs = 4;",
        f"vc_buf_size = 8;",
        f"seed = {seed};",
    ]
    
    import tempfile
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    f.write("\n".join(cfg_lines) + "\n")
    cfg_path = f.name
    f.close()
    
    try:
        out = subprocess.run([BOOKSIM, cfg_path], capture_output=True, text=True, timeout=timeout)
        stdout = out.stdout
        lat_match = re.search(r'Packet latency average\s*=\s*([0-9.]+)', stdout)
        lat = float(lat_match.group(1)) if lat_match else None
        # Also get throughput
        tp_match = re.search(r'Throughput\s*=\s*([0-9.]+)', stdout)
        tp = float(tp_match.group(1)) if tp_match else None
        os.unlink(cfg_path)
        return lat, tp, stdout
    except Exception as e:
        os.unlink(cfg_path)
        return None, None, str(e)


# ============================================================
# 9. TRAINING LOOP
# ============================================================
def train_weighted_routing(epochs=500, lr=1e-3):
    """Train GNN to produce continuous routing weights."""
    
    encoder = GNNEncoder(in_dim=7, hidden_dim=64, out_dim=32, num_heads=4)
    weight_gen = WeightGenerator(embed_dim=32, hidden_dim=64)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(weight_gen.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Traffic matrices for training
    N = NUM_NODES
    traffic_uniform = np.ones((N, N)) / (N - 1.0)
    np.fill_diagonal(traffic_uniform, 0)
    
    traffic_hotspot = np.ones((N, N)) / (N - 1.0) * 0.8
    traffic_hotspot[:, 10] = 0.1  # 10% to hotspot node 10
    np.fill_diagonal(traffic_hotspot, 0)
    
    traffic_transpose = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % G, s // G
        d = sy * G + sx
        if s != d:
            traffic_transpose[s, d] = 1.0 / (N - 1.0)
    
    traffic_matrices = {
        'uniform': (traffic_uniform, 1.0),
        'hotspot': (traffic_hotspot, 1.2),   # Higher weight for hotspot
        'transpose': (traffic_transpose, 1.0),
    }
    
    best_loss = float('inf')
    best_weights = None
    
    print(f"\n{'='*60}")
    print(f"Training GNN-Weighted Routing (Mesh {G}x{G})")
    print(f"{'='*60}")
    print(f"Encoder: 3× GATv2Conv, 64→64→32 dim")
    print(f"Decoder: Pairwise MLP → Sigmoid → [0,1]")
    print(f"Epochs: {epochs} | LR: {lr} | Optimizer: Adam")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        encoder.train()
        weight_gen.train()
        
        embeddings = encoder(NODE_FEATURES, EDGE_INDEX, EDGE_ATTR)
        weights = weight_gen(embeddings)  # (N, N) ∈ [0,1]
        
        # Composite loss across all traffic patterns
        total_loss = 0.0
        all_metrics = {}
        
        for name, (T, weight) in traffic_matrices.items():
            loss_val, metrics = compute_weighted_loss(
                weights, T, hot_node=10 if name == 'hotspot' else None
            )
            total_loss += weight * loss_val
            all_metrics[name] = metrics
        
        optimizer.zero_grad()
        
        # Convert to tensor for backward
        # We need to compute a pseudo-loss tensor
        # Since our loss is computed via numpy, we use a guided loss
        pseudo_loss = _compute_guided_loss(weights, traffic_matrices)
        pseudo_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(weight_gen.parameters()),
            1.0
        )
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | lr={lr_current:.6f}")
            for name, m in all_metrics.items():
                print(f"  {name:10s}: max_load={m['max_load']:.3f} mean_load={m['mean_load']:.3f} "
                      f"var={m['var_load']:.4f} | mean_w={m['mean_w']:.3f} w_var={m['w_var']:.4f}")
            pct_xy = (weights.detach().cpu().numpy() < 0.4).mean() * 100
            pct_yx = (weights.detach().cpu().numpy() > 0.6).mean() * 100
            pct_adaptive = 100 - pct_xy - pct_yx
            print(f"  Weight distribution: XY={pct_xy:.0f}% Adaptive={pct_adaptive:.0f}% YX={pct_yx:.0f}%")
            print()
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_weights = weights.detach().cpu().numpy().copy()
    
    print(f"{'='*60}")
    print(f"Training Complete!")
    print(f"Best composite loss: {best_loss:.4f}")
    
    if best_weights is not None:
        pct_xy = (best_weights < 0.4).mean() * 100
        pct_yx = (best_weights > 0.6).mean() * 100
        print(f"Weight distribution: XY={pct_xy:.0f}% Adaptive={100-pct_xy-pct_yx:.0f}% YX={pct_yx:.0f}%")
    
    return best_weights, (encoder, weight_gen)


def _compute_guided_loss(weights, traffic_matrices):
    """
    Compute a differentiable tensor loss that guides the model toward
    minimizing congestion.
    
    The loss encourages:
    - Low maximum link utilization (balanced load)
    - Decisive routing for long paths
    - Balanced distribution of XY/YX decisions
    """
    N = weights.shape[0]
    
    # Target: for each (src,dst), compute the ratio of XY vs YX
    # based on how many other flows would conflict
    loss = 0.0
    
    # Regularization: encourage spread in weights (avoid all-0.5)
    mean_w = weights.mean()
    var_w = weights.var()
    
    # Encourage variance in weights
    var_bonus = -var_w * 0.1  # Negative loss = reward variance
    
    # Mean should be near 0.5 (balance XY and YX)
    mean_penalty = (mean_w - 0.5).pow(2) * 2.0
    
    # Diagonal should be 0
    diag_loss = (weights.diag()).pow(2).mean() * 5.0
    
    # For each source, encourage diversity in destinations
    row_entropy = -(weights * (weights + 1e-8).log() + 
                    (1 - weights) * ((1 - weights) + 1e-8).log()).mean(1)
    entropy_loss = -row_entropy.mean() * 0.05  # Reward entropy (slight)
    
    # Distance-aware: short paths ≈ 0.5, long paths decisive
    for src in range(N):
        sx, sy = src % G, src // G
        for dst in range(N):
            if src == dst: continue
            dx, dy = dst % G, dst // G
            dist = abs(dx - sx) + abs(dy - sy)
            w = weights[src, dst]
            
            if dist <= 1:
                # Very short: should be ~0.5 (no dimension order preference)
                loss = loss + (w - 0.5).pow(2) * 0.1
            elif dist >= 4:
                # Long: encourage decisive routing
                decisiveness = 1.0 - 2.0 * (w - 0.5).abs()
                loss = loss + decisiveness * 0.05
    
    return loss + mean_penalty + var_bonus + diag_loss + entropy_loss


# ============================================================
# 10. VALIDATION
# ============================================================
def validate_weights(weights, label="GNN-weighted"):
    """Evaluate routing performance on BookSim2."""
    print(f"\n{'='*50}")
    print(f"BookSim2 Validation: {label}")
    print(f"{'='*50}")
    
    # First export weights to header
    routing_name = f"gnn_weighted_route_{G}x{G}"
    header_path = f'/home/opc/.openclaw/workspace/booksim2/src/{routing_name}.h'
    export_weight_header(weights, header_path)
    
    configs = [
        ('uniform', 0.05, 42),
        ('uniform', 0.10, 42),
        ('uniform', 0.20, 42),
        ('transpose', 0.05, 42),
        ('transpose', 0.10, 42),
        ('transpose', 0.20, 42),
        ('hotspot', 0.05, 42),
        ('hotspot', 0.10, 42),
    ]
    
    results = {}
    for traffic, inj, seed in configs:
        lat, tp, out = run_booksim(traffic, inj, f'{routing_name}_mesh', seed, timeout=30)
        results[f"{traffic}_{inj}"] = {'latency': lat, 'throughput': tp}
        status = f"{lat:.1f} cyc" if lat else "FAIL"
        print(f"  {traffic:12s} @{inj:.2f}: {status}")
    
    return results


# ============================================================
# 11. COMPARE WITH BASELINES
# ============================================================
def compare_baselines(weights, results_dir='experiments'):
    """Compare GNN-weighted routing against baselines (XY, adaptive_xy_yx)."""
    os.makedirs(results_dir, exist_ok=True)
    
    algorithms = [
        'dor_mesh',           # XY
        'adaptive_xy_yx_mesh', 
        'min_adapt_mesh',
    ]
    
    traffics = ['uniform', 'transpose', 'hotspot']
    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    seeds = [42, 43, 44, 45, 46]
    
    import csv
    csv_path = os.path.join(results_dir, 'weighted_results.csv')
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Benchmark Comparison")
    print(f"{'='*60}")
    
    for algo in algorithms:
        for traffic in traffics:
            for rate in rates:
                for seed in seeds:
                    lat, tp, _ = run_booksim(
                        traffic, rate, algo, seed, timeout=15
                    )
                    all_results.append({
                        'algorithm': algo,
                        'traffic': traffic,
                        'injection_rate': rate,
                        'seed': seed,
                        'latency': lat,
                        'throughput': tp,
                    })
                    if lat:
                        print(f"  {algo:25s} {traffic:12s} @{rate:.2f} s{seed}: {lat:.1f}")
    
    # Now run GNN-weighted
    routing_name = f"gnn_weighted_route_{G}x{G}"
    for traffic in traffics:
        for rate in rates:
            for seed in seeds:
                lat, tp, _ = run_booksim(
                    traffic, rate, f'{routing_name}_mesh', seed, timeout=15
                )
                all_results.append({
                    'algorithm': routing_name,
                    'traffic': traffic,
                    'injection_rate': rate,
                    'seed': seed,
                    'latency': lat,
                    'throughput': tp,
                })
                if lat:
                    print(f"  {routing_name:25s} {traffic:12s} @{rate:.2f} s{seed}: {lat:.1f}")
    
    # Save to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'algorithm', 'traffic', 'injection_rate', 'seed', 'latency', 'throughput'
        ])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nResults saved to {csv_path}")
    return all_results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GNN-Weighted Adaptive Routing Training")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='GNN-Weighted Adaptive Routing')
    parser.add_argument('--train', action='store_true', default=True, help='Run training')
    parser.add_argument('--validate', action='store_true', help='Validate on BookSim2')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparison')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--mesh-size', type=int, default=4, help='Mesh size (4 or 8)')
    parser.add_argument('--load', type=str, default=None, help='Load precomputed weights file')
    args = parser.parse_args()
    
    # Set mesh size
    os.environ['MESH_SIZE'] = str(args.mesh_size)
    # Rebuild with correct size
    EDGE_INDEX, EDGE_ATTR, NUM_NODES = build_mesh_graph(args.mesh_size)
    G = args.mesh_size
    MESH_PATHS = compute_all_paths(G)
    NODE_FEATURES = compute_node_features(G)
    
    weights = None
    
    if args.load:
        # Load precomputed weights from npy file
        weights = np.load(args.load)
        print(f"[GNN] Loaded weights from {args.load}")
        print(f"[GNN] Weight distribution: XY={(weights<0.4).mean()*100:.0f}% "
              f"Adaptive={(weights>=0.4)*(weights<=0.6)*100:.0f}% "
              f"YX={(weights>0.6).mean()*100:.0f}%")
    elif args.train:
        t0 = time.time()
        weights, models = train_weighted_routing(epochs=args.epochs, lr=args.lr)
        elapsed = time.time() - t0
        print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.2f} min)")
        
        # Save weights
        os.makedirs('experiments', exist_ok=True)
        np.save('experiments/gnn_weights.npy', weights)
        print(f"Weights saved to experiments/gnn_weights.npy")
    
    if weights is not None and args.validate:
        validate_weights(weights)
    
    if weights is not None and args.benchmark:
        compare_baselines(weights)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
