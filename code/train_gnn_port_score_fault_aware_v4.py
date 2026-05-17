#!/usr/bin/env python3
"""
GNN-Port-Score Routing: v4 - Fault-Aware Training
==================================================
Retrain GNN Port Score với faulty topology support.

Key differences from v3:
1. Node features mở rộng: thêm fault-aware features (7→12 dim)
2. Data augmentation: 1 fault-free + 20 faulty topologies
3. Training targets: chỉ consider available (non-faulty) ports
4. Validation: unseen faulty patterns
5. Export header có fault-aware fallback chain

Author: Ngoc Anh for Thay Hieu
Date: 16/05/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
import os, time, json, math, sys

DEVICE = 'cpu'
print(f"[GNN-PortScore-v4] Using device: {DEVICE}")

# ============================================================
# 1. MESH TOPOLOGY
# ============================================================
def build_mesh_graph(G=4):
    edges = []
    edge_attr = []
    for y in range(G):
        for x in range(G):
            idx = y * G + x
            if x > 0:
                edges.append([idx, y * G + (x-1)])
                edges.append([y * G + (x-1), idx])
                edge_attr.append([1.0])
                edge_attr.append([1.0])
            if y > 0:
                edges.append([idx, (y-1) * G + x])
                edges.append([(y-1) * G + x, idx])
                edge_attr.append([2.0])
                edge_attr.append([2.0])
    edge_index = torch.LongTensor(edges).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attr)
    return edge_index, edge_attr

G = 4
N = G * G
EDGE_INDEX, EDGE_ATTR = build_mesh_graph(G)

# Port mapping: 0=E(x+1), 1=W(x-1), 2=S(y+1), 3=N(y-1)
PORT_NAMES = ['E', 'W', 'S', 'N']

def port_direction(node, port, G=4):
    """Return neighbor if port leads there, or None for edge exits."""
    x, y = node % G, node // G
    if port == 0:  # E (x+1)
        return node + 1 if x < G - 1 else None
    elif port == 1:  # W (x-1)
        return node - 1 if x > 0 else None
    elif port == 2:  # S (y+1)
        return node + G if y < G - 1 else None
    elif port == 3:  # N (y-1)
        return node - G if y > 0 else None
    return None

def is_minimal_port(cur, dst, port, G=4):
    """Check if port moves packet toward dst (Manhattan distance)."""
    cx, cy = cur % G, cur // G
    dx, dy = dst % G, dst // G
    nx, ny = cx, cy
    if port == 0:
        nx = cx + 1
    elif port == 1:
        nx = cx - 1
    elif port == 2:
        ny = cy + 1
    elif port == 3:
        ny = cy - 1
    
    cur_dist = abs(dx - cx) + abs(dy - cy)
    new_dist = abs(dx - nx) + abs(dy - ny)
    return new_dist < cur_dist

def get_minimal_ports(cur, dst, G=4):
    """Return list of minimal ports for (cur, dst)."""
    return [p for p in range(4) if is_minimal_port(cur, dst, p, G)]

def get_next_node(cur, port, G=4):
    """Get neighbor node reached by taking port from cur."""
    x, y = cur % G, cur // G
    if port == 0:
        return cur + 1 if x < G - 1 else -1
    elif port == 1:
        return cur - 1 if x > 0 else -1
    elif port == 2:
        return cur + G if y < G - 1 else -1
    elif port == 3:
        return cur - G if y > 0 else -1
    return -1

def is_at_edge(node, port, G=4):
    """Check if port at node exits the mesh (no neighbor)."""
    return get_next_node(node, port, G) < 0


# ============================================================
# 2. FAULTY LINK GENERATOR
# ============================================================
def generate_faulty_links(num_fails, fail_seed, G=4):
    """
    Generate faulty links matching BookSim2's InsertRandomFaults.
    
    Returns:
        faulty_links: set of (node, port) tuples indicating failed output channels
    """
    N = G * G
    rng = np.random.RandomState(fail_seed)
    
    total_channels = N * 4  # 4 ports per node
    faulty_links = set()
    used_indices = set()
    
    max_attempts = total_channels * 2
    faults_placed = 0
    
    for _ in range(max_attempts):
        if faults_placed >= num_fails:
            break
        
        fault_node = rng.randint(0, N)
        fault_port = rng.randint(0, 4)
        
        # Skip edge exits (can't fail external links)
        if is_at_edge(fault_node, fault_port, G):
            continue
        
        idx = fault_node * 4 + fault_port
        if idx in used_indices:
            continue
        
        used_indices.add(idx)
        faulty_links.add((fault_node, fault_port))
        faults_placed += 1
    
    return faulty_links


def get_active_ports(node, faulty_links, G=4):
    """Return set of active (non-faulty) ports for a given node."""
    all_ports = set(range(4))
    faulty = {p for p in range(4) if (node, p) in faulty_links}
    return all_ports - faulty


# ============================================================
# 3. NODE FEATURES (FAULT-AWARE)
# ============================================================
def compute_node_features_fault_aware(G=4, faulty_links=None):
    """
    Compute 12-dim node features with fault awareness.
    
    Features:
        0: x / (G-1)
        1: y / (G-1)
        2: active_degree / 4.0
        3: betweenness centrality
        4: corner flag
        5: edge flag
        6: center flag
        7: active_neighbors / 4.0
        8: port 0 (E) active (1=active, 0=failed)
        9: port 1 (W) active
        10: port 2 (S) active
        11: port 3 (N) active
    """
    if faulty_links is None:
        faulty_links = set()
    
    N = G * G
    features = np.zeros((N, 12))
    
    betweenness = {
        0: 0.00, 1: 0.07, 2: 0.07, 3: 0.00,
        4: 0.07, 5: 0.33, 6: 0.33, 7: 0.07,
        8: 0.07, 9: 0.33, 10: 0.33, 11: 0.07,
        12: 0.00, 13: 0.07, 14: 0.07, 15: 0.00
    }
    
    for y in range(G):
        for x in range(G):
            idx = y * G + x
            
            # Position
            features[idx, 0] = x / max(G-1, 1)
            features[idx, 1] = y / max(G-1, 1)
            
            # Degree (counting active ports)
            ports_active = get_active_ports(idx, faulty_links, G)
            active_deg = 0
            for p in range(4):
                if p in ports_active and not is_at_edge(idx, p, G):
                    active_deg += 1
            features[idx, 2] = active_deg / 4.0
            
            # Betweenness
            features[idx, 3] = betweenness.get(idx, 0)
            
            # Position type
            corner = (x == 0 or x == G-1) and (y == 0 or y == G-1)
            edge = (x == 0 or x == G-1 or y == 0 or y == G-1) and not corner
            center = not corner and not edge
            features[idx, 4] = 1.0 if corner else 0.0
            features[idx, 5] = 1.0 if edge else 0.0
            features[idx, 6] = 1.0 if center else 0.0
            
            # Active neighbors count / 4
            active_neighbors = 0
            for p in range(4):
                if p in ports_active and not is_at_edge(idx, p, G):
                    active_neighbors += 1
            features[idx, 7] = active_neighbors / 4.0
            
            # Per-port active status
            for p in range(4):
                if p in ports_active and not is_at_edge(idx, p, G):
                    features[idx, 8 + p] = 1.0
                else:
                    features[idx, 8 + p] = 0.0
    
    return torch.FloatTensor(features)


def compute_node_features_fault_free(G=4):
    """Compute standard 7-dim features (fallback, no fault info)."""
    N = G * G
    features = np.zeros((N, 7))
    betweenness = {
        0: 0.00, 1: 0.07, 2: 0.07, 3: 0.00,
        4: 0.07, 5: 0.33, 6: 0.33, 7: 0.07,
        8: 0.07, 9: 0.33, 10: 0.33, 11: 0.07,
        12: 0.00, 13: 0.07, 14: 0.07, 15: 0.00
    }
    for y in range(G):
        for x in range(G):
            idx = y * G + x
            features[idx, 0] = x / max(G-1, 1)
            features[idx, 1] = y / max(G-1, 1)
            deg = 2 + (x > 0) + (x < G-1) + (y > 0) + (y < G-1)
            features[idx, 2] = deg / 4.0
            features[idx, 3] = betweenness.get(idx, 0)
            corner = (x == 0 or x == G-1) and (y == 0 or y == G-1)
            edge = (x == 0 or x == G-1 or y == 0 or y == G-1) and not corner
            center = not corner and not edge
            features[idx, 4] = 1.0 if corner else 0.0
            features[idx, 5] = 1.0 if edge else 0.0
            features[idx, 6] = 1.0 if center else 0.0
    return torch.FloatTensor(features)


# ============================================================
# 4. COMPUTE OPTIMAL PORT SCORES (FAULT-AWARE)
# ============================================================
def compute_port_loads_fault_aware(traffic_matrix, faulty_links, G=4, N=16):
    """
    Compute link load for each port, considering faulty links.
    
    For faulty links: traffic that would use them must be rerouted
    via the other available minimal port.
    
    Returns:
        L_available: load on each available port at each node
        L_min: minimum possible load (best case)
    """
    L_available = np.zeros((N, 4))
    
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            
            # Adaptive minimal routing on faulty mesh:
            # At each hop, try available minimal ports, prefer less loaded ones
            cur = src
            max_hops = 20  # safety limit
            hops_taken = 0
            
            while cur != dst and hops_taken < max_hops:
                hops_taken += 1
                min_ports = get_minimal_ports(cur, dst, G)
                
                # Filter available (non-faulty) minimal ports
                available_ports = [p for p in min_ports if (cur, p) not in faulty_links]
                
                if not available_ports:
                    # Misroute: try any non-faulty port that isn't edge-exit
                    available_ports = [p for p in range(4) 
                                       if (cur, p) not in faulty_links 
                                       and not is_at_edge(cur, p, G)]
                
                if not available_ports:
                    break  # stuck (shouldn't happen with reasonable fault rates)
                
                # Pick port with lowest load so far (congestion-aware)
                best_port = min(available_ports, key=lambda p: L_available[cur, p])
                L_available[cur, best_port] += rate
                cur = get_next_node(cur, best_port, G)
                if cur < 0:
                    break
    
    # Also compute XY-only and YX-only loads (with fault avoidance) as reference
    L_xy = np.zeros((N, 4))
    L_yx = np.zeros((N, 4))
    
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            
            # XY on faulty mesh: try X first, if faulty try Y, repeat
            cur = src
            max_hops = 20
            hops = 0
            while cur != dst and hops < max_hops:
                hops += 1
                cx, cy = cur % G, cur // G
                dx, dy = dst % G, dst // G
                
                taken = False
                # Try X first
                if cx != dx:
                    if dx > cx and (cur, 0) not in faulty_links:
                        L_xy[cur, 0] += rate; cur = cur + 1; taken = True
                    elif dx < cx and (cur, 1) not in faulty_links:
                        L_xy[cur, 1] += rate; cur = cur - 1; taken = True
                if not taken and cy != dy:
                    if dy > cy and (cur, 2) not in faulty_links:
                        L_xy[cur, 2] += rate; cur = cur + G; taken = True
                    elif dy < cy and (cur, 3) not in faulty_links:
                        L_xy[cur, 3] += rate; cur = cur - G; taken = True
                if not taken:
                    # Fault avoidance: try any available port
                    for p in range(4):
                        if (cur, p) not in faulty_links and not is_at_edge(cur, p, G):
                            L_xy[cur, p] += rate
                            cur = get_next_node(cur, p, G)
                            taken = True
                            break
                if not taken:
                    break
            
            cur = src
            hops = 0
            while cur != dst and hops < max_hops:
                hops += 1
                cx, cy = cur % G, cur // G
                dx, dy = dst % G, dst // G
                
                taken = False
                # Try Y first
                if cy != dy:
                    if dy > cy and (cur, 2) not in faulty_links:
                        L_yx[cur, 2] += rate; cur = cur + G; taken = True
                    elif dy < cy and (cur, 3) not in faulty_links:
                        L_yx[cur, 3] += rate; cur = cur - G; taken = True
                if not taken and cx != dx:
                    if dx > cx and (cur, 0) not in faulty_links:
                        L_yx[cur, 0] += rate; cur = cur + 1; taken = True
                    elif dx < cx and (cur, 1) not in faulty_links:
                        L_yx[cur, 1] += rate; cur = cur - 1; taken = True
                if not taken:
                    for p in range(4):
                        if (cur, p) not in faulty_links and not is_at_edge(cur, p, G):
                            L_yx[cur, p] += rate
                            cur = get_next_node(cur, p, G)
                            taken = True
                            break
                if not taken:
                    break
    
    L_min = np.minimum(L_xy, L_yx)
    
    return L_available, L_min


def compute_optimal_port_scores_fault_aware(traffic_matrix, faulty_links, G=4, N=16):
    """
    Compute optimal port scores for each (cur, dst) pair on faulty topology.
    
    Key difference from v3:
    - Only available (non-faulty) minimal ports get positive scores
    - If no minimal port available, fallback to any available port (misroute)
    - Targets reflect congestion-aware adaptive routing
    
    Returns:
        target_scores[cur][dst][port]: 4 scores for each (cur,dst) pair
    """
    L_adapt, L_min = compute_port_loads_fault_aware(traffic_matrix, faulty_links, G, N)
    L_avg = L_adapt.copy()
    
    target_scores = np.zeros((N, N, 4))
    
    for cur in range(N):
        for dst in range(N):
            if cur == dst:
                continue
            
            min_ports = get_minimal_ports(cur, dst, G)
            available_min_ports = [p for p in min_ports if (cur, p) not in faulty_links and not is_at_edge(cur, p, G)]
            
            if available_min_ports:
                # Score available minimal ports
                for port in available_min_ports:
                    load = L_avg[cur, port]
                    if load > 0:
                        target_scores[cur, dst, port] = 1.0 / (1.0 + load)
                    else:
                        target_scores[cur, dst, port] = 1.0
                
                # Normalize among available minimal ports
                total = sum(target_scores[cur, dst, p] for p in available_min_ports)
                if total > 0:
                    for p in available_min_ports:
                        target_scores[cur, dst, p] /= total
            else:
                # No minimal port available: misroute
                # Try any non-faulty non-edge port
                available_ports = [p for p in range(4) 
                                   if (cur, p) not in faulty_links 
                                   and not is_at_edge(cur, p, G)]
                
                if available_ports:
                    for port in available_ports:
                        load = L_avg[cur, port]
                        if load > 0:
                            target_scores[cur, dst, port] = 1.0 / (1.0 + load)
                        else:
                            target_scores[cur, dst, port] = 1.0
                    
                    total = sum(target_scores[cur, dst, p] for p in available_ports)
                    if total > 0:
                        for p in available_ports:
                            target_scores[cur, dst, p] /= total
                else:
                    # All ports faulty or edge: uniform (shouldn't happen)
                    for p in range(4):
                        if not is_at_edge(cur, p, G):
                            target_scores[cur, dst, p] = 0.25
    
    return target_scores, L_min


# ============================================================
# 5. FAULTY DATASET GENERATION
# ============================================================
def generate_traffic_patterns(G=4, N=16):
    """Generate diverse traffic patterns for training (same as v3)."""
    patterns = []
    
    # Pattern 1: Uniform
    T1 = np.ones((N, N)) / (N - 1.0)
    np.fill_diagonal(T1, 0)
    patterns.append((T1, "uniform"))
    
    # Pattern 2: Hotspot at node 10 (center)
    T2 = np.ones((N, N)) * 0.03
    T2[:, 10] = 0.15
    np.fill_diagonal(T2, 0)
    patterns.append((T2, "hotspot10"))
    
    # Pattern 3: Transpose
    T3 = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % G, s // G
        d = sy * G + sx
        if s != d:
            T3[s, d] = 1.0 / 15.0
    patterns.append((T3, "transpose"))
    
    # Pattern 4: Bit complement (3-x, 3-y)
    T4 = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % G, s // G
        d = (3-sy) * G + (3-sx)
        if s != d:
            T4[s, d] = 1.0 / 15.0
    patterns.append((T4, "bitcomp"))
    
    # Pattern 5: Hotspot at node 5
    T5 = np.ones((N, N)) * 0.03
    T5[:, 5] = 0.15
    np.fill_diagonal(T5, 0)
    patterns.append((T5, "hotspot5"))
    
    # Pattern 6: Shuffle
    np.random.seed(42)
    perm = np.random.permutation(N)
    T6 = np.zeros((N, N))
    for s in range(N):
        d = perm[s]
        if s != d:
            T6[s, d] = 1.0 / sum(1 for s2 in range(N) if perm[s2] != s2)
    patterns.append((T6, "shuffle"))
    
    # Pattern 7: Bit reversal (transpose coordinates)
    T7 = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % G, s // G
        d = sx * G + sy
        if s != d:
            T7[s, d] = 1.0 / 15.0
    patterns.append((T7, "bitrev"))
    
    return patterns


def generate_faulty_dataset(G=4, N=16, seed=12345):
    """
    Generate training dataset: 1 fault-free + 20 faulty topologies.
    
    Fault patterns:
    - 5 topologies với 2 link failures (4.2% of 48 channels)
    - 5 topologies với 3 link failures (6.25%)
    - 5 topologies với 5 link failures (10.4%)
    - 5 topologies với 7 link failures (14.6%)
    
    Returns:
        dataset: list of (traffic_matrix, faulty_links, name) tuples
    """
    patterns = generate_traffic_patterns(G, N)
    dataset = []
    
    rng = np.random.RandomState(seed)
    
    # Fault-free topology (all traffic patterns)
    base_seed = seed
    for T, pname in patterns:
        dataset.append({
            'traffic': T.copy(),
            'faulty_links': set(),
            'name': f"fault_free_{pname}",
            'n_fails': 0,
        })
    
    # Faulty topologies with increasing fault levels
    fault_levels = [2, 2, 2, 2, 2,       # 5 topologies × 2 fails
                    3, 3, 3, 3, 3,       # 5 × 3 fails
                    5, 5, 5, 5, 5,       # 5 × 5 fails
                    7, 7, 7, 7, 7]       # 5 × 7 fails
    
    for idx, n_fails in enumerate(fault_levels):
        fseed = base_seed + 1000 + idx
        faulty_links = generate_faulty_links(n_fails, fseed, G)
        
        for T, pname in patterns:
            dataset.append({
                'traffic': T.copy(),
                'faulty_links': faulty_links.copy(),
                'name': f"fault{n_fails}_{idx}_{pname}",
                'n_fails': n_fails,
            })
    
    print(f"[Dataset] Generated {len(dataset)} training samples:")
    n_unique_topo = len(set(tuple(sorted(d['faulty_links'])) for d in dataset))
    print(f"  Unique topologies: {n_unique_topo}")
    print(f"  Traffic patterns: {len(patterns)}")
    print(f"  Total: {len(patterns)} × {n_unique_topo} = {len(dataset)}")
    
    return dataset


def generate_validation_dataset(G=4, N=16):
    """
    Generate validation dataset: 3 unseen faulty topologies with different seeds.
    Used for validation during training.
    """
    patterns = generate_traffic_patterns(G, N)
    val_dataset = []
    
    # Unseen fault patterns (different seeds)
    val_fault_configs = [
        (3, 5555),   # 3 failures
        (5, 7777),   # 5 failures
        (6, 9999),   # 6 failures
    ]
    
    for n_fails, fseed in val_fault_configs:
        faulty_links = generate_faulty_links(n_fails, fseed, G)
        for T, pname in patterns:
            val_dataset.append({
                'traffic': T.copy(),
                'faulty_links': faulty_links.copy(),
                'name': f"val_fault{n_fails}_{pname}",
                'n_fails': n_fails,
            })
    
    print(f"[Val Dataset] {len(val_dataset)} samples")
    return val_dataset


# ============================================================
# 6. GNN PORT SCORE MODEL (FAULT-AWARE)
# ============================================================
class GNNPortScoreFaultAware(nn.Module):
    """
    GNN Port Score model với fault-aware features (12-dim input).
    
    Architecture (same as v3 but with larger input dimension):
    - GATv2 encoder → node embeddings (32-dim)
    - Port decoder: for each (cur, dst), compute 4 port scores
    - 12 input features: 7 spatial + 5 fault-aware
    """
    
    def __init__(self, in_dim=12, hidden_dim=64, embed_dim=32):
        super().__init__()
        # Encoder (3-layer GATv2)
        self.conv1 = GATv2Conv(in_dim, hidden_dim // 4, heads=4, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=1)
        self.conv3 = GATv2Conv(hidden_dim, embed_dim, heads=1, edge_dim=1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.15)
        
        # Port-specific decoders (one per port)
        self.port_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, 32),
                nn.LayerNorm(32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 1),
            ) for _ in range(4)
        ])
    
    def encode(self, x, edge_index, edge_attr):
        x = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.elu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = self.norm3(self.conv3(x, edge_index, edge_attr))
        return x
    
    def decode_scores(self, embeddings):
        """For each (cur, dst), compute 4 port scores."""
        n = embeddings.size(0)
        scores = torch.zeros(n, n, 4)
        
        cur_emb = embeddings.unsqueeze(1).expand(n, n, -1)  # [N, N, D]
        dst_emb = embeddings.unsqueeze(0).expand(n, n, -1)  # [N, N, D]
        pairs = torch.cat([cur_emb, dst_emb], dim=-1)  # [N, N, 2D]
        
        for port in range(4):
            port_scores = self.port_decoders[port](pairs).squeeze(-1)  # [N, N]
            scores[:, :, port] = port_scores
        
        return scores
    
    def forward(self, x, edge_index, edge_attr):
        emb = self.encode(x, edge_index, edge_attr)
        scores = self.decode_scores(emb)
        return scores, emb
    
    def precompute_table(self, node_features):
        """Precompute 16×16×4 score table for given node features."""
        with torch.no_grad():
            self.eval()
            scores, emb = self(node_features, EDGE_INDEX, EDGE_ATTR)
            return scores.cpu().numpy()


# ============================================================
# 7. TRAINING WITH FAULT-AWARE DATA
# ============================================================
def compute_batch_loss(model, dataset, node_features_cache):
    """
    Compute average loss across a dataset.
    
    For each sample in dataset:
    1. Get node features (fault-aware, from cache or compute)
    2. Forward through model
    3. Compute KL divergence + MSE loss vs target port scores
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for sample in dataset:
            faulty_links = sample['faulty_links']
            key = tuple(sorted(faulty_links))
            
            if key in node_features_cache:
                node_feats = node_features_cache[key]
            else:
                node_feats = compute_node_features_fault_aware(G, faulty_links)
                node_features_cache[key] = node_feats
            
            traffic = sample['traffic']
            target_scores, _ = compute_optimal_port_scores_fault_aware(
                traffic, faulty_links, G, N)
            target_t = torch.FloatTensor(target_scores)
            
            logits, _ = model(node_feats, EDGE_INDEX, EDGE_ATTR)
            
            # Prepare target distribution
            target_dist = target_t.clone()
            for i in range(N):
                target_dist[i, i, :] = 0.25
            target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-10)
            
            log_probs = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean')
            logits_softmax = F.softmax(logits, dim=-1)
            mse_loss = F.mse_loss(logits_softmax, target_dist)
            
            loss = kl_loss + 0.5 * mse_loss
            total_loss += loss.item()
    
    return total_loss / len(dataset)


def train_fault_aware(epochs=300, lr=3e-4, seed=12345):
    """Train fault-aware GNN Port Score model."""
    
    print(f"\n{'='*70}")
    print(f"GNN-PortScore v4: Fault-Aware Training")
    print(f"{'='*70}")
    
    t0 = time.time()
    train_dataset = generate_faulty_dataset(G, N, seed)
    print(f"  Dataset generation: {time.time()-t0:.1f}s")
    val_dataset = generate_validation_dataset(G, N)
    
    # Model
    model = GNNPortScoreFaultAware(in_dim=12, hidden_dim=64, embed_dim=32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Input features: 12 (7 spatial + 5 fault-aware)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Group training data by topology (same faulty_links -> same node features)
    print(f"\n  Precomputing targets and grouping by topology...")
    t1 = time.time()
    
    # Group: unique_topology_key -> [node_feats, [(traffic_idx, target_scores), ...]]
    from collections import defaultdict
    topo_groups = defaultdict(list)
    node_features_cache = {}
    
    for i, sample in enumerate(train_dataset):
        faulty_links = sample['faulty_links']
        key = tuple(sorted(faulty_links))
        
        if key not in node_features_cache:
            node_features_cache[key] = compute_node_features_fault_aware(G, faulty_links)
        
        target_scores, _ = compute_optimal_port_scores_fault_aware(
            sample['traffic'], faulty_links, G, N)
        topo_groups[key].append(target_scores)
    
    # Convert to list for iteration
    train_groups = [(key, node_features_cache[key], targets) for key, targets in topo_groups.items()]
    
    # Validation groups
    val_groups_dict = defaultdict(list)
    val_features_cache = {}
    for sample in val_dataset:
        fl = sample['faulty_links']
        key = tuple(sorted(fl))
        if key not in val_features_cache:
            val_features_cache[key] = compute_node_features_fault_aware(G, fl)
        val_groups_dict[key].append(
            compute_optimal_port_scores_fault_aware(sample['traffic'], fl, G, N)[0])
    val_groups = [(key, val_features_cache[key], targets) for key, targets in val_groups_dict.items()]
    
    print(f"  Precomputation: {time.time()-t1:.1f}s")
    print(f"  Train topology groups: {len(train_groups)} (each has 7 traffic patterns)")
    print(f"  Val topology groups: {len(val_groups)}")
    
    print(f"\n  Training ({epochs} epochs, {len(train_groups)} forward passes/epoch)...")
    print(f"{'='*70}")
    
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        total_train_loss = 0.0
        
        # Process each unique topology once (each has 7 traffic patterns)
        for key, node_feats, target_list in train_groups:
            logits, _ = model(node_feats, EDGE_INDEX, EDGE_ATTR)
            
            for target_scores in target_list:
                target_t = torch.FloatTensor(target_scores)
                target_dist = target_t.clone()
                for i_n in range(N):
                    target_dist[i_n, i_n, :] = 0.25
                target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-10)
                
                log_probs = F.log_softmax(logits, dim=-1)
                kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean')
                logits_softmax = F.softmax(logits, dim=-1)
                mse_loss = F.mse_loss(logits_softmax, target_dist)
                
                loss = kl_loss + 0.5 * mse_loss
                total_train_loss = total_train_loss + loss
        
        avg_train_loss = total_train_loss / len(train_dataset)
        avg_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for key, node_feats, target_list in val_groups:
                logits, _ = model(node_feats, EDGE_INDEX, EDGE_ATTR)
                for target_scores in target_list:
                    target_t = torch.FloatTensor(target_scores)
                    target_dist = target_t.clone()
                    for i_n in range(N):
                        target_dist[i_n, i_n, :] = 0.25
                    target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-10)
                    log_probs = F.log_softmax(logits, dim=-1)
                    kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean')
                    logits_softmax = F.softmax(logits, dim=-1)
                    mse_loss = F.mse_loss(logits_softmax, target_dist)
                    val_loss += (kl_loss + 0.5 * mse_loss).item()
        val_loss /= len(val_dataset)
        
        if avg_train_loss.item() < best_train_loss:
            best_train_loss = avg_train_loss.item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        model.train()
        
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                model.eval()
                ff_key = tuple(sorted(set()))
                ff_feats = node_features_cache.get(ff_key, compute_node_features_fault_aware(G, set()))
                ff_scores, _ = model(ff_feats, EDGE_INDEX, EDGE_ATTR)
                scores_np = ff_scores.cpu().numpy()
                
                correct = 0
                total_pairs = 0
                for cur in range(N):
                    for dst in range(N):
                        if cur == dst: continue
                        min_ports = get_minimal_ports(cur, dst, G)
                        if len(min_ports) == 0: continue
                        best_p = np.argmax(scores_np[cur, dst])
                        if best_p in min_ports:
                            correct += 1
                        total_pairs += 1
                pct_correct = correct / total_pairs * 100 if total_pairs > 0 else 0
            
            print(f"  Epoch {epoch+1:4d}/{epochs} | train={avg_train_loss.item():.6f} | val={val_loss:.6f} | lr={scheduler.get_last_lr()[0]:.6f} | ff_acc={pct_correct:.1f}%")
            
            # Fault-aware accuracy (inside no_grad)
            with torch.no_grad():
                if val_groups:
                    k_fn, fn_feats, _ = val_groups[0]
                    val_fl = set(k_fn) if k_fn else set()
                    fscores, _ = model(fn_feats, EDGE_INDEX, EDGE_ATTR)
                    scores_np = fscores.cpu().numpy()
                    
                    f_correct = 0
                    f_total = 0
                    for cur in range(N):
                        for dst in range(N):
                            if cur == dst: continue
                            min_ports = get_minimal_ports(cur, dst, G)
                            avail = [p for p in min_ports if (cur, p) not in val_fl]
                            if not avail: continue
                            best_p = np.argmax(scores_np[cur, dst])
                            if best_p in avail:
                                f_correct += 1
                            f_total += 1
                    if f_total > 0:
                        f_acc = f_correct / f_total * 100
                        print(f"         fault-avail-acc={f_acc:.1f}% (n_fails={len(val_fl)})")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n[Best model] Loaded from epoch with val_loss={best_val_loss:.6f}")
    
    # Final evaluation
    with torch.no_grad():
        model.eval()
        print(f"\n{'='*70}")
        print(f"Final Evaluation")
        print(f"{'='*70}")
        
        # Fault-free
        ff_feats = compute_node_features_fault_aware(G, set())
        ff_scores, _ = model(ff_feats, EDGE_INDEX, EDGE_ATTR)
        scores_np = ff_scores.detach().cpu().numpy()
        
        correct = 0
        total_pairs = 0
        for cur in range(N):
            for dst in range(N):
                if cur == dst: continue
                min_ports = get_minimal_ports(cur, dst, G)
                if len(min_ports) == 0: continue
                best_p = np.argmax(scores_np[cur, dst])
                if best_p in min_ports:
                    correct += 1
                total_pairs += 1
        print(f"  Fault-free minimal port acc: {correct}/{total_pairs} = {correct/total_pairs*100:.1f}%")
        
        # Faulty
        for n_fails, fseed in [(3, 5555), (5, 7777), (6, 9999)]:
            fl = generate_faulty_links(n_fails, fseed, G)
            nf = compute_node_features_fault_aware(G, fl)
            fscores, _ = model(nf, EDGE_INDEX, EDGE_ATTR)
            fnp = fscores.detach().cpu().numpy()
            
            f_correct = 0
            f_total = 0
            stuck = 0
            for cur in range(N):
                for dst in range(N):
                    if cur == dst: continue
                    min_ports = get_minimal_ports(cur, dst, G)
                    avail = [p for p in min_ports if (cur, p) not in fl]
                    if not avail:
                        # Check any available port
                        all_avail = [p for p in range(4) if (cur, p) not in fl and not is_at_edge(cur, p, G)]
                        if all_avail:
                            best_p = np.argmax(fnp[cur, dst])
                            if best_p in all_avail:
                                f_correct += 1
                            stuck += 1
                        f_total += 1
                        continue
                    best_p = np.argmax(fnp[cur, dst])
                    if best_p in avail:
                        f_correct += 1
                    f_total += 1
            print(f"  Fault avail ({n_fails} fails): {f_correct}/{f_total} = {f_correct/f_total*100:.1f}% (stuck={stuck})")
        
        # Sample scores
        print(f"\n  Sample port scores (fault-free):")
        for cur in [0, 5, 10, 15]:
            for dst in [0, 7, 10, 15]:
                if cur != dst:
                    s_str = ", ".join(f"{scores_np[cur, dst, p]:.3f}" for p in range(4))
                    min_ports = get_minimal_ports(cur, dst, G)
                    print(f"    [{cur}]→[{dst}]: [{s_str}]  min={min_ports}")
    
    return model, scores_np


def export_port_score_header_v4(scores, filename, G=4):
    """Export NxNx4 port score tensor to C++ header."""
    N = G * G
    lines = []
    lines.append("// Auto-generated by train_gnn_port_score_fault_aware_v4.py")
    lines.append(f"// Mesh {G}x{G}, generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("// GNN Port Score Routing v4: Fault-Aware")
    lines.append("// Port mapping: 0=E(x+1), 1=W(x-1), 2=S(y+1), 3=N(y-1)")
    lines.append("// Node features: 12-dim (7 spatial + 5 fault-aware)")
    lines.append("#ifndef _GNN_PORT_SCORE_ROUTE_4X4_V4_H_")
    lines.append("#define _GNN_PORT_SCORE_ROUTE_4X4_V4_H_")
    lines.append("")
    lines.append('#include "booksim.hpp"')
    lines.append('#include "routefunc.hpp"')
    lines.append('#include "kncube.hpp"')
    lines.append("")
    lines.append("extern int dor_next_mesh(int,int,bool);")
    lines.append("extern int gNumVCs;")
    lines.append("")
    
    lines.append(f"static const float gnn_port_scores_4x4_v4[{N}][{N}][4]={{")
    for cur in range(N):
        lines.append("  {  // cur=" + str(cur))
        for dst in range(N):
            row = "    {" + ",".join(f"{scores[cur, dst, p]:.6f}f" for p in range(4)) + "}"
            if cur < N-1 or dst < N-1:
                row += ","
            lines.append(row)
        if cur < N - 1:
            lines.append("  },")
        else:
            lines.append("  }")
    lines.append("};")
    lines.append("")
    
    helpers = """
// Helper: check if port is minimal toward destination
static inline bool gnn_is_minimal_port_v4(int cur, int dest, int port, int G=4) {
  int cx = cur % G, cy = cur / G;
  int dx = dest % G, dy = dest / G;
  int nx = cx, ny = cy;
  switch (port) {
    case 0: nx = cx + 1; break;
    case 1: nx = cx - 1; break;
    case 2: ny = cy + 1; break;
    case 3: ny = cy - 1; break;
  }
  int cur_dist = abs(dx - cx) + abs(dy - cy);
  int new_dist = abs(dx - nx) + abs(dy - ny);
  return new_dist < cur_dist;
}
"""
    lines.append(helpers.strip())
    lines.append("")
    
    routing_func = f"""
void gnn_port_score_route_4x4_v4_mesh(const Router*r, const Flit*f,
int in_channel, OutputSet*outputs, bool inject){{
  int vcBegin=0, vcEnd=gNumVCs-1;
  if (f->type == Flit::READ_REQUEST)   {{vcBegin=gReadReqBeginVC;  vcEnd=gReadReqEndVC;}}
  if (f->type == Flit::WRITE_REQUEST)  {{vcBegin=gWriteReqBeginVC; vcEnd=gWriteReqEndVC;}}
  if (f->type == Flit::READ_REPLY)     {{vcBegin=gReadReplyBeginVC;vcEnd=gReadReplyEndVC;}}
  if (f->type == Flit::WRITE_REPLY)    {{vcBegin=gWriteReplyBeginVC;vcEnd=gWriteReplyEndVC;}}
  assert(((f->vc>=vcBegin)&&(f->vc<=vcEnd))||(inject&&(f->vc<0)));
  (void)in_channel;
  int out_port;
  if(inject){{out_port=-1;outputs->Clear();outputs->AddRange(out_port,vcBegin,vcEnd);return;}}
  if(r->GetID()==f->dest){{out_port=2*gN;outputs->Clear();outputs->AddRange(out_port,vcBegin,vcEnd);return;}}
  
  int cur=r->GetID();
  int dest=f->dest;
  
  float best_score = -1e9f;
  int best_port = -1;
  float second_score = -1e9f;
  int second_port = -1;
  
  for (int p = 0; p < 4; p++) {{
    if (!gnn_is_minimal_port_v4(cur, dest, p)) continue;
    if (r->IsFaultyOutput(p)) continue;
    
    float score = gnn_port_scores_4x4_v4[cur][dest][p];
    int credit = r->GetUsedCredit(p);
    float congestion = (float)credit / 16.0f;
    float effective = score - 0.3f * congestion;
    
    if (effective > best_score) {{
      second_score = best_score; second_port = best_port;
      best_score = effective; best_port = p;
    }} else if (effective > second_score) {{
      second_score = effective; second_port = p;
    }}
  }}
  
  int const available_vcs = (vcEnd - vcBegin + 1) / 2;
  assert(available_vcs > 0);
  
  if (best_port >= 0) {{
    int best_credit = r->GetUsedCredit(best_port);
    if (best_credit > 12 && second_port >= 0) {{
      best_port = second_port;
    }}
  }}
  
  // If no port found via GNN scores, try XY/YX DOR with fault avoidance
  if (best_port < 0) {{
    int cx = cur % 4, cy = cur / 4;
    int dx = dest % 4, dy = dest / 4;
    
    // XY DOR
    if (cx != dx) {{
      int xp = (dx > cx) ? 0 : 1;
      if (!r->IsFaultyOutput(xp)) best_port = xp;
    }}
    if (best_port < 0 && cy != dy) {{
      int yp = (dy > cy) ? 2 : 3;
      if (!r->IsFaultyOutput(yp)) best_port = yp;
    }}
    
    // YX DOR (if XY failed due to faults)
    if (best_port < 0 && cy != dy) {{
      int yp = (dy > cy) ? 2 : 3;
      if (!r->IsFaultyOutput(yp)) best_port = yp;
    }}
    if (best_port < 0 && cx != dx) {{
      int xp = (dx > cx) ? 0 : 1;
      if (!r->IsFaultyOutput(xp)) best_port = xp;
    }}
  }}
  
  // ULTIMATE fallback: DOR without fault checking
  if (best_port < 0) {{
    best_port = dor_next_mesh(cur, dest, false);
  }}
  
  if (best_port < 0) best_port = 0;
  out_port = best_port;
  
  if(out_port == 1 || out_port == 3) {{
    vcBegin = vcBegin + available_vcs;
  }} else {{
    vcEnd = vcBegin + available_vcs - 1;
  }}
  
  outputs->Clear();
  outputs->AddRange(out_port, vcBegin, vcEnd);
}}
"""
    lines.append(routing_func.strip())
    lines.append("")
    lines.append("#endif /* _GNN_PORT_SCORE_ROUTE_4X4_V4_H_ */")
    
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[GNN-PortScore-v4] Header written to {filename}")
    print(f"  Table size: {N}x{N}x4 = {N*N*4} float values")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("GNN-Port-Score v4: Fault-Aware Training")
    print("=" * 70)
    
    t0 = time.time()
    
    # Also compute fault-free baseline (7-dim) for comparison
    model_v4, scores_v4 = train_fault_aware(epochs=500, lr=3e-4, seed=12345)
    
    print(f"\nTotal training + evaluation time: {time.time()-t0:.1f}s")
    
    # Save scores first
    out_dir = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments'
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/gnn_port_scores_v4.npy', scores_v4)
    print(f"[GNN-PortScore-v4] Scores saved to {out_dir}/gnn_port_scores_v4.npy")
    
    # Export fault-free header for BookSim2
    header_path = '/home/opc/.openclaw/workspace/booksim2/src/gnn_port_score_route_4x4_v4.h'
    export_port_score_header_v4(scores_v4, header_path)
    
    # Save model
    torch.save(model_v4.state_dict(), f'{out_dir}/gnn_port_score_v4_model.pt')
    print(f"[GNN-PortScore-v4] Model saved to {out_dir}/gnn_port_score_v4_model.pt")
    
    print(f"\n{'='*70}")
    print(f"Done! To compile BookSim2:")
    print(f"  1. Register in routefunc.cpp:")
    print(f"     add '#include \"gnn_port_score_route_4x4_v4.h\"'")
    print(f"     add 'gRoutingFunctionMap[\"gnn_port_score_route_4x4_v4_mesh\"] = &gnn_port_score_route_4x4_v4_mesh;'")
    print(f"  2. cd /home/opc/.openclaw/workspace/booksim2/src && make -j4")
    print(f"{'='*70}")


