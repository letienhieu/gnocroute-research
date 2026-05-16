#!/usr/bin/env python3
"""
GNN-Port-Score Routing: v3 - Port Scoring
===========================================
Instead of 1 binary weight (XY vs YX), GNN outputs 4 port scores (N/S/E/W).
At each router hop: pick minimal port (toward dst) with highest score.

Key innovations:
1. GNN encoder → node embeddings (same architecture as v2)
2. Port score decoder: for each (cur,dst), compute 4 scores
3. Training: cross-entropy, minimal ports preferred, congestion-aware
4. Precompute 4x16x4 tensor for BookSim2

Author: Ngoc Anh for Thay Hieu
Date: 16/05/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
import os, time, json, math

DEVICE = 'cpu'
print(f"[GNN-PortScore] Using device: {DEVICE}")

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

# ============================================================
# 2. NODE FEATURES
# ============================================================
def compute_node_features(G=4):
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

NODE_FEATURES = compute_node_features(G)

# ============================================================
# 3. COMPUTE OPTIMAL PORT SCORES (TRAINING TARGETS)
# ============================================================
def compute_port_loads(traffic_matrix, G=4, N=16):
    """
    Compute link load for each port under XY-only and YX-only routing.
    
    Returns:
        L_xy[node][port]: load on port at node under XY routing
        L_yx[node][port]: load on port at node under YX routing
        L_min[node][port]: minimum of XY and YX loads
    """
    # XY-only load
    L_xy = np.zeros((N, 4))
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            
            # XY path
            cur = src
            sx, sy = src % G, src // G
            dx, dy = dst % G, dst // G
            
            # X first
            step_x = 1 if dx > sx else -1
            for _ in range(abs(dx - sx)):
                port = 0 if step_x > 0 else 1
                L_xy[cur, port] += rate
                cur = cur + step_x
            
            # Y second
            step_y = 1 if dy > sy else -1
            for _ in range(abs(dy - sy)):
                port = 2 if step_y > 0 else 3
                L_xy[cur, port] += rate
                cur = cur + G * step_y
    
    # YX-only load
    L_yx = np.zeros((N, 4))
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            
            # YX path
            cur = src
            sx, sy = src % G, src // G
            dx, dy = dst % G, dst // G
            
            # Y first
            step_y = 1 if dy > sy else -1
            for _ in range(abs(dy - sy)):
                port = 2 if step_y > 0 else 3
                L_yx[cur, port] += rate
                cur = cur + G * step_y
            
            # X second
            step_x = 1 if dx > sx else -1
            for _ in range(abs(dx - sx)):
                port = 0 if step_x > 0 else 1
                L_yx[cur, port] += rate
                cur = cur + step_x
    
    # Minimum load (best case)
    L_min = np.minimum(L_xy, L_yx)
    
    return L_xy, L_yx, L_min


def compute_optimal_port_scores(traffic_matrix, G=4, N=16):
    """
    Compute optimal port scores for each (cur, dst) pair.
    
    For each (cur, dst):
    - Minimal ports (toward dst) get positive scores based on congestion
    - Non-minimal ports get score 0
    - Among minimal ports: lower congestion → higher score
    
    Returns:
        target_scores[cur][dst][port]: 4 scores (will be normalized)
    """
    L_xy, L_yx, L_min = compute_port_loads(traffic_matrix, G, N)
    
    # Use average of XY and YX as "expected" load
    L_avg = (L_xy + L_yx) / 2.0
    
    target_scores = np.zeros((N, N, 4))
    
    for cur in range(N):
        for dst in range(N):
            if cur == dst:
                continue
            
            min_ports = get_minimal_ports(cur, dst, G)
            
            if len(min_ports) == 0:
                continue
            
            # Score each minimal port based on congestion
            # Lower load → higher score (inverse relationship)
            for port in min_ports:
                load = L_avg[cur, port]
                if load > 0:
                    # Inverse proportional to load
                    target_scores[cur, dst, port] = 1.0 / (1.0 + load)
                else:
                    target_scores[cur, dst, port] = 1.0
            
            # Normalize among minimal ports to sum to 1
            total = target_scores[cur, dst, min_ports].sum()
            if total > 0:
                target_scores[cur, dst, min_ports] /= total
    
    return target_scores, L_min


# ============================================================
# 4. GNN PORT SCORE MODEL
# ============================================================
class GNNPortScore(nn.Module):
    """
    GNN that produces port scores for routing.
    
    Architecture:
    - GATv2 encoder → node embeddings (32-dim)
    - Port decoder: for each (cur, dst), compute 4 port scores
    - Port-specific bias: each port has a learned embedding
    """
    
    def __init__(self, in_dim=7, hidden_dim=64, embed_dim=32):
        super().__init__()
        # Encoder (same as v2)
        self.conv1 = GATv2Conv(in_dim, hidden_dim // 4, heads=4, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=1)
        self.conv3 = GATv2Conv(hidden_dim, embed_dim, heads=1, edge_dim=1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.15)
        
        # Port embeddings (learned, one per port)
        self.port_embed = nn.Embedding(4, embed_dim // 2)
        
        # Decoder: takes (embed[cur], embed[dst], port_embed) → score
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim // 2, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )
        
        # Alternative: simpler decoder for each port separately
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
        """
        For each (cur, dst), compute 4 port scores.
        Returns: tensor [N, N, 4]
        """
        n = embeddings.size(0)
        scores = torch.zeros(n, n, 4)
        
        for port in range(4):
            # For each (cur, dst): decoder(embed[cur], embed[dst])
            cur_emb = embeddings.unsqueeze(1).expand(n, n, -1)  # [N, N, D]
            dst_emb = embeddings.unsqueeze(0).expand(n, n, -1)  # [N, N, D]
            pairs = torch.cat([cur_emb, dst_emb], dim=-1)  # [N, N, 2D]
            
            # Get port embedding
            port_e = self.port_embed(torch.tensor([port]))  # [1, Dp]
            port_e = port_e.unsqueeze(0).unsqueeze(0).expand(n, n, -1)
            
            # Concatenate with port embedding
            combined = torch.cat([pairs, port_e], dim=-1)
            
            # Decode
            port_scores = self.decoder(combined).squeeze(-1)  # [N, N]
            scores[:, :, port] = port_scores
        
        return scores
    
    def decode_scores_simple(self, embeddings):
        """Faster: separate decoder per port."""
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
        scores = self.decode_scores_simple(emb)
        return scores, emb
    
    def precompute_port_table(self):
        """Precompute 4x16x4 table [cur][dst][port] = score."""
        with torch.no_grad():
            self.eval()
            scores, emb = self(NODE_FEATURES, EDGE_INDEX, EDGE_ATTR)
            return scores.cpu().numpy()


# ============================================================
# 5. TRAINING
# ============================================================
def generate_traffic_patterns(G=4, N=16):
    """Generate diverse traffic patterns for training."""
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
    
    # Pattern 5: Hotspot at node 5 (another center)
    T5 = np.ones((N, N)) * 0.03
    T5[:, 5] = 0.15
    np.fill_diagonal(T5, 0)
    patterns.append((T5, "hotspot5"))
    
    # Pattern 6: Shuffle (random permutation)
    np.random.seed(42)
    perm = np.random.permutation(N)
    T6 = np.zeros((N, N))
    for s in range(N):
        d = perm[s]
        if s != d:
            T6[s, d] = 1.0 / sum(1 for s2 in range(N) if perm[s2] != s2)
    patterns.append((T6, "shuffle"))
    
    # Pattern 7: Bit reversal
    T7 = np.zeros((N, N))
    for s in range(N):
        # Bit reversal for 4-bit address (4x4 mesh)
        sx, sy = s % G, s // G
        # Reverse bits: (x,y) → (y,x) for 2-bit coordinates
        d = sx * G + sy
        if s != d:
            T7[s, d] = 1.0 / 15.0
    patterns.append((T7, "bitrev"))
    
    return patterns


def train_port_score(epochs=2000, lr=3e-4):
    """Train GNN to predict optimal port scores."""
    
    model = GNNPortScore(in_dim=7, hidden_dim=64, embed_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Generate traffic patterns
    patterns = generate_traffic_patterns(G, N)
    
    # Compute target scores for each pattern
    targets = []
    for T, name in patterns:
        scores, L_min = compute_optimal_port_scores(T, G, N)
        targets.append((T, scores, name))
        max_load = L_min.max()
        print(f"  {name:12s}: max_min_load={max_load:.3f}")
    
    print(f"\n{'='*60}")
    print(f"Training Port Score GNN (epochs={epochs}, lr={lr})")
    print(f"{'='*60}")
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        all_scores, emb = model(NODE_FEATURES, EDGE_INDEX, EDGE_ATTR)
        
        # Cross-entropy loss for port scores
        total_loss = 0
        for T, target, name in targets:
            target_t = torch.FloatTensor(target)  # [N, N, 4]
            
            # Softmax over ports
            logits = all_scores  # [N, N, 4]
            
            # KL divergence: minimize KL(target || softmax(logits))
            # First, ensure target is a valid probability distribution
            target_dist = target_t.clone()
            
            # For same-node pairs (cur == dst): uniform target
            for i in range(N):
                target_dist[i, i, :] = 0.25  # uniform
            
            # Normalize to sum to 1
            target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Log-softmax of predictions
            log_probs = F.log_softmax(logits, dim=-1)
            
            # KL divergence
            kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean')
            
            # Additional loss: encourage correct port ordering
            # For each (cur, dst), the best minimal port should have highest score
            logits_softmax = F.softmax(logits, dim=-1)
            
            # MSE between softmax scores and targets (helps ordering)
            mse_loss = F.mse_loss(logits_softmax, target_dist)
            
            # Combined loss
            loss = kl_loss + 0.5 * mse_loss
            total_loss = total_loss + loss
        
        total_loss = total_loss / len(targets)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 200 == 0:
            with torch.no_grad():
                model.eval()
                scores, _ = model(NODE_FEATURES, EDGE_INDEX, EDGE_ATTR)
                scores_np = scores.cpu().numpy()
                
                # Evaluate correctness: at each (cur, dst), is best port a minimal port?
                correct = 0
                total_pairs = 0
                for cur in range(N):
                    for dst in range(N):
                        if cur == dst: continue
                        min_ports = get_minimal_ports(cur, dst, G)
                        if len(min_ports) == 0: continue
                        best_port = np.argmax(scores_np[cur, dst])
                        if best_port in min_ports:
                            correct += 1
                        total_pairs += 1
                
                pct_correct = correct / total_pairs * 100 if total_pairs > 0 else 0
            
            print(f"  Epoch {epoch+1:4d}/{epochs} | loss={total_loss.item():.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f} | "
                  f"minimal_port_acc={pct_correct:.1f}%")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\nBest loss: {best_loss:.6f}")
    
    # Evaluate final model
    with torch.no_grad():
        model.eval()
        final_scores, _ = model(NODE_FEATURES, EDGE_INDEX, EDGE_ATTR)
        scores_np = final_scores.cpu().numpy()
        
        print(f"\nFinal port score evaluation:")
        correct = 0
        total_pairs = 0
        for cur in range(N):
            for dst in range(N):
                if cur == dst: continue
                min_ports = get_minimal_ports(cur, dst, G)
                if len(min_ports) == 0: continue
                best_port = np.argmax(scores_np[cur, dst])
                if best_port in min_ports:
                    correct += 1
                total_pairs += 1
        
        pct_correct = correct / total_pairs * 100 if total_pairs > 0 else 0
        print(f"  Minimal port accuracy: {pct_correct:.1f}% ({correct}/{total_pairs})")
        
        # Show score distribution
        for cur in [0, 5, 10, 15]:
            for dst in [0, 7, 10, 15]:
                if cur != dst:
                    scores_str = ", ".join(f"{scores_np[cur, dst, p]:.3f}" for p in range(4))
                    min_ports = get_minimal_ports(cur, dst, G)
                    print(f"  scores[{cur}][{dst}] = [{scores_str}]  min={min_ports}")
    
    return scores_np, model


# ============================================================
# 6. EXPORT HEADER
# ============================================================
def export_port_score_header(scores, filename, G=4):
    """
    Export N×N×4 port score tensor to C++ header.
    Scores are stored as float arrays.
    """
    N = G * G
    lines = []
    lines.append("// Auto-generated by train_gnn_port_score_v3.py")
    lines.append(f"// Mesh {G}x{G}, generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("// GNN Port Score Routing: 4 scores per (cur,dst) pair")
    lines.append("// Port mapping: 0=E(x+1), 1=W(x-1), 2=S(y+1), 3=N(y-1)")
    lines.append("#ifndef _GNN_PORT_SCORE_ROUTE_4X4_H_")
    lines.append("#define _GNN_PORT_SCORE_ROUTE_4X4_H_")
    lines.append("")
    lines.append('#include "booksim.hpp"')
    lines.append('#include "routefunc.hpp"')
    lines.append('#include "kncube.hpp"')
    lines.append("")
    lines.append("extern int dor_next_mesh(int,int,bool);")
    lines.append("extern int gNumVCs;")
    lines.append("")
    
    # Port score table: scores[cur][dst][4]
    # Output shape: [N][N][4]
    lines.append(f"static const float gnn_port_scores_4x4[{N}][{N}][4]={{{{")
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
    
    # Helper: normalize scores among minimal ports
    lines.append("""
// Helper: check if port is minimal toward destination
static inline bool gnn_is_minimal_port(int cur, int dest, int port, int G=4) {
  int cx = cur % G, cy = cur / G;
  int dx = dest % G, dy = dest / G;
  int nx = cx, ny = cy;
  switch (port) {
    case 0: nx = cx + 1; break; // E
    case 1: nx = cx - 1; break; // W
    case 2: ny = cy + 1; break; // S
    case 3: ny = cy - 1; break; // N
  }
  int cur_dist = abs(dx - cx) + abs(dy - cy);
  int new_dist = abs(dx - nx) + abs(dy - ny);
  return new_dist < cur_dist;
}
""")
    
    # Routing function
    lines.append(f"""
void gnn_port_score_route_4x4_mesh(const Router*r, const Flit*f,
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
  
  // Find minimal ports and pick the one with highest GNN score
  // Also consider congestion: if best-scored port is congested, try alternatives
  float best_score = -1e9f;
  int best_port = -1;
  float second_best_score = -1e9f;
  int second_best_port = -1;
  
  for (int p = 0; p < 4; p++) {{
    if (!gnn_is_minimal_port(cur, dest, p)) continue;
    float score = gnn_port_scores_4x4[cur][dest][p];
    int credit = r->GetUsedCredit(p);
    float congestion_penalty = (float)credit / 16.0f;  // Normalize by max buffer
    float effective_score = score - 0.3f * congestion_penalty;
    
    if (effective_score > best_score) {{
      second_best_score = best_score;
      second_best_port = best_port;
      best_score = effective_score;
      best_port = p;
    }} else if (effective_score > second_best_score) {{
      second_best_score = effective_score;
      second_best_port = p;
    }}
  }}
  
  // Use VC-based deadlock avoidance (2 VC classes)
  int const available_vcs = (vcEnd - vcBegin + 1) / 2;
  assert(available_vcs > 0);
  
  // If the best port is severely congested, try second best
  if (best_port >= 0) {{
    int best_credit = r->GetUsedCredit(best_port);
    if (best_credit > 12 && second_best_port >= 0) {{
      best_port = second_best_port;
    }}
  }}
  
  // Fallback: if no minimal port found (shouldn't happen in practice)
  if (best_port < 0) {{
    best_port = dor_next_mesh(cur, dest, false);
  }}
  
  out_port = best_port;
  
  // VC assignment for deadlock freedom:
  // Use XY VC partition for backward compatibility
  // VC set 0: odd ports (1,3 = West, North) use upper VCs
  // VC set 1: even ports (0,2 = East, South) use lower VCs
  if(out_port == 1 || out_port == 3) {{
    vcBegin = vcBegin + available_vcs;
  }} else {{
    vcEnd = vcBegin + available_vcs - 1;
  }}
  
  outputs->Clear();
  outputs->AddRange(out_port, vcBegin, vcEnd);
}}
""")
    
    lines.append("#endif /* _GNN_PORT_SCORE_ROUTE_4X4_H_ */")
    
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[GNN-PortScore] Header written to {filename}")
    print(f"  Table size: {N}x{N}x4 = {N*N*4} float values")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GNN-Port-Score Routing v3: Training")
    print("=" * 60)
    
    t0 = time.time()
    scores, model = train_port_score(epochs=2000, lr=3e-4)
    print(f"\nTraining time: {time.time()-t0:.1f}s")
    
    # Export to header
    header_path = '/home/opc/.openclaw/workspace/booksim2/src/gnn_port_score_route_4x4.h'
    export_port_score_header(scores, header_path)
    
    # Save scores
    out_dir = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments'
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/gnn_port_scores.npy', scores)
    print(f"Scores saved to {out_dir}/gnn_port_scores.npy")
    
    print(f"\n{'='*60}")
    print("Done! To compile BookSim2:")
    print("  cd /home/opc/.openclaw/workspace/booksim2/src && make -j4")
    print("  (then register in routefunc.cpp if not already)")
    print(f"{'='*60}")
