#!/usr/bin/env python3
"""
GNN-Weighted Adaptive Routing: Improved Training
=================================================
Uses supervised learning: for each traffic pattern, compute optimal routing
decision (XY vs YX) per (src,dst) pair, then train GNN to predict it.

This is more effective than reinforcement learning because:
1. Optimal decisions can be computed analytically
2. No need for differentiable simulator
3. GNN learns to generalize across different traffic patterns

Author: Ngoc Anh for Thay Hieu
Date: 16/05/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
import numpy as np
import os, time, json, math

DEVICE = 'cpu'
print(f"[GNN] Using device: {DEVICE}")

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
# 3. COMPUTE OPTIMAL ROUTING FOR TRAFFIC PATTERNS
# ============================================================
def compute_paths(G=4):
    N = G * G
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

PATHS = compute_paths(G)

def compute_optimal_weights(traffic_matrix, G=4, N=16):
    """
    For each (src,dst) pair, compute the optimal routing weight.
    
    Method: for each pair, compare the max link load contribution
    of XY vs YX and decide which is better.
    """
    # Full load with XY-only
    L_xy = np.zeros((N, 4))
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            for node, port in PATHS[(src, dst)]['xy']:
                L_xy[node, port] += rate
    
    # Full load with YX-only
    L_yx = np.zeros((N, 4))
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            for node, port in PATHS[(src, dst)]['yx']:
                L_yx[node, port] += rate
    
    max_xy = L_xy.max()
    max_yx = L_yx.max()
    
    # Per-pair contribution analysis
    weights = np.ones((N, N)) * 0.5
    np.fill_diagonal(weights, 0.0)
    
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            
            # Check if XY or YX path goes through heavily loaded links
            xy_cong = max([L_xy[node, port] for node, port in PATHS[(src, dst)]['xy']])
            yx_cong = max([L_yx[node, port] for node, port in PATHS[(src, dst)]['yx']])
            
            # Simple heuristic: prefer the path with less congestion
            if xy_cong < yx_cong - 0.05:
                weights[src, dst] = 0.15  # Strong XY
            elif yx_cong < xy_cong - 0.05:
                weights[src, dst] = 0.85  # Strong YX
            else:
                weights[src, dst] = 0.5   # Balanced
    
    return weights, max_xy, max_yx


# ============================================================
# 4. GNN MODEL
# ============================================================
class GNNWeightPredictor(nn.Module):
    """GNN that predicts routing weights W[i][j] for all (src,dst) pairs."""
    
    def __init__(self, in_dim=7, hidden_dim=64, embed_dim=32):
        super().__init__()
        # Encoder
        self.conv1 = GATv2Conv(in_dim, hidden_dim // 4, heads=4, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=1)
        self.conv3 = GATv2Conv(hidden_dim, embed_dim, heads=1, edge_dim=1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.15)
        
        # Decoder (pairwise)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def encode(self, x, edge_index, edge_attr):
        x = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.elu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = self.norm3(self.conv3(x, edge_index, edge_attr))
        return x
    
    def decode(self, embeddings):
        n = embeddings.size(0)
        src = embeddings.unsqueeze(1).expand(n, n, -1)
        dst = embeddings.unsqueeze(0).expand(n, n, -1)
        pairs = torch.cat([src, dst], dim=-1)
        weights = self.decoder(pairs).squeeze(-1)
        return weights
    
    def forward(self, x, edge_index, edge_attr):
        emb = self.encode(x, edge_index, edge_attr)
        weights = self.decode(emb)
        return weights, emb


# ============================================================
# 5. TRAINING
# ============================================================
def train_with_optimal_targets(epochs=2000, lr=3e-4):
    """Train GNN to predict optimal routing weights for multiple traffic patterns."""
    
    model = GNNWeightPredictor(in_dim=7, hidden_dim=64, embed_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Generate training data from multiple traffic patterns
    N = 16
    
    # Pattern 1: Uniform
    T1 = np.ones((N, N)) / (N - 1.0)
    np.fill_diagonal(T1, 0)
    w1, max1_xy, max1_yx = compute_optimal_weights(T1)
    
    # Pattern 2: Hotspot at node 10
    T2 = np.ones((N, N)) * 0.03
    T2[:, 10] = 0.15  # High hotspot traffic
    np.fill_diagonal(T2, 0)
    w2, max2_xy, max2_yx = compute_optimal_weights(T2)
    
    # Pattern 3: Transpose
    T3 = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % G, s // G
        d = sy * G + sx
        if s != d:
            T3[s, d] = 1.0 / 15.0
    w3, max3_xy, max3_yx = compute_optimal_weights(T3)
    
    # Pattern 4: Bit complement (for 4x4, this is (3-x, 3-y))
    T4 = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % G, s // G
        d = (3-sy) * G + (3-sx)
        if s != d:
            T4[s, d] = 1.0 / 15.0
    w4, max4_xy, max4_yx = compute_optimal_weights(T4)
    
    # Pattern 5: Hotspot at node 5 (another center)
    T5 = np.ones((N, N)) * 0.03
    T5[:, 5] = 0.15
    np.fill_diagonal(T5, 0)
    w5, max5_xy, max5_yx = compute_optimal_weights(T5)
    
    targets_list = [
        (T1, w1, "uniform"),
        (T2, w2, "hotspot10"),
        (T3, w3, "transpose"),
        (T4, w4, "bitcomp"),
        (T5, w5, "hotspot5"),
    ]
    
    print(f"{'='*60}")
    print(f"Training Data Generated")
    print(f"{'='*60}")
    max_xy_list = [max1_xy, max2_xy, max3_xy, max4_xy, max5_xy]
    max_yx_list = [max1_yx, max2_yx, max3_yx, max4_yx, max5_yx]
    for i, (T, w, name) in enumerate(targets_list):
        pct_xy = (w < 0.4).mean() * 100
        pct_yx = (w > 0.6).mean() * 100
        pct_mid = 100 - pct_xy - pct_yx
        print(f"  {name:12s}: XY_max={max_xy_list[i]:.3f} YX_max={max_yx_list[i]:.3f} | "
              f"opt: XY={pct_xy:.0f}% Mid={pct_mid:.0f}% YX={pct_yx:.0f}%")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training (epochs={epochs}, lr={lr})")
    print(f"{'='*60}")
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        weights, _ = model(NODE_FEATURES, EDGE_INDEX, EDGE_ATTR)
        
        # MSE loss against optimal targets
        total_loss = 0
        for T, target, name in targets_list:
            target_t = torch.FloatTensor(target)
            
            # MSE with different weights for decisive decisions
            mse = (weights - target_t).pow(2)
            
            # Weight: give more importance to decisive targets (close to 0 or 1)
            # and less to neutral (close to 0.5)
            decisiveness = 1.0 - 2.0 * (target_t - 0.5).abs()
            confidence_weight = 1.0 + 2.0 * (1.0 - decisiveness)  # Weight 1 for decisive, 3 for neutral
            
            # Diagonal should be 0
            diag_penalty = weights.diag().pow(2).mean()
            
            loss = (mse * confidence_weight).mean() + 0.5 * diag_penalty
            total_loss = total_loss + loss
        
        total_loss = total_loss / len(targets_list)
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
                w_pred, _ = model(NODE_FEATURES, EDGE_INDEX, EDGE_ATTR)
                w_np = w_pred.cpu().numpy()
                pct_xy = (w_np < 0.4).mean() * 100
                pct_yx = (w_np > 0.6).mean() * 100
                pct_mid = 100 - pct_xy - pct_yx
            
            print(f"  Epoch {epoch+1:4d}/{epochs} | loss={total_loss.item():.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f} | "
                  f"XY={pct_xy:.0f}% Mid={pct_mid:.0f}% YX={pct_yx:.0f}%")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\nBest loss: {best_loss:.6f}")
    
    # Generate final weights
    with torch.no_grad():
        model.eval()
        final_weights, _ = model(NODE_FEATURES, EDGE_INDEX, EDGE_ATTR)
        w_np = final_weights.cpu().numpy()
        
        print(f"\nFinal weight matrix ({N}x{N}):")
        pct_xy = (w_np < 0.4).mean() * 100
        pct_yx = (w_np > 0.6).mean() * 100
        print(f"  XY={pct_xy:.0f}% Mid={100-pct_xy-pct_yx:.0f}% YX={pct_yx:.0f}%")
        print(f"  Mean={w_np.mean():.3f} Std={w_np.std():.3f}")
        print(f"  Min={w_np[w_np>0].min():.3f} Max={w_np[w_np>0].max():.3f}")
        
        # Show sample
        print(f"\n  Sample (first 5 rows, first 5 cols):")
        np.set_printoptions(precision=3, suppress=True)
        for i in range(min(5, N)):
            print(f"    [{', '.join(f'{v:.3f}' for v in w_np[i, :5])}]")
    
    return w_np, model


# ============================================================
# 6. EXPORT HEADER
# ============================================================
def export_header(weights, filename, G=4, routing_func_name="gnn_weighted_route_4x4_mesh"):
    N = G * G
    lines = []
    lines.append("// Auto-generated by GNN-Weighted Routing supervised training")
    lines.append(f"// Mesh {G}x{G}, generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"// Training: supervised on 5 traffic patterns (uniform/hotspot/transpose/bitcomp/hotspot5)")
    lines.append("#ifndef _GNN_WEIGHTED_ROUTE_H_")
    lines.append("#define _GNN_WEIGHTED_ROUTE_H_")
    lines.append("")
    lines.append('#include "booksim.hpp"')
    lines.append('#include "routefunc.hpp"')
    lines.append('#include "kncube.hpp"')
    lines.append("")
    lines.append("extern int dor_next_mesh(int,int,bool);")
    lines.append("extern int gNumVCs;")
    lines.append("")
    lines.append(f"static const float gnn_weight_table_4x4[{N}][{N}]={{")
    
    for i in range(N):
        row = "  {" + ",".join(f"{v:.6f}f" for v in weights[i]) + "}"
        if i < N - 1:
            row += ","
        lines.append(row)
    
    lines.append("};")
    lines.append("")
    lines.append("#define GNN_ALPHA 0.5f")
    lines.append("#define GNN_BETA  0.3f")
    lines.append("#define GNN_MAX_CREDIT 8")
    lines.append("")
    
    # Routing function code
    lines.append(f"""
void {routing_func_name}(const Router*r, const Flit*f,
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
  int const available_vcs=(vcEnd-vcBegin+1)/2;
  assert(available_vcs>0);
  
  int out_port_xy=dor_next_mesh(cur,dest,false);
  int out_port_yx=dor_next_mesh(cur,dest,true);
  bool x_then_y;
  
  if(in_channel<2*gN){{
    x_then_y=(f->vc<(vcBegin+available_vcs));
  }}else{{
    float w=gnn_weight_table_4x4[cur][dest];
    int credit_xy=r->GetUsedCredit(out_port_xy);
    int credit_yx=r->GetUsedCredit(out_port_yx);
    float norm_credit_xy=(float)credit_xy/(float)GNN_MAX_CREDIT;
    float norm_credit_yx=(float)credit_yx/(float)GNN_MAX_CREDIT;
    float local_congestion=(norm_credit_xy+norm_credit_yx)/2.0f;
    float threshold=GNN_ALPHA+GNN_BETA*(local_congestion-0.5f);
    
    if(w>threshold){{
      if(credit_yx<=credit_xy+3){{x_then_y=false;}}
      else{{x_then_y=true;}}
    }}else if(w<1.0f-threshold){{
      if(credit_xy<=credit_yx+3){{x_then_y=true;}}
      else{{x_then_y=false;}}
    }}else{{
      int diff=credit_xy-credit_yx;
      if(diff>0){{x_then_y=false;}}
      else if(diff<0){{x_then_y=true;}}
      else{{x_then_y=(w<0.5f);}}
    }}
  }}
  
  if(x_then_y){{out_port=out_port_xy;vcEnd=vcBegin+available_vcs-1;}}
  else{{out_port=out_port_yx;vcBegin=vcBegin+available_vcs;}}
  
  outputs->Clear();
  outputs->AddRange(out_port,vcBegin,vcEnd);
}}
""")
    
    lines.append("#endif /* _GNN_WEIGHTED_ROUTE_H_ */")
    
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[GNN] Header written to {filename}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GNN-Weighted Routing: Supervised Training")
    print("=" * 60)
    
    # Train
    t0 = time.time()
    weights, model = train_with_optimal_targets(epochs=2000, lr=3e-4)
    print(f"\nTraining time: {time.time()-t0:.1f}s")
    
    # Export to header
    header_path = '/home/opc/.openclaw/workspace/booksim2/src/gnn_weighted_route_4x4.h'
    export_header(weights, header_path)
    
    # Save weights
    os.makedirs('experiments', exist_ok=True)
    np.save('experiments/gnn_supervised_weights.npy', weights)
    print(f"Weights saved to experiments/gnn_supervised_weights.npy")
    
    print(f"\n{'='*60}")
    print("Done! To use in BookSim2, recompile:")
    print("  cd /home/opc/.openclaw/workspace/booksim2/src && make -j4")
    print(f"{'='*60}")
