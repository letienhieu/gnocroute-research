#!/usr/bin/env python3
"""
Zero-shot GNN Port Score for 8x8 Mesh using v4 Fault-Aware Model
=================================================================
Load the 4x4-trained GNN (12-dim features) and generate port scores
for an 8x8 Mesh, proving zero-shot generalization capability.

Approach:
1. Build 8x8 mesh topology (edge_index, edge_attr)
2. Compute 12-dim node features for 8x8 (position-normalized, approximate betweenness)
3. Load GNNPortScoreFaultAware state dict
4. Forward pass → 64×64×4 score table
5. Export C++ routing header + numpy scores
6. Print minimal-port accuracy stats

Author: Ngoc Anh for Thay Hieu
Date: 16/05/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
import os, sys, math, time

# Add parent to path so we can import the model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = 'cpu'
print(f"[zeroshot-8x8] Using device: {DEVICE}")

# ============================================================
# 1. MODEL ARCHITECTURE (identical to v4 training code)
# ============================================================

class GNNPortScoreFaultAware(nn.Module):
    """GNN Port Score model với fault-aware features (12-dim input)."""
    
    def __init__(self, in_dim=12, hidden_dim=64, embed_dim=32):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim // 4, heads=4, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=1)
        self.conv3 = GATv2Conv(hidden_dim, embed_dim, heads=1, edge_dim=1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.15)
        
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
        n = embeddings.size(0)
        scores = torch.zeros(n, n, 4)
        cur_emb = embeddings.unsqueeze(1).expand(n, n, -1)
        dst_emb = embeddings.unsqueeze(0).expand(n, n, -1)
        pairs = torch.cat([cur_emb, dst_emb], dim=-1)
        for port in range(4):
            port_scores = self.port_decoders[port](pairs).squeeze(-1)
            scores[:, :, port] = port_scores
        return scores
    
    def forward(self, x, edge_index, edge_attr):
        emb = self.encode(x, edge_index, edge_attr)
        scores = self.decode_scores(emb)
        return scores, emb


# ============================================================
# 2. MESH TOPOLOGY FUNCTIONS (generic for any G)
# ============================================================

def build_mesh_graph(G):
    """Build edge_index and edge_attr for G×G mesh."""
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


def compute_node_features_8x8(G=8, faulty_links=None):
    """
    Compute 12-dim node features for 8x8 mesh.
    
    Features (same as v4):
        0: x / (G-1)
        1: y / (G-1)
        2: active_degree / 4.0
        3: APPROXIMATE betweenness centrality (position-based)
        4: corner flag
        5: edge flag
        6: center flag
        7: active_neighbors / 4.0
        8-11: per-port active status (E, W, S, N)
    
    Betweenness approximation for 8x8:
    - Use the 4x4 trained values as a reference, but since 8x8 is larger,
      we approximate based on position. The model is expected to be robust
      to betweenness approximations since it was trained on 4x4 only.
    - For nodes on the perimeter: lower betweenness
    - For nodes in the center: higher betweenness
    - Roughly: betweenness ~ (dx_from_edge)*(dy_from_edge) / max_bc
    """
    if faulty_links is None:
        faulty_links = set()
    
    N = G * G
    features = np.zeros((N, 12))
    
    # Approximate betweenness based on Manhattan distance from edges
    # 8x8 mesh: center nodes (3,4) in x,y have highest betweenness
    max_betweenness = 1.0  # normalized
    
    for y in range(G):
        for x in range(G):
            idx = y * G + x
            
            # Position (normalized to [0,1])
            features[idx, 0] = x / max(G-1, 1)
            features[idx, 1] = y / max(G-1, 1)
            
            # Active degree
            active_deg = 0
            for p in range(4):
                is_at_edge_flag = (p == 0 and x >= G-1) or (p == 1 and x <= 0) or \
                                  (p == 2 and y >= G-1) or (p == 3 and y <= 0)
                if not is_at_edge_flag and (idx, p) not in faulty_links:
                    active_deg += 1
            features[idx, 2] = active_deg / 4.0
            
            # Approximate betweenness centrality
            # Center nodes have more shortest paths passing through
            dx_from_center = min(x, G-1-x)
            dy_from_center = min(y, G-1-y)
            approx_bc = (dx_from_center + 1) * (dy_from_center + 1) / ((G//2)**2)
            approx_bc = min(approx_bc, 1.0)
            features[idx, 3] = approx_bc
            
            # Position type
            corner = (x == 0 or x == G-1) and (y == 0 or y == G-1)
            edge = (x == 0 or x == G-1 or y == 0 or y == G-1) and not corner
            center = not corner and not edge
            features[idx, 4] = 1.0 if corner else 0.0
            features[idx, 5] = 1.0 if edge else 0.0
            features[idx, 6] = 1.0 if center else 0.0
            
            # Active neighbors count
            active_neighbors = active_deg  # same since edge nodes have fewer neighbors
            features[idx, 7] = active_neighbors / 4.0
            
            # Per-port active status
            for p in range(4):
                is_at_edge_flag = (p == 0 and x >= G-1) or (p == 1 and x <= 0) or \
                                  (p == 2 and y >= G-1) or (p == 3 and y <= 0)
                if not is_at_edge_flag and (idx, p) not in faulty_links:
                    features[idx, 8 + p] = 1.0
                else:
                    features[idx, 8 + p] = 0.0
    
    return torch.FloatTensor(features)


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


def is_at_edge(node, port, G=8):
    """Check if port at node exits the mesh."""
    x, y = node % G, node // G
    if port == 0:  # E
        return x >= G - 1
    elif port == 1:  # W
        return x <= 0
    elif port == 2:  # S
        return y >= G - 1
    elif port == 3:  # N
        return y <= 0
    return True


# ============================================================
# 3. MAIN: LOAD MODEL + GEN 8x8 SCORES
# ============================================================

def main():
    t0 = time.time()
    
    # --- Config ---
    G_src = 4  # trained on 4x4
    G_dst = 8  # zero-shot target
    N_dst = G_dst * G_dst  # 64
    
    exp_dir = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments'
    model_path = f'{exp_dir}/gnn_port_score_v4_model.pt'
    output_scores = f'{exp_dir}/gnn_port_scores_8x8_v4.npy'
    header_path = '/home/opc/.openclaw/workspace/booksim2/src/gnn_port_score_route_8x8_v4.h'
    
    # --- Build 8x8 topology ---
    print(f"[zeroshot-8x8] Building {G_dst}x{G_dst} mesh...")
    edge_index, edge_attr = build_mesh_graph(G_dst)
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Edge attr shape: {edge_attr.shape}")
    print(f"  Nodes: {N_dst}, Edges: {edge_index.shape[1] // 2} (bidirectional)")
    
    # --- Compute 8x8 node features (12-dim) ---
    print(f"[zeroshot-8x8] Computing 12-dim features for {G_dst}x{G_dst}...")
    node_features = compute_node_features_8x8(G_dst, set())  # fault-free
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Feature range: [{node_features.min():.3f}, {node_features.max():.3f}]")
    
    # --- Load model (trained on 4x4, 12-dim input) ---
    print(f"[zeroshot-8x8] Loading model from {model_path}...")
    model = GNNPortScoreFaultAware(in_dim=12, hidden_dim=64, embed_dim=32)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} params")
    
    # --- Forward pass on 8x8 ---
    print(f"[zeroshot-8x8] Forward pass on 64 nodes...")
    with torch.no_grad():
        embeddings = model.encode(node_features, edge_index, edge_attr)
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # Decode all (cur, dst) pairs → [64, 64, 4]
        scores = model.decode_scores(embeddings)
        scores_np = scores.cpu().numpy()
    
    print(f"  Scores shape: {scores_np.shape}")
    print(f"  Score range: [{scores_np.min():.4f}, {scores_np.max():.4f}]")
    
    # --- Validate minimal port accuracy (fault-free) ---
    print(f"\n[zeroshot-8x8] Validating minimal port accuracy...")
    correct = 0
    total = 0
    for cur in range(N_dst):
        for dst in range(N_dst):
            if cur == dst:
                continue
            min_ports = get_minimal_ports(cur, dst, G_dst)
            if len(min_ports) == 0:
                continue
            best_p = int(np.argmax(scores_np[cur, dst]))
            if best_p in min_ports:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"  Minimal port accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    # --- Show sample scores ---
    print(f"\n  Sample port scores (8x8):")
    sample_nodes = [0, 7, 8, 15, 27, 28, 35, 36, 63]
    for cur in [0, 9, 21, 36, 63]:
        for dst in [0, 27, 36, 63]:
            if cur != dst:
                scores_str = ", ".join(f"{scores_np[cur, dst, p]:.3f}" for p in range(4))
                min_ports = get_minimal_ports(cur, dst, G_dst)
                print(f"    [{cur:2d}]→[{dst:2d}]: [{scores_str}]  min={min_ports}")
    
    # --- Save numpy scores ---
    np.save(output_scores, scores_np)
    print(f"\n[zeroshot-8x8] Scores saved to {output_scores}")
    
    # --- Export C++ header ---
    print(f"\n[zeroshot-8x8] Exporting C++ header...")
    export_port_score_header_8x8(scores_np, header_path, G_dst)
    
    print(f"\n[zeroshot-8x8] Done! Total time: {time.time()-t0:.1f}s")
    return scores_np


def export_port_score_header_8x8(scores, filename, G=8):
    """Export 64×64×4 port score tensor to C++ header for 8x8 mesh."""
    N = G * G
    lines = []
    lines.append("// Auto-generated by gen_8x8_scores.py")
    lines.append(f"// Mesh {G}x{G}, generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("// GNN Port Score Routing v4: Zero-shot 4x4→8x8")
    lines.append("// Port mapping: 0=E(x+1), 1=W(x-1), 2=S(y+1), 3=N(y-1)")
    lines.append("// Node features: 12-dim (7 spatial + 5 fault-aware), GNN trained on 4x4")
    lines.append("#ifndef _GNN_PORT_SCORE_ROUTE_8X8_V4_H_")
    lines.append("#define _GNN_PORT_SCORE_ROUTE_8X8_V4_H_")
    lines.append("")
    lines.append('#include "booksim.hpp"')
    lines.append('#include "routefunc.hpp"')
    lines.append('#include "kncube.hpp"')
    lines.append("")
    lines.append("extern int dor_next_mesh(int,int,bool);")
    lines.append("extern int gNumVCs;")
    lines.append("")
    
    lines.append(f"static const float gnn_port_scores_8x8_v4[{N}][{N}][4]={{")
    for cur in range(N):
        lines.append(f"  {{  // cur={cur}")
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
    
    helpers = f"""
// Helper: check if port is minimal toward destination
static inline bool gnn_is_minimal_port_v4_8x8(int cur, int dest, int port) {{
  int G = 8;
  int cx = cur % G, cy = cur / G;
  int dx = dest % G, dy = dest / G;
  int nx = cx, ny = cy;
  switch (port) {{
    case 0: nx = cx + 1; break;
    case 1: nx = cx - 1; break;
    case 2: ny = cy + 1; break;
    case 3: ny = cy - 1; break;
  }}
  int cur_dist = abs(dx - cx) + abs(dy - cy);
  int new_dist = abs(dx - nx) + abs(dy - ny);
  return new_dist < cur_dist;
}}
"""
    lines.append(helpers.strip())
    lines.append("")
    
    routing_func = f"""
void gnn_port_score_route_8x8_v4_mesh(const Router*r, const Flit*f,
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
  if(r->GetID()==f->dest){{out_port=2*G;outputs->Clear();outputs->AddRange(out_port,vcBegin,vcEnd);return;}}
  
  int cur=r->GetID();
  int dest=f->dest;
  
  float best_score = -1e9f;
  int best_port = -1;
  float second_score = -1e9f;
  int second_port = -1;
  
  // G=8 constant
  const int G_param = 8;
  
  for (int p = 0; p < 4; p++) {{
    if (!gnn_is_minimal_port_v4_8x8(cur, dest, p)) continue;
    if (r->IsFaultyOutput(p)) continue;
    
    float score = gnn_port_scores_8x8_v4[cur][dest][p];
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
  
  // Fallback: XY/YX DOR with fault avoidance
  if (best_port < 0) {{
    int cx = cur % 8, cy = cur / 8;
    int dx = dest % 8, dy = dest / 8;
    
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
  
  // Ultimate fallback: DOR without fault checking
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
    lines.append("#endif /* _GNN_PORT_SCORE_ROUTE_8X8_V4_H_ */")
    
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    
    # Calculate table size
    table_size_bytes = N * N * 4 * 4  # 64*64*4*4 = 65,536 bytes
    print(f"[Header] Written to {filename}")
    print(f"  Table size: {N}x{N}x4 = {N*N*4} float values ({table_size_bytes/1024:.1f} KB)")


if __name__ == '__main__':
    main()
