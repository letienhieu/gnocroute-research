#!/usr/bin/env python3
"""
Fine-tune GNN Port Score on 8x8 Mesh (v5)
=========================================
Load v4 model (trained on 4x4) and fine-tune for 8x8.

Key changes from zero-shot v4:
1. Fine-tune node embeddings + port decoders for 8x8 topology
2. Training data: 8 traffic patterns on 8x8 (uniform, transpose, 
   hotspot, bit_reversal, shuffle, mixed)
3. Soft target scores from load-aware simulation
4. Architecture identical to v4 (same model class) but learned on 8x8

Author: Ngoc Anh for Thay Hieu
Date: 17/05/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
import os, time, json, math, sys

DEVICE = 'cpu'
print(f"[GNN-PortScore-v5] Using device: {DEVICE}")

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
G = 8
N = G * G  # 64

# Port mapping: 0=E(x+1), 1=W(x-1), 2=S(y+1), 3=N(y-1)
PORT_NAMES = ['E', 'W', 'S', 'N']

EXP_DIR = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments'


# ============================================================
# 1. MESH TOPOLOGY (generic for any G)
# ============================================================

def build_mesh_graph(G_val=G):
    """Build edge_index and edge_attr for G_val×G_val mesh."""
    edges = []
    edge_attr = []
    for y in range(G_val):
        for x in range(G_val):
            idx = y * G_val + x
            if x > 0:
                edges.append([idx, y * G_val + (x-1)])
                edges.append([y * G_val + (x-1), idx])
                edge_attr.append([1.0])
                edge_attr.append([1.0])
            if y > 0:
                edges.append([idx, (y-1) * G_val + x])
                edges.append([(y-1) * G_val + x, idx])
                edge_attr.append([2.0])
                edge_attr.append([2.0])
    edge_index = torch.LongTensor(edges).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attr)
    return edge_index, edge_attr


EDGE_INDEX_8x8, EDGE_ATTR_8x8 = build_mesh_graph(G)


def get_minimal_ports(cur, dst, G_val=G):
    """Return list of minimal ports for (cur, dst)."""
    cx, cy = cur % G_val, cur // G_val
    dx, dy = dst % G_val, dst // G_val
    minimal = []
    if dx > cx: minimal.append(0)
    if dx < cx: minimal.append(1)
    if dy > cy: minimal.append(2)
    if dy < cy: minimal.append(3)
    return minimal


def get_next_node(cur, port, G_val=G):
    """Get neighbor node reached by taking port from cur."""
    x, y = cur % G_val, cur // G_val
    if port == 0:
        return cur + 1 if x < G_val - 1 else -1
    elif port == 1:
        return cur - 1 if x > 0 else -1
    elif port == 2:
        return cur + G_val if y < G_val - 1 else -1
    elif port == 3:
        return cur - G_val if y > 0 else -1
    return -1


def is_at_edge(node, port, G_val=G):
    """Check if port at node exits the mesh."""
    x, y = node % G_val, node // G_val
    if port == 0:
        return x >= G_val - 1
    elif port == 1:
        return x <= 0
    elif port == 2:
        return y >= G_val - 1
    elif port == 3:
        return y <= 0
    return True


# ============================================================
# 2. NODE FEATURES FOR 8x8 (12-dim, same as v4)
# ============================================================

def compute_node_features_8x8(G_val=G, faulty_links=None):
    """
    Compute 12-dim node features for 8x8 mesh.
    
    Features (same as v4):
        0: x / (G_val-1)
        1: y / (G_val-1)
        2: active_degree / 4.0
        3: APPROXIMATE betweenness centrality
        4: corner flag
        5: edge flag
        6: center flag
        7: active_neighbors / 4.0
        8-11: per-port active status (E, W, S, N)
    """
    if faulty_links is None:
        faulty_links = set()
    
    N_val = G_val * G_val
    features = np.zeros((N_val, 12))
    
    for y in range(G_val):
        for x in range(G_val):
            idx = y * G_val + x
            
            # Position (normalized to [0,1])
            features[idx, 0] = x / max(G_val-1, 1)
            features[idx, 1] = y / max(G_val-1, 1)
            
            # Active degree
            active_deg = 0
            for p in range(4):
                if not is_at_edge(idx, p, G_val) and (idx, p) not in faulty_links:
                    active_deg += 1
            features[idx, 2] = active_deg / 4.0
            
            # Approximate betweenness centrality
            dx_center = min(x, G_val-1-x)
            dy_center = min(y, G_val-1-y)
            approx_bc = (dx_center + 1) * (dy_center + 1) / ((G_val//2)**2)
            approx_bc = min(approx_bc, 1.0)
            features[idx, 3] = approx_bc
            
            # Position type
            corner = (x == 0 or x == G_val-1) and (y == 0 or y == G_val-1)
            edge = (x == 0 or x == G_val-1 or y == 0 or y == G_val-1) and not corner
            center_flag = not corner and not edge
            features[idx, 4] = 1.0 if corner else 0.0
            features[idx, 5] = 1.0 if edge else 0.0
            features[idx, 6] = 1.0 if center_flag else 0.0
            
            # Active neighbors count / 4
            features[idx, 7] = active_deg / 4.0
            
            # Per-port active status
            for p in range(4):
                if not is_at_edge(idx, p, G_val) and (idx, p) not in faulty_links:
                    features[idx, 8 + p] = 1.0
                else:
                    features[idx, 8 + p] = 0.0
    
    return torch.FloatTensor(features)


# ============================================================
# 3. MODEL (identical to v4 architecture)
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
# 4. LOAD TRAINING DATA
# ============================================================

def load_training_data():
    """Load pre-computed 8x8 training targets."""
    
    targets_path = f'{EXP_DIR}/gnn_8x8_targets_v5.npy'
    labels_path = f'{EXP_DIR}/gnn_8x8_labels_v5.npy'
    meta_path = f'{EXP_DIR}/gnn_8x8_training_metadata_v5.json'
    
    targets = np.load(targets_path)        # [8, 64, 64, 4]
    labels = np.load(labels_path)           # [8, 64, 64]
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print(f"[Data] Loaded {len(meta['pattern_names'])} patterns:")
    for name in meta['pattern_names']:
        print(f"  - {name}")
    print(f"[Data] Targets shape: {targets.shape}")
    print(f"[Data] Labels shape: {labels.shape}")
    
    return targets, labels, meta['pattern_names']


# ============================================================
# 5. FINE-TUNE ON 8x8
# ============================================================

def fine_tune_v5(epochs=100, lr=1e-4, seed=42):
    """Fine-tune v4 model on 8x8 data."""
    
    print(f"\n{'='*70}")
    print(f"GNN-PortScore v5: Fine-tune on 8x8 Mesh")
    print(f"{'='*70}")
    
    t0 = time.time()
    
    # Load training data
    targets_array, labels_array, pattern_names = load_training_data()
    num_patterns = len(pattern_names)
    
    # Build node features for 8x8 (fault-free)
    node_features = compute_node_features_8x8(G, set())
    print(f"\n[Model] Node features: {node_features.shape}")
    
    # Load pretrained v4 model (4x4 trained)
    v4_model_path = f'{EXP_DIR}/gnn_port_score_v4_model.pt'
    print(f"[Model] Loading pretrained v4 model from {v4_model_path}...")
    
    model = GNNPortScoreFaultAware(in_dim=12, hidden_dim=64, embed_dim=32)
    v4_state = torch.load(v4_model_path, map_location=DEVICE)
    model.load_state_dict(v4_state)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Loaded v4 model: {total_params:,} params")
    
    # Verify minimal port accuracy on 8x8 before fine-tuning
    with torch.no_grad():
        model.eval()
        scores_before, _ = model(node_features, EDGE_INDEX_8x8, EDGE_ATTR_8x8)
        scores_np = scores_before.cpu().numpy()
        
        correct = 0
        total = 0
        for cur in range(N):
            for dst in range(N):
                if cur == dst: continue
                min_ports = get_minimal_ports(cur, dst, G)
                if not min_ports: continue
                best = int(np.argmax(scores_np[cur, dst]))
                if best in min_ports:
                    correct += 1
                total += 1
        acc_before = correct / total * 100 if total > 0 else 0
        print(f"[Model] Pre-finetune minimal port accuracy: {correct}/{total} = {acc_before:.2f}%")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Fine-tune
    print(f"\n[Training] Fine-tuning {epochs} epochs, lr={lr}...")
    print(f"{'='*70}")
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        total_loss = 0.0
        
        # Forward pass once (node features don't change)
        logits, embeddings = model(node_features, EDGE_INDEX_8x8, EDGE_ATTR_8x8)
        
        # Compute loss against all training patterns
        for i in range(num_patterns):
            target_t = torch.FloatTensor(targets_array[i])
            
            # Set diagonal to uniform (self->self doesn't exist)
            for cur in range(N):
                target_t[cur, cur, :] = 0.25
            target_t = target_t / (target_t.sum(dim=-1, keepdim=True) + 1e-10)
            
            # KL divergence + CrossEntropy
            log_probs = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, target_t, reduction='batchmean')
            
            # MSE on softmax probabilities
            logits_softmax = F.softmax(logits, dim=-1)
            mse_loss = F.mse_loss(logits_softmax, target_t)
            
            # Cross-entropy with hard labels (optimal port)
            labels_t = torch.LongTensor(labels_array[i])
            labels_t[labels_t < 0] = 0  # replace -1 (diagonal) with 0
            # Reshape logits: [64, 64, 4] -> [4096, 4]
            ce_loss = F.cross_entropy(
                logits.reshape(-1, 4), 
                labels_t.reshape(-1),
                reduction='mean'
            )
            
            # Combined loss
            loss = kl_loss + 0.5 * mse_loss + 0.5 * ce_loss
            total_loss += loss
        
        avg_loss = total_loss / num_patterns
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track best model
        current_loss = avg_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                model.eval()
                vscores, _ = model(node_features, EDGE_INDEX_8x8, EDGE_ATTR_8x8)
                vnp = vscores.cpu().numpy()
                
                correct_m = 0
                total_m = 0
                for cur in range(N):
                    for dst in range(N):
                        if cur == dst: continue
                        min_ports = get_minimal_ports(cur, dst, G)
                        if not min_ports: continue
                        best = int(np.argmax(vnp[cur, dst]))
                        if best in min_ports:
                            correct_m += 1
                        total_m += 1
                acc_v = correct_m / total_m * 100 if total_m > 0 else 0
            
            print(f"  Epoch {epoch+1:4d}/{epochs} | loss={current_loss:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f} | ff_acc={acc_v:.1f}%")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n[Best model] Loaded from best loss: {best_loss:.6f}")
    
    # Final evaluation
    with torch.no_grad():
        model.eval()
        print(f"\n{'='*70}")
        print(f"Final Evaluation")
        print(f"{'='*70}")
        
        final_scores, _ = model(node_features, EDGE_INDEX_8x8, EDGE_ATTR_8x8)
        scores_np = final_scores.cpu().numpy()
        
        # Minimal port accuracy
        correct_f = 0
        total_f = 0
        for cur in range(N):
            for dst in range(N):
                if cur == dst: continue
                min_ports = get_minimal_ports(cur, dst, G)
                if not min_ports: continue
                best = int(np.argmax(scores_np[cur, dst]))
                if best in min_ports:
                    correct_f += 1
                total_f += 1
        print(f"  Fault-free minimal port acc: {correct_f}/{total_f} = {correct_f/total_f*100:.1f}%")
        
        # Label matching accuracy (hard labels)
        all_correct = 0
        all_total = 0
        for i in range(num_patterns):
            correct_l = 0
            for cur in range(N):
                for dst in range(N):
                    if cur == dst: continue
                    optimal = int(labels_array[i, cur, dst])
                    predicted = int(np.argmax(scores_np[cur, dst]))
                    if optimal == predicted:
                        correct_l += 1
            all_correct += correct_l
            all_total += N * (N - 1)
            print(f"  Pattern '{pattern_names[i]}': label match = {correct_l}/{N*(N-1)} = {correct_l/(N*(N-1))*100:.1f}%")
        
        # Compare with v4 (zero-shot) if available
        v4_scores_path = f'{EXP_DIR}/gnn_port_scores_8x8_v4.npy'
        if os.path.exists(v4_scores_path):
            v4_scores = np.load(v4_scores_path)
            
            # How many (cur,dst) pairs have different best ports?
            diff_count = 0
            for cur in range(N):
                for dst in range(N):
                    if cur == dst: continue
                    best_v4 = int(np.argmax(v4_scores[cur, dst]))
                    best_v5 = int(np.argmax(scores_np[cur, dst]))
                    if best_v4 != best_v5:
                        diff_count += 1
            print(f"\n  v4 vs v5: {diff_count}/{N*(N-1)} pairs changed best port ({diff_count/(N*(N-1))*100:.1f}%)")
        
        # Sample scores
        print(f"\n  Sample port scores:")
        sample_pairs = [(0, 63), (27, 36), (7, 56), (9, 21), (28, 35)]
        for cur, dst in sample_pairs:
            s_str = ", ".join(f"{scores_np[cur, dst, p]:.3f}" for p in range(4))
            min_ports = get_minimal_ports(cur, dst, G)
            best = int(np.argmax(scores_np[cur, dst]))
            print(f"    [{cur:2d}]→[{dst:2d}]: [{s_str}]  min={min_ports} best={PORT_NAMES[best]}")
    
    return model, scores_np


# ============================================================
# 6. EXPORT C++ HEADER (v5)
# ============================================================

def export_port_score_header_v5(scores, filename, G_val=G):
    """Export 64×64×4 port score tensor to C++ header for 8x8 mesh (v5)."""
    N_val = G_val * G_val
    lines = []
    lines.append("// Auto-generated by train_gnn_port_score_8x8_v5.py")
    lines.append(f"// Mesh {G_val}x{G_val}, generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("// GNN Port Score Routing v5: Fine-tuned on 8x8")
    lines.append("// Port mapping: 0=E(x+1), 1=W(x-1), 2=S(y+1), 3=N(y-1)")
    lines.append("// Node features: 12-dim (7 spatial + 5 fault-aware), fine-tuned from v4")
    lines.append("#ifndef _GNN_PORT_SCORE_ROUTE_8X8_V5_H_")
    lines.append("#define _GNN_PORT_SCORE_ROUTE_8X8_V5_H_")
    lines.append("")
    lines.append('#include "booksim.hpp"')
    lines.append('#include "routefunc.hpp"')
    lines.append('#include "kncube.hpp"')
    lines.append("")
    lines.append("extern int dor_next_mesh(int,int,bool);")
    lines.append("extern int gNumVCs;")
    lines.append("")
    
    lines.append(f"static const float gnn_port_scores_8x8_v5[{N_val}][{N_val}][4]={{")
    for cur in range(N_val):
        lines.append(f"  {{  // cur={cur}")
        for dst in range(N_val):
            row = "    {" + ",".join(f"{scores[cur, dst, p]:.6f}f" for p in range(4)) + "}"
            if cur < N_val - 1 or dst < N_val - 1:
                row += ","
            lines.append(row)
        if cur < N_val - 1:
            lines.append("  },")
        else:
            lines.append("  }")
    lines.append("};")
    lines.append("")
    
    helpers = f"""
// Helper: check if port is minimal toward destination
static inline bool gnn_is_minimal_port_v5(int cur, int dest, int port) {{
  int G_PARAM = {G_val};
  int cx = cur % G_PARAM, cy = cur / G_PARAM;
  int dx = dest % G_PARAM, dy = dest / G_PARAM;
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
void gnn_port_score_route_8x8_v5_mesh(const Router*r, const Flit*f,
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
    if (!gnn_is_minimal_port_v5(cur, dest, p)) continue;
    if (r->IsFaultyOutput(p)) continue;
    
    float score = gnn_port_scores_8x8_v5[cur][dest][p];
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
    int cx = cur % {G_val}, cy = cur / {G_val};
    int dx = dest % {G_val}, dy = dest / {G_val};
    
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
    lines.append("#endif /* _GNN_PORT_SCORE_ROUTE_8X8_V5_H_ */")
    
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    
    table_size_bytes = N_val * N_val * 4 * 4
    print(f"[Header] Written to {filename}")
    print(f"  Table size: {N_val}x{N_val}x4 = {N_val*N_val*4} float values ({table_size_bytes/1024:.1f} KB)")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("GNN-Port-Score v5: Fine-tune on 8x8 Mesh")
    print("=" * 70)
    
    t0 = time.time()
    
    # Fine-tune
    model_v5, scores_v5 = fine_tune_v5(epochs=100, lr=1e-4, seed=42)
    
    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s")
    
    # Save scores
    np.save(f'{EXP_DIR}/gnn_port_scores_8x8_v5.npy', scores_v5)
    print(f"[GNN-PortScore-v5] Scores saved to {EXP_DIR}/gnn_port_scores_8x8_v5.npy")
    
    # Export C++ header
    header_path = '/home/opc/.openclaw/workspace/booksim2/src/gnn_port_score_route_8x8_v5.h'
    export_port_score_header_v5(scores_v5, header_path)
    
    # Save model
    torch.save(model_v5.state_dict(), f'{EXP_DIR}/gnn_port_score_v5_model.pt')
    print(f"[GNN-PortScore-v5] Model saved to {EXP_DIR}/gnn_port_score_v5_model.pt")
    
    print(f"\n{'='*70}")
    print(f"Done! To compile BookSim2:")
    print(f"  1. Register in routefunc.cpp:")
    print(f"     add '#include \"gnn_port_score_route_8x8_v5.h\"'")
    print(f"     add 'gRoutingFunctionMap[\"gnn_port_score_route_8x8_v5_mesh\"] = &gnn_port_score_route_8x8_v5_mesh;'")
    print(f"  2. cd /home/opc/.openclaw/workspace/booksim2/src && make -j4")
    print(f"{'='*70}")
