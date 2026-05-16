#!/usr/bin/env python3
"""
GNN-Weighted Routing: Mesh 8x8 Training & Zero-Shot Generalization
===================================================================
Two modes:
  1. Direct training on Mesh 8x8
  2. Zero-shot: Train on 4x4, interpolate to 8x8

Author: Ngoc Anh for Thay Hieu
Date: 16/05/2026
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
import os, time, math

DEVICE = 'cpu'
print(f"[GNN] Using device: {DEVICE}")

# ============================================================
# 1. MESH TOPOLOGY BUILDER
# ============================================================
def build_mesh_graph(G=4):
    N = G * G
    edges, edge_attr = [], []
    for y in range(G):
        for x in range(G):
            idx = y * G + x
            if x > 0:
                edges += [[idx, y*G+(x-1)], [y*G+(x-1), idx]]
                edge_attr += [[1.0], [1.0]]
            if y > 0:
                edges += [[idx, (y-1)*G+x], [(y-1)*G+x, idx]]
                edge_attr += [[2.0], [2.0]]
    return (torch.LongTensor(edges).t().contiguous(),
            torch.FloatTensor(edge_attr))

# ============================================================
# 2. NODE FEATURES (position + BC + corner/edge/center)
# ============================================================
def compute_node_features(G=4):
    N = G * G
    features = np.zeros((N, 7))
    for y in range(G):
        for x in range(G):
            idx = y * G + x
            features[idx, 0] = x / (G - 1.0)
            features[idx, 1] = y / (G - 1.0)
            deg = 2 + (x > 0) + (x < G-1) + (y > 0) + (y < G-1)
            features[idx, 2] = deg / 4.0
            bc = (G - x - 1) * x * (G - y - 1) * y * 4.0 / (G**3)
            features[idx, 3] = bc
            corner = (x == 0 or x == G-1) and (y == 0 or y == G-1)
            edge = (x == 0 or x == G-1 or y == 0 or y == G-1) and not corner
            center = not corner and not edge
            features[idx, 4] = 1.0 if corner else 0.0
            features[idx, 5] = 1.0 if edge else 0.0
            features[idx, 6] = 1.0 if center else 0.0
    return torch.FloatTensor(features)

# ============================================================
# 3. GNN MODEL
# ============================================================
class GNNWeightPredictor(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=64, embed_dim=32):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim//4, heads=4, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim//4, heads=4, edge_dim=1)
        self.conv3 = GATv2Conv(hidden_dim, embed_dim, heads=1, edge_dim=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.15)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim*2, 64), nn.LayerNorm(64), nn.LeakyReLU(0.1),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.LeakyReLU(0.1),
            nn.Linear(32, 1), nn.Sigmoid(),
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
        return self.decoder(pairs).squeeze(-1)

    def forward(self, x, edge_index, edge_attr):
        emb = self.encode(x, edge_index, edge_attr)
        return self.decode(emb), emb

# ============================================================
# 4. TRAINING TARGETS (analytical optimal routing)
# ============================================================
def compute_minimal_paths(G=4):
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

def compute_optimal_weights(traffic_matrix, paths, G=4, N=16):
    L_xy = np.zeros((N, 4))
    L_yx = np.zeros((N, 4))
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            for node, port in paths[(src, dst)]['xy']:
                L_xy[node, port] += rate
            for node, port in paths[(src, dst)]['yx']:
                L_yx[node, port] += rate

    weights = np.ones((N, N)) * 0.5
    np.fill_diagonal(weights, 0.0)
    for src in range(N):
        for dst in range(N):
            if src == dst: continue
            rate = traffic_matrix[src, dst]
            if rate == 0: continue
            xy_cong = max([L_xy[node, port] for node, port in paths[(src, dst)]['xy']])
            yx_cong = max([L_yx[node, port] for node, port in paths[(src, dst)]['yx']])
            if xy_cong < yx_cong - 0.05:
                weights[src, dst] = 0.15
            elif yx_cong < xy_cong - 0.05:
                weights[src, dst] = 0.85
            else:
                weights[src, dst] = 0.5
    return weights, L_xy.max(), L_yx.max()

def generate_training_data(G=4, N=16):
    paths = compute_minimal_paths(G)
    targets = []
    # Uniform
    T = np.ones((N, N)) / (N - 1.0); np.fill_diagonal(T, 0)
    w, _, _ = compute_optimal_weights(T, paths, G, N)
    targets.append((T, w, "uniform"))
    # Transpose
    T = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % G, s // G
        d = sy * G + sx
        if s != d: T[s, d] = 1.0 / (N - 1.0)
    w, _, _ = compute_optimal_weights(T, paths, G, N)
    targets.append((T, w, "transpose"))
    # Hotspot center
    T = np.ones((N, N)) * 0.03
    center = (G//2) * G + (G//2)
    T[:, center] = 0.15
    np.fill_diagonal(T, 0)
    w, _, _ = compute_optimal_weights(T, paths, G, N)
    targets.append((T, w, "hotspot_center"))
    # Hotspot corner
    T = np.ones((N, N)) * 0.03
    T[:, 0] = 0.15
    np.fill_diagonal(T, 0)
    w, _, _ = compute_optimal_weights(T, paths, G, N)
    targets.append((T, w, "hotspot_corner"))
    # Bit complement
    T = np.zeros((N, N))
    for s in range(N):
        sx, sy = s % G, s // G
        d = (G-1-sy) * G + (G-1-sx)
        if s != d: T[s, d] = 1.0 / (N - 1.0)
    w, _, _ = compute_optimal_weights(T, paths, G, N)
    targets.append((T, w, "bitcomp"))
    return targets

# ============================================================
# 5. TRAINING LOOP
# ============================================================
def train_model(G, epochs=2000, lr=3e-4):
    N = G * G
    edge_index, edge_attr = build_mesh_graph(G)
    node_features = compute_node_features(G)
    targets_list = generate_training_data(G, N)

    model = GNNWeightPredictor(in_dim=7, hidden_dim=64, embed_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"Training Mesh {G}x{G} ({N} nodes)")
    print(f"{'='*60}")

    best_loss, best_state = float('inf'), None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        weights, _ = model(node_features, edge_index, edge_attr)

        total_loss = 0
        for _, target, name in targets_list:
            target_t = torch.FloatTensor(target)
            mse = (weights - target_t).pow(2)
            decisiveness = 1.0 - 2.0 * (target_t - 0.5).abs()
            confidence_weight = 1.0 + 2.0 * (1.0 - decisiveness)
            diag_penalty = weights.diag().pow(2).mean()
            loss = (mse * confidence_weight).mean() + 0.5 * diag_penalty
            total_loss = total_loss + loss

        total_loss = total_loss / len(targets_list)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                w_pred, _ = model(node_features, edge_index, edge_attr)
                w_np = w_pred.detach().cpu().numpy()
            print(f"  Ep {epoch+1:4d}/{epochs} loss={loss_val:.6f} "
                  f"XY={(w_np<0.4).mean()*100:.0f}% "
                  f"Mid={(w_np[(w_np>=0.4)&(w_np<=0.6)]).size/N/N*100:.0f}% "
                  f"YX={(w_np>0.6).mean()*100:.0f}%")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_weights, _ = model(node_features, edge_index, edge_attr)
        w_np = final_weights.detach().cpu().numpy()

    print(f"\n  Best loss: {best_loss:.6f}")
    print(f"  Final: XY={(w_np<0.4).mean()*100:.0f}% "
          f"YX={(w_np>0.6).mean()*100:.0f}% "
          f"Mean={w_np.mean():.3f} Std={w_np.std():.3f}")

    return w_np, model

# ============================================================
# 6. ZERO-SHOT: Bilinear Interpolation 4x4 -> 8x8
# ============================================================
def interpolate_weights_4x4_to_8x8(w_4x4):
    """
    Extend 16x16 -> 64x64 via bilinear interpolation.
    For each 8x8 (src,dst) pair, map to normalized coords [0,1] and
    bilinearly interpolate in the 4x4 weight space.
    """
    G_small, G_large = 4, 8
    N_large = 64
    w_large = np.zeros((N_large, N_large))

    # Precompute node positions mapped to 4x4 space
    def small_coord(pos, n):
        return pos / (n - 1.0) * (G_small - 1.0)

    for y2 in range(G_large):
        for x2 in range(G_large):
            s_idx = y2 * G_large + x2
            sx = small_coord(x2, G_large)
            sy = small_coord(y2, G_large)
            sx_l = min(max(int(math.floor(sx)), 0), G_small-2)
            sx_h = sx_l + 1
            sy_l = min(max(int(math.floor(sy)), 0), G_small-2)
            sy_h = sy_l + 1
            fx_s = sx - sx_l
            fy_s = sy - sy_l

            for y2d in range(G_large):
                for x2d in range(G_large):
                    d_idx = y2d * G_large + x2d
                    if s_idx == d_idx:
                        w_large[s_idx, d_idx] = 0.0
                        continue

                    dx = small_coord(x2d, G_large)
                    dy = small_coord(y2d, G_large)
                    dx_l = min(max(int(math.floor(dx)), 0), G_small-2)
                    dx_h = dx_l + 1
                    dy_l = min(max(int(math.floor(dy)), 0), G_small-2)
                    dy_h = dy_l + 1
                    fx_d = dx - dx_l
                    fy_d = dy - dy_l

                    # For each of 4 source corners, interpolate destination
                    vals = []
                    for s_row in [sy_l, sy_h]:
                        for s_col in [sx_l, sx_h]:
                            sr = int(s_row)
                            sc = int(s_col)
                            # Get row from 4x4 matrix for this source node
                            row = w_4x4[sr * G_small + sc].reshape(G_small, G_small)
                            # Bilinear interpolation at destination
                            v = (row[dy_l, dx_l] * (1-fx_d) + row[dy_l, dx_h] * fx_d) * (1-fy_d) + \
                                (row[dy_h, dx_l] * (1-fx_d) + row[dy_h, dx_h] * fx_d) * fy_d
                            vals.append(v)

                    v00, v01, v10, v11 = vals
                    v_top = v00 * (1-fx_s) + v01 * fx_s
                    v_bot = v10 * (1-fx_s) + v11 * fx_s
                    w_large[s_idx, d_idx] = v_top * (1-fy_s) + v_bot * fy_s

    print(f"\n{'='*60}")
    print(f"Zero-Shot: 4x4 -> 8x8 (Bilinear Interp)")
    print(f"{'='*60}")
    print(f"  XY={(w_large<0.4).mean()*100:.0f}% "
          f"Mid={(w_large[(w_large>=0.4)&(w_large<=0.6)]).size/N_large/N_large*100:.0f}% "
          f"YX={(w_large>0.6).mean()*100:.0f}%")
    print(f"  Mean={w_large.mean():.3f} Std={w_large.std():.3f}")
    return w_large

def nearest_neighbor_extension_4x4_to_8x8(w_4x4):
    """Simpler: nearest neighbor mapping."""
    G_small, G_large = 4, 8
    N_large = 64
    def map4(x, y):
        nx = round(x / (G_large-1.0) * (G_small-1.0))
        ny = round(y / (G_large-1.0) * (G_small-1.0))
        return min(max(int(ny), 0), 3) * G_small + min(max(int(nx), 0), 3)
    w = np.ones((N_large, N_large)) * 0.5
    for y2 in range(G_large):
        for x2 in range(G_large):
            s4 = map4(x2, y2)
            for y2d in range(G_large):
                for x2d in range(G_large):
                    d4 = map4(x2d, y2d)
                    w[y2*G_large+x2, y2d*G_large+x2d] = w_4x4[s4, d4]
    np.fill_diagonal(w, 0.0)
    return w

# ============================================================
# 7. EXPORT HEADER
# ============================================================
def export_header(weights, G, filename, training_note=""):
    N = G * G
    tag = f"{G}x{G}"
    table_name = f"gnn_weight_table_{tag}"
    func_name = f"gnn_weighted_route_{tag}_mesh"
    guard = f"_GNN_WEIGHTED_ROUTE_{G}X{G}_H_"

    lines = []
    lines.append(f"// Auto-generated by train_gnn_weighted_v2_scalable.py")
    lines.append(f"// Mesh {tag}, generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if training_note:
        lines.append(f"// {training_note}")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append('')
    lines.append('#include "booksim.hpp"')
    lines.append('#include "routefunc.hpp"')
    lines.append('#include "kncube.hpp"')
    lines.append('')
    lines.append("extern int dor_next_mesh(int,int,bool);")
    lines.append("extern int gNumVCs;")
    lines.append('')

    w_c = np.clip(weights, 0.01, 0.99)
    lines.append(f"static const float {table_name}[{N}][{N}]={{")
    for i in range(N):
        row = "  {" + ",".join(f"{v:.6f}f" for v in w_c[i]) + "}"
        if i < N - 1: row += ","
        lines.append(row)
    lines.append("};")
    lines.append('')
    lines.append("#define GNN_ALPHA 0.5f")
    lines.append("#define GNN_BETA  0.3f")
    lines.append("#define GNN_MAX_CREDIT 8")
    lines.append('')

    lines.append(f"""
void {func_name}(const Router*r, const Flit*f,
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
    float w={table_name}[cur][dest];
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

    lines.append(f"#endif /* {guard} */")
    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as f:
        f.write(content)
    size_kb = len(content) / 1024
    print(f"[EXPORT] {filename} ({N}x{N}={N*N} entries, {size_kb:.0f}KB)")
    return content

# ============================================================
# 8. MAIN
# ============================================================
if __name__ == '__main__':
    import sys
    BOOKSIM_SRC = '/home/opc/.openclaw/workspace/booksim2/src'
    EXP_DIR = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments'

    print("=" * 60)
    print("GNN-Weighted Routing: Scalable Training Pipeline")
    print("=" * 60)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode in ('train_4x4', 'all'):
        print("\n>>> STEP 1: Training Mesh 4x4")
        t0 = time.time()
        w_4x4, model_4 = train_model(G=4, epochs=2000, lr=3e-4)
        print(f"[TIME] 4x4: {time.time()-t0:.1f}s")
        export_header(w_4x4, 4, f"{BOOKSIM_SRC}/gnn_weighted_route_4x4.h",
                      training_note="Training: supervised on 5 traffic patterns for Mesh 4x4")
        np.save(f"{EXP_DIR}/weights_4x4.npy", w_4x4)
        w_4x4_prev = w_4x4

    if mode in ('train_8x8', 'all'):
        print("\n>>> STEP 2: Training Mesh 8x8 (direct)")
        t0 = time.time()
        w_8x8_direct, model_8 = train_model(G=8, epochs=2000, lr=3e-4)
        print(f"[TIME] 8x8: {time.time()-t0:.1f}s")
        export_header(w_8x8_direct, 8, f"{BOOKSIM_SRC}/gnn_weighted_route_8x8.h",
                      training_note="Training: supervised on 5 traffic patterns for Mesh 8x8")
        np.save(f"{EXP_DIR}/weights_8x8_direct.npy", w_8x8_direct)

    if mode in ('zeroshot', 'all'):
        print("\n>>> STEP 3: Zero-Shot (4x4 -> 8x8)")
        if 'w_4x4' not in locals():
            p = f"{EXP_DIR}/weights_4x4.npy"
            if os.path.exists(p):
                w_4x4 = np.load(p)
                print(f"[ZERO] Loaded from {p}")
            else:
                print("[ZERO] Training 4x4 first...")
                w_4x4, _ = train_model(G=4, epochs=2000)
                np.save(p, w_4x4)

        w_z = interpolate_weights_4x4_to_8x8(w_4x4)
        export_header(w_z, 8, f"{BOOKSIM_SRC}/gnn_weighted_route_8x8.h",
                      training_note="Zero-shot: bilinear interpolation from Mesh 4x4 supervised training")
        np.save(f"{EXP_DIR}/weights_8x8_zeroshot.npy", w_z)
        np.save(f"{EXP_DIR}/weights_8x8.npy", w_z)

        # Also NN version as separate file
        w_nn = nearest_neighbor_extension_4x4_to_8x8(w_4x4)
        np.save(f"{EXP_DIR}/weights_8x8_zeroshot_nn.npy", w_nn)

    if mode in ('analyze', 'all'):
        print("\n>>> STEP 4: Analysis")
        for fname in ['weights_8x8_direct.npy', 'weights_8x8_zeroshot.npy']:
            fpath = f"{EXP_DIR}/{fname}"
            if os.path.exists(fpath):
                w = np.load(fpath)
                print(f"  {fname}: XY={(w<0.4).mean()*100:.1f}% "
                      f"YX={(w>0.6).mean()*100:.1f}% "
                      f"Mean={w.mean():.3f}")

        d = f"{EXP_DIR}/weights_8x8_direct.npy"
        z = f"{EXP_DIR}/weights_8x8_zeroshot.npy"
        if os.path.exists(d) and os.path.exists(z):
            wd = np.load(d)
            wz = np.load(z)
            diff = wd - wz
            print(f"\n  Direct vs Zero-Shot comparison:")
            print(f"    MAE={np.abs(diff).mean():.4f}")
            print(f"    MaxAE={np.abs(diff).max():.4f}")
            print(f"    Pearson r={np.corrcoef(wd.ravel(), wz.ravel())[0,1]:.4f}")
            same_dec = (np.sign(wd-0.5)==np.sign(wz-0.5)).mean()*100
            print(f"    Same decision: {same_dec:.1f}%")

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
