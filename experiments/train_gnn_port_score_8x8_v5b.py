#!/usr/bin/env python3
"""
GNN Port Score v5b: Fine-tune for Score Differentiation (vectorized)
====================================================================
Key: CE + margin loss (vectorized) to preserve good score gaps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np, os, time

DEVICE = 'cpu'
G, N = 8, 64
EXP_DIR = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa/experiments'

# Import the original model class from v5 training script
import sys
sys.path.insert(0, EXP_DIR)
os.environ['PYTORCH_JIT'] = '0'
from train_gnn_port_score_8x8_v5 import GNNPortScoreFaultAware as Model

# Precompute minimal port mask: [64, 64, 4] bool
MIN_MASK = np.zeros((N, N, 4), dtype=bool)
for cur in range(N):
    for dst in range(N):
        if cur == dst: continue
        cx, cy = cur % G, cur // G
        dx, dy = dst % G, dst // G
        if dx > cx: MIN_MASK[cur,dst,0] = True
        if dx < cx: MIN_MASK[cur,dst,1] = True
        if dy > cy: MIN_MASK[cur,dst,2] = True
        if dy < cy: MIN_MASK[cur,dst,3] = True
MIN_MASK_T = torch.BoolTensor(MIN_MASK)

def build_mesh_graph(G_val=G):
    edges, attr = [], []
    for y in range(G_val):
        for x in range(G_val):
            idx = y * G_val + x
            if x > 0:
                edges += [[idx, y*G_val+x-1], [y*G_val+x-1, idx]]
                attr += [[1.0], [1.0]]
            if y > 0:
                edges += [[idx, (y-1)*G_val+x], [(y-1)*G_val+x, idx]]
                attr += [[2.0], [2.0]]
    ei = torch.LongTensor(edges).t().contiguous()
    ea = torch.FloatTensor(attr)
    return ei, ea
EDGE_INDEX, EDGE_ATTR = build_mesh_graph()

def node_features(G_val=G):
    Nv = G_val * G_val
    f = np.zeros((Nv, 12))
    for y in range(G_val):
        for x in range(G_val):
            i = y * G_val + x
            f[i,0] = x / (G_val-1); f[i,1] = y / (G_val-1)
            d = (x>0)+(x<G_val-1)+(y>0)+(y<G_val-1)
            f[i,2] = d / 4.0
            bc = (min(x,G_val-1-x)+1)*(min(y,G_val-1-y)+1)/((G_val//2)**2)
            f[i,3] = min(bc, 1.0)
            c = (x==0 or x==G_val-1) and (y==0 or y==G_val-1)
            e = ((x==0 or x==G_val-1 or y==0 or y==G_val-1) and not c)
            f[i,4]=c; f[i,5]=e; f[i,6]=not(c or e); f[i,7]=d/4.0
            f[i,8]=float(x<G_val-1); f[i,9]=float(x>0)
            f[i,10]=float(y<G_val-1); f[i,11]=float(y>0)
    return torch.FloatTensor(f)



def finetune(epochs=200):
    print(f"\n{'='*60}")
    print("v5b: CE + margin loss (vectorized)")
    print(f"{'='*60}")
    
    labels = np.load(f'{EXP_DIR}/gnn_8x8_labels_v5.npy')
    nf = node_features()
    
    model = Model(in_dim=12, hidden_dim=64, embed_dim=32)
    model.load_state_dict(torch.load(f'{EXP_DIR}/gnn_port_score_v4_model.pt'))
    
    # Pre-finetune diff
    with torch.no_grad():
        model.eval()
        s = model(nf, EDGE_INDEX, EDGE_ATTR)[0].numpy()
        good = sum(1 for cur in range(N) for dst in range(N) if cur!=dst and 
                   (lambda mp: len(mp)>=2 and max(s[cur,dst,p] for p in mp)-min(s[cur,dst,p] for p in mp)>0.5)([p for p in range(4) if MIN_MASK[cur,dst,p]]))
        tot = sum(1 for cur in range(N) for dst in range(N) if cur!=dst and sum(MIN_MASK[cur,dst])>=2)
        print(f"  Pre-finetune diff>0.5: {good}/{tot} ({good/tot*100:.1f}%)")
    
    # CE is fast - compute per-pattern
    num_patterns = 8
    
    # Phase 1: pure CE
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5)
    best_state, best_loss = None, float('inf')
    
    MIN_MASK_EX = MIN_MASK_T.unsqueeze(0)  # [1,64,64,4]
    
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(nf, EDGE_INDEX, EDGE_ATTR)[0]  # [64,64,4]
        
        total_loss = 0.0
        for i in range(num_patterns):
            lt = torch.LongTensor(labels[i]); lt[lt<0] = 0
            total_loss += F.cross_entropy(logits.reshape(-1,4), lt.reshape(-1))
        
        # Vectorized margin loss: apply only to pairs where MIN_MASK has >=2 true
        # For each (cur,dst), find min score among MIN ports, and keep it close to max score... 
        # Actually we want CORRECT label's score to be higher than INCORRECT minimal ports
        # This is hard to vectorize. Let's use a simpler approach:
        # Just encourage the TOP minimal port score to be at least 0.5 above the bottom minimal port score
        # Among pairs where MIN_MASK >= 2 ports active
        
        n_min_ports = MIN_MASK_T.sum(dim=-1)  # [64,64]
        mask_2plus = (n_min_ports >= 2).float()  # [64,64]
        count_2plus = mask_2plus.sum()
        
        if count_2plus > 0:
            # For each (cur,dst) with >=2 minimal ports
            # Get scores for minimal ports only
            logits_masked = logits * MIN_MASK_T.float() + (-1e9) * (~MIN_MASK_T).float()
            max_min = logits_masked.max(dim=-1)[0]  # [64,64]
            
            # For min, set non-minimal to huge positive number
            logits_masked_inv = logits * MIN_MASK_T.float() + 1e9 * (~MIN_MASK_T).float()
            min_min = logits_masked_inv.min(dim=-1)[0]  # [64,64]
            
            # Gap target: at least 0.5
            gap = max_min - min_min
            margin_loss = (mask_2plus * torch.clamp(0.5 - gap, min=0)).sum() / count_2plus
            total_loss += 0.2 * margin_loss
        
        avg_loss = total_loss / num_patterns
        avg_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if avg_loss.item() < best_loss:
            best_loss = avg_loss.item()
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        
        if (ep+1)%25==0:
            with torch.no_grad():
                model.eval()
                s = model(nf, EDGE_INDEX, EDGE_ATTR)[0].numpy()
                good = sum(1 for cur in range(N) for dst in range(N) if cur!=dst and 
                          (lambda mp: len(mp)>=2 and max(s[cur,dst,p] for p in mp)-min(s[cur,dst,p] for p in mp)>0.5)(
                              [p for p in range(4) if MIN_MASK[cur,dst,p]]))
                print(f"  Ep {ep+1:3d} loss={avg_loss.item():.4f} diff>{'0.5'}: {good}/{tot} ({good/tot*100:.1f}%)")
    
    model.load_state_dict(best_state)
    with torch.no_grad():
        model.eval()
        s = model(nf, EDGE_INDEX, EDGE_ATTR)[0].numpy()
    return model, s

if __name__ == '__main__':
    t0 = time.time()
    model, scores = finetune(200)
    np.save(f'{EXP_DIR}/gnn_port_scores_8x8_v5b.npy', scores)
    
    # Export header (same function name gnn_port_score_route_8x8_v5_mesh for compatibility)
    from train_gnn_port_score_8x8_v5 import export_port_score_header_v5
    export_port_score_header_v5(scores, '/home/opc/.openclaw/workspace/booksim2/src/gnn_port_score_route_8x8_v5.h', 8)
    torch.save(model.state_dict(), f'{EXP_DIR}/gnn_port_score_v5b_model.pt')
    print(f"Total: {time.time()-t0:.1f}s")
