#!/usr/bin/env python3
"""
GNN-NOC Experiment: Graph Neural Networks for NoC Topology Analysis
====================================================================
Muc tieu:
  1. Xay dung NoC topologies (Mesh, Torus, Fat-Tree) duoi dang PyG Data objects
  2. Chay GCN/GAT encoder, so sanh node embeddings voi Betweenness Centrality
  3. Do inference latency tren CPU
  4. Danh gia kha nang GNN learning topology structure

Author: Ngoc Anh for Thay Hieu
Date: 14/05/2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np
import time
import json
import os, sys, math
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/home/opc/.openclaw/workspace/research"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# 1. BUILD NOC TOPOLOGIES
# ═══════════════════════════════════════════════════════════════

def build_mesh(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    return G

def build_torus(rows, cols):
    G = nx.grid_2d_graph(rows, cols, periodic=True)
    return G

def build_fat_tree(k=4):
    G = nx.Graph()
    half = k // 2
    for pod in range(k):
        for switch in range(half):
            G.add_node(f"l0_p{pod}_s{switch}", layer=0)
    for agg in range(half):
        for pod in range(k):
            node = f"l1_a{agg}"
            G.add_node(node, layer=1)
            for sw in range(half):
                G.add_edge(f"l0_p{pod}_s{sw}", node)
    core_count = half * half
    for core in range(core_count):
        node = f"l2_c{core}"
        G.add_node(node, layer=2)
        for agg in range(half):
            G.add_edge(f"l1_a{agg}", node)
    return G

def build_smallworld(n, k, p):
    return nx.watts_strogatz_graph(n, k, p, seed=42)

# ═══════════════════════════════════════════════════════════════
# 2. NETWORKX → PYG CONVERTER
# ═══════════════════════════════════════════════════════════════

def nx_to_pyg(G_nx):
    """Convert NetworkX graph to PyG Data object with node features"""
    # Node mapping
    nodes = list(G_nx.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Edge index (undirected)
    edge_index = []
    for u, v in G_nx.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        edge_index.append([i, j])
        edge_index.append([j, i])  # undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Node features: degree (one-hot-ish), normalized
    degrees = torch.tensor([float(G_nx.degree(n)) for n in nodes])
    deg_norm = degrees / degrees.max()  # [0,1]
    
    # Add positional features: for mesh/torus, use row/col
    node_features = []
    for n in nodes:
        feat = [deg_norm[node_to_idx[n]].item()]
        # Try to extract position for grid graphs
        if isinstance(n, tuple) and len(n) == 2:
            feat.extend([n[0]/10.0, n[1]/10.0])  # normalized position
        else:
            feat.extend([0.0, 0.0])
        node_features.append(feat)
    
    x = torch.tensor(node_features, dtype=torch.float)

    # Ground truth: Betweenness Centrality
    bc = nx.betweenness_centrality(G_nx)
    y = torch.tensor([bc[n] for n in nodes], dtype=torch.float).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, y=y, 
                num_nodes=n_nodes, name=G_nx.graph.get('name', 'unknown'))

# ═══════════════════════════════════════════════════════════════
# 3. GNN MODELS
# ═══════════════════════════════════════════════════════════════

class GCNEncoder(nn.Module):
    """GCN-based NoC node encoder"""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if n_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_dim))
        else:
            self.convs = nn.ModuleList([GCNConv(in_dim, out_dim)])
        self.n_layers = n_layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x  # Node embeddings


class GATEncoder(nn.Module):
    """GAT-based NoC node encoder"""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        # First layer: multi-head
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
        # Intermediate layers
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
        # Output layer: single head
        if n_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1, concat=False))
        else:
            self.convs = nn.ModuleList([GATConv(in_dim, out_dim, heads=1, concat=False)])
        self.n_layers = n_layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x  # Node embeddings

# ═══════════════════════════════════════════════════════════════
# 4. EXPERIMENT 1: Embedding-BC Correlation
# ═══════════════════════════════════════════════════════════════

def experiment_correlation():
    """Check if GNN node embeddings encode BC-like information"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: GNN Embedding vs Betweenness Centrality")
    print("="*60)

    # Build topologies
    topologies = {
        "Mesh 4x4": build_mesh(4, 4),
        "Mesh 8x8": build_mesh(8, 8),
        "Torus 4x4": build_torus(4, 4),
        "SmallWorld 36": build_smallworld(36, 4, 0.15),
    }

    results = []

    for name, G_nx in topologies.items():
        print(f"\n  --- {name} ---")
        data = nx_to_pyg(G_nx).to(DEVICE)
        
        models = {
            "GCN": GCNEncoder(data.x.size(1), 32, 16, 2).to(DEVICE),
            "GAT": GATEncoder(data.x.size(1), 16, 16, 2, heads=4).to(DEVICE),
        }

        for model_name, model in models.items():
            # Forward pass (untrained — just structural encoding)
            with torch.no_grad():
                embeddings = model(data)
            
            # Compute correlation between embedding dimensions and BC
            bc_vals = data.y.cpu().numpy().flatten()
            embeddings_np = embeddings.cpu().numpy()
            
            # Best dimension correlation
            correlations = []
            for d in range(embeddings_np.shape[1]):
                corr = np.corrcoef(embeddings_np[:, d], bc_vals)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            max_corr = max(correlations) if correlations else 0
            avg_corr = np.mean(correlations) if correlations else 0
            
            print(f"    {model_name:6s}: max|r|={max_corr:.4f}, avg|r|={avg_corr:.4f}, " +
                  f"nodes={data.num_nodes}, emb_dim={embeddings_np.shape[1]}")
            
            results.append({
                "topology": name,
                "model": model_name,
                "num_nodes": data.num_nodes,
                "max_corr": round(max_corr, 4),
                "avg_corr": round(avg_corr, 4),
                "emb_dim": embeddings_np.shape[1],
            })
    
    return results


# ═══════════════════════════════════════════════════════════════
# 5. EXPERIMENT 2: Train GNN to Predict BC
# ═══════════════════════════════════════════════════════════════

def experiment_bc_prediction():
    """Train GNN to predict Betweenness Centrality from topology"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: GNN trained to predict BC (supervised)")
    print("="*60)

    # Train on Mesh 4x4, test on Mesh 8x8 and Torus 4x4
    train_topos = {"Mesh 4x4": build_mesh(4, 4)}
    test_topos = {
        "Mesh 8x8": build_mesh(8, 8),
        "Torus 4x4": build_torus(4, 4),
        "SmallWorld 36": build_smallworld(36, 4, 0.15),
    }

    # Prepare training data
    train_data_list = []
    for name, G in train_topos.items():
        data = nx_to_pyg(G).to(DEVICE)
        train_data_list.append(data)
    
    # Use one topology for training
    train_data = train_data_list[0]
    
    results = []
    
    for model_class, model_name in [(GCNEncoder, "GCN"), (GATEncoder, "GAT")]:
        model = model_class(train_data.x.size(1), 32, 1, 2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Train
        model.train()
        for epoch in range(500):
            optimizer.zero_grad()
            out = model(train_data)
            loss = F.mse_loss(out, train_data.y)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 100 == 0:
                print(f"    {model_name} epoch {epoch+1}: loss={loss.item():.6f}")
        
        model.eval()
        
        # Test on train topology
        with torch.no_grad():
            pred = model(train_data)
            train_r2 = 1 - F.mse_loss(pred, train_data.y).item() / train_data.y.var().item()
        
        print(f"\n    {model_name} trained on Mesh 4x4:")
        print(f"      Train (Mesh 4x4): R2={train_r2:.4f}")
        
        # Test on other topologies (generalization)
        for test_name, G_test in test_topos.items():
            test_data = nx_to_pyg(G_test).to(DEVICE)
            with torch.no_grad():
                pred = model(test_data)
                mse = F.mse_loss(pred, test_data.y).item()
                # R2 score (can be negative if model is worse than mean)
                var = test_data.y.var().item()
                r2 = 1 - mse / var if var > 0 else 0
            
            print(f"      Test ({test_name}): MSE={mse:.6f}, R2={r2:.4f}")
            
            # Pearson correlation between prediction and ground truth
            pred_np = pred.cpu().numpy().flatten()
            gt_np = test_data.y.cpu().numpy().flatten()
            corr = np.corrcoef(pred_np, gt_np)[0, 1] if len(pred_np) > 1 else 0
            
            results.append({
                "train_on": "Mesh 4x4",
                "model": model_name,
                "test_on": test_name,
                "test_nodes": test_data.num_nodes,
                "mse": round(mse, 6),
                "r2": round(r2, 4),
                "pearson_r": round(corr, 4),
            })
    
    return results


# ═══════════════════════════════════════════════════════════════
# 6. EXPERIMENT 3: Inference Latency
# ═══════════════════════════════════════════════════════════════

def experiment_latency():
    """Measure GNN inference latency on CPU"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: GNN Inference Latency on CPU")
    print("="*60)

    topo_sizes = [
        ("Mesh 4x4", build_mesh(4, 4)),
        ("Mesh 8x8", build_mesh(8, 8)),
        ("Mesh 16x16", build_mesh(16, 16)),
        ("Torus 8x8", build_torus(8, 8)),
    ]

    results = []
    n_warmup = 50
    n_runs = 500

    for name, G_nx in topo_sizes:
        data = nx_to_pyg(G_nx).to(DEVICE)
        in_dim = data.x.size(1)
        
        for model_class, model_name in [(GCNEncoder, "GCN"), (GATEncoder, "GAT")]:
            # Try different hidden sizes
            for hidden_dim in [16, 32, 64]:
                if model_name == "GAT":
                    model = model_class(in_dim, hidden_dim // 4, 16, 2, heads=4).to(DEVICE)
                else:
                    model = model_class(in_dim, hidden_dim, 16, 2).to(DEVICE)
                
                # Warmup
                for _ in range(n_warmup):
                    _ = model(data)
                
                # Measure
                if DEVICE.type == 'cpu':
                    torch.cuda.synchronize if hasattr(torch.cuda, 'synchronize') else lambda: None
                
                start = time.perf_counter()
                for _ in range(n_runs):
                    _ = model(data)
                elapsed = time.perf_counter() - start
                
                avg_ms = (elapsed / n_runs) * 1000
                avg_us = avg_ms * 1000
                
                # Estimate cycles (assuming 1 GHz = 1 ns/cycle)
                # Cortex-A72 typically runs at 1.5 GHz
                # 1 us = 1500 cycles at 1.5 GHz
                est_cycles_at_1GHz = avg_us * 1000  # cycles at 1GHz
                est_cycles_at_15GHz = avg_us * 1500  # cycles at 1.5GHz
                
                print(f"    {model_name:6s} h={hidden_dim:3d} | {name:12s}: "
                      f"{avg_ms:.3f} ms = {avg_us:.1f} us, "
                      f"~{est_cycles_at_1GHz:.0f} cyc@1GHz")
                
                results.append({
                    "topology": name,
                    "num_nodes": data.num_nodes,
                    "model": model_name,
                    "hidden_dim": hidden_dim,
                    "avg_ms": round(avg_ms, 4),
                    "avg_us": round(avg_us, 1),
                    "est_cycles_1ghz": round(est_cycles_at_1GHz, 0),
                    "est_cycles_15ghz": round(est_cycles_at_15GHz, 0),
                })

    return results


# ═══════════════════════════════════════════════════════════════
# 7. EXPERIMENT 4: Scalability
# ═══════════════════════════════════════════════════════════════

def experiment_scalability():
    """GNN inference time vs topology size"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Scalability Analysis")
    print("="*60)

    sizes = [4, 6, 8, 10, 12, 16]
    results = []

    n_warmup = 20
    n_runs = 200

    for size in sizes:
        G_nx = build_mesh(size, size)
        data = nx_to_pyg(G_nx).to(DEVICE)
        
        # GCN with hidden=32
        model = GCNEncoder(data.x.size(1), 32, 16, 2).to(DEVICE)
        
        for _ in range(n_warmup):
            _ = model(data)
        
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(data)
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / n_runs) * 1_000_000
        
        nodes = data.num_nodes
        edges = data.edge_index.size(1) // 2
        
        print(f"    Mesh {size:2d}x{size:2d}: nodes={nodes:4d}, edges={edges:4d}, "
              f"inference={avg_us:.1f} us, O(nodes)={avg_us/nodes:.3f} us/node")
        
        results.append({
            "mesh_size": size,
            "num_nodes": nodes,
            "num_edges": edges,
            "inference_us": round(avg_us, 1),
            "us_per_node": round(avg_us/nodes, 3),
        })

    return results


# ═══════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GNN-NOC EXPERIMENTS")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {DEVICE}")

    all_results = {}
    
    # Exp 1: Embedding vs BC correlation
    corr_results = experiment_correlation()
    all_results["embedding_correlation"] = corr_results
    
    # Exp 2: BC prediction
    pred_results = experiment_bc_prediction()
    all_results["bc_prediction"] = pred_results
    
    # Exp 3: Latency
    lat_results = experiment_latency()
    all_results["inference_latency"] = lat_results
    
    # Exp 4: Scalability
    scal_results = experiment_scalability()
    all_results["scalability"] = scal_results

    # Save all results
    out_path = os.path.join(OUT_DIR, "gnn-experiment-results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All results saved: {out_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n--- Embedding-BC Correlation ---")
    for r in corr_results:
        print(f"  {r['model']:6s} on {r['topology']:15s}: max|r|={r['max_corr']:.4f}")
    
    print("\n--- BC Prediction (trained on Mesh 4x4) ---")
    for r in pred_results:
        print(f"  {r['model']:6s} on {r['test_on']:15s}: R2={r['r2']:.4f}, r={r['pearson_r']:.4f}")
    
    print("\n--- Inference Latency (GCN h=32 on mesh) ---")
    for r in lat_results:
        if r['model'] == 'GCN' and r['hidden_dim'] == 32:
            print(f"  {r['topology']:12s}: {r['avg_us']:8.1f} us = {r['est_cycles_1ghz']:6.0f} cyc@1GHz")
    
    print("\n--- Scalability ---")
    for r in scal_results:
        print(f"  Mesh {r['mesh_size']}x{r['mesh_size']}: {r['inference_us']:.1f} us, {r['us_per_node']:.3f} us/node")
