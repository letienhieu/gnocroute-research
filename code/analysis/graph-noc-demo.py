#!/usr/bin/env python3
"""
Graph-NOC Demo: Ứng dụng lý thuyết đồ thị cho tối ưu Network-on-Chip
===================================================================
Author : Ngọc Anh (AI Assistant) for Thầy Hiếu
Purpose: Nghiên cứu cơ chế định tuyến thích nghi cho NoC dựa trên AI
Date   : 2026-05-14

Hiển thị cách dùng Graph Theory (NetworkX) để:
  1. Xây dựng các topology NoC (Mesh, Torus, Ring, Fat-Tree, Butterfly)
  2. Đo lường metrics: diameter, avg path length, degree, centrality
  3. Mô phỏng định tuyến XY vs Shortest Path
  4. Gợi ý hướng ứng dụng GNN cho tối ưu NoC
"""

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import os
from collections import defaultdict

# ── Global ──
OUT_DIR = "/home/opc/.openclaw/workspace/research"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. XÂY DỰNG NOC TOPOLOGIES DƯỚI DẠNG GRAPH
# ═══════════════════════════════════════════════════════════════

def build_mesh(rows=4, cols=4):
    """2D Mesh: mỗi node kết nối với 4 hướng (tối đa)"""
    G = nx.grid_2d_graph(rows, cols)
    # Đặt tên node dạng (r,c)
    G.graph["name"] = f"Mesh {rows}x{cols}"
    G.graph["topo_type"] = "mesh"
    return G

def build_torus(rows=4, cols=4):
    """2D Torus: Mesh + wrap-around edges"""
    G = nx.grid_2d_graph(rows, cols, periodic=True)
    G.graph["name"] = f"Torus {rows}x{cols}"
    G.graph["topo_type"] = "torus"
    return G

def build_ring(n=16):
    """Ring topology"""
    G = nx.cycle_graph(n)
    G.graph["name"] = f"Ring n={n}"
    G.graph["topo_type"] = "ring"
    return G

def build_fat_tree(k=4):
    """
    Fat-Tree topology (k-ary n-tree, 3-layer Clos network)
    Dùng trong nhiều NoC và Datacenter.
    - Layer 0 (leaf): k pod, mỗi pod có k/2 switch
    - Layer 1,2: k/2 spine switches mỗi layer
    """
    G = nx.Graph()
    G.graph["name"] = f"Fat-Tree k={k}"
    G.graph["topo_type"] = "fat_tree"

    half = k // 2
    # Layer 0: access switches (k pods × k/2 switches)
    for pod in range(k):
        for switch in range(half):
            node = f"l0_p{pod}_s{switch}"
            G.add_node(node, layer=0, pod=pod)

    # Layer 1: aggregation switches (k/2)
    for agg in range(half):
        for pod in range(k):
            node = f"l1_a{agg}"
            G.add_node(node, layer=1)
            # Connect to each access switch in pod
            for sw in range(half):
                leaf = f"l0_p{pod}_s{sw}"
                G.add_edge(leaf, node)

    # Layer 2: core switches (k/2)^2
    core_count = half * half
    for core in range(core_count):
        node = f"l2_c{core}"
        G.add_node(node, layer=2)
        # Connect to all aggregation switches
        for agg in range(half):
            agg_node = f"l1_a{agg}"
            G.add_edge(agg_node, node)

    return G

def build_butterfly(n_ports=4, n_stages=3):
    """
    Butterfly topology dạng k-ary n-fly
    Mỗi stage có n_ports^(n_stages-1) nodes
    """
    G = nx.DiGraph()
    G.graph["name"] = f"Butterfly {n_ports}-ary {n_stages}-fly"
    G.graph["topo_type"] = "butterfly"

    nodes_per_stage = n_ports ** (n_stages - 1)
    for stage in range(n_stages):
        for node_idx in range(nodes_per_stage):
            node = f"s{stage}_n{node_idx}"
            G.add_node(node, stage=stage)

    # Connect between stages using butterfly pattern
    for stage in range(n_stages - 1):
        for node_idx in range(nodes_per_stage):
            src = f"s{stage}_n{node_idx}"
            for port in range(n_ports):
                # Butterfly connection pattern
                block_size = n_ports ** (n_stages - stage - 1)
                dst_idx = (node_idx // block_size) * block_size * n_ports + port * block_size + (node_idx % block_size)
                # Avoid out-of-range
                if dst_idx < nodes_per_stage:
                    dst = f"s{stage+1}_n{dst_idx}"
                    G.add_edge(src, dst)

    return G

def build_random_noc(n=20, p=0.12):
    """Random graph for baseline comparison (Erdos-Renyi)"""
    G = nx.erdos_renyi_graph(n, p, seed=42)
    G.graph["name"] = f"Random n={n}, p={p}"
    G.graph["topo_type"] = "random"
    return G

def build_smallworld_noc(n=20, k=4, p=0.15):
    """Small-world graph (Watts-Strogatz) - mô phỏng NoC có shortcut"""
    G = nx.watts_strogatz_graph(n, k, p, seed=42)
    G.graph["name"] = f"Small-World n={n}, k={k}, p={p}"
    G.graph["topo_type"] = "smallworld"
    return G


# ═══════════════════════════════════════════════════════════════
# 2. PHÂN TÍCH GRAPH METRICS CHO NOC
# ═══════════════════════════════════════════════════════════════

def analyze_noc_topology(G):
    """Phân tích các metrics quan trọng cho NoC"""
    metrics = {}

    # Cơ bản
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()

    if G.is_directed():
        metrics["avg_degree"] = 2 * metrics["num_edges"] / metrics["num_nodes"]
    else:
        metrics["avg_degree"] = 2 * metrics["num_edges"] / metrics["num_nodes"]

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    metrics["min_degree"] = min(degrees)
    metrics["max_degree"] = max(degrees)
    metrics["degree_std"] = np.std(degrees)

    # Connectivity
    if not G.is_directed():
        metrics["is_connected"] = nx.is_connected(G)
    else:
        undirected = G.to_undirected()
        metrics["is_connected"] = nx.is_connected(undirected)

    # Đường kính (quan trọng: latency tối đa)
    if G.number_of_nodes() < 2000:
        try:
            if G.is_directed() and not metrics["is_connected"]:
                metrics["diameter"] = float("inf")
                metrics["avg_shortest_path"] = float("inf")
            else:
                G_use = G.to_undirected() if G.is_directed() else G
                if nx.is_connected(G_use):
                    metrics["diameter"] = nx.diameter(G_use)
                    metrics["avg_shortest_path"] = nx.average_shortest_path_length(G_use)
                else:
                    # Largest connected component
                    largest = max(nx.connected_components(G_use), key=len)
                    subg = G_use.subgraph(largest)
                    metrics["diameter"] = nx.diameter(subg)
                    metrics["avg_shortest_path"] = nx.average_shortest_path_length(subg)
                    metrics["lcc_fraction"] = len(largest) / G.number_of_nodes()
        except:
            metrics["diameter"] = -1
            metrics["avg_shortest_path"] = -1
    else:
        # Với graph lớn, sample
        metrics["diameter"] = "large_graph_sampled"
        metrics["avg_shortest_path"] = "large_graph_sampled"

    # Betweenness centrality (quan trọng cho traffic congestion)
    if G.number_of_nodes() < 1000:
        if G.is_directed():
            bc = nx.betweenness_centrality(G, normalized=True)
        else:
            bc = nx.betweenness_centrality(G, normalized=True)
        metrics["avg_betweenness"] = np.mean(list(bc.values()))
        metrics["max_betweenness"] = max(bc.values())
        metrics["betweenness_std"] = np.std(list(bc.values()))
        # Top nodes có betweenness cao nhất (dễ congestion)
        top_k = min(5, G.number_of_nodes())
        top_nodes = sorted(bc.items(), key=lambda x: -x[1])[:top_k]
        metrics["top_betweenness_nodes"] = [(str(n), round(v, 4)) for n, v in top_nodes]

    metric_names = {
        "num_nodes": "Số nodes", "num_edges": "Số links",
        "avg_degree": "Bậc TB", "min_degree": "Bậc nhỏ nhất",
        "max_degree": "Bậc lớn nhất", "degree_std": "Độ lệch chuẩn bậc",
        "is_connected": "Liên thông", "diameter": "Đường kính (hops)",
        "avg_shortest_path": "Đường đi TB (hops)",
        "avg_betweenness": "Betweenness TB", "max_betweenness": "Betweenness max",
        "betweenness_std": "Std Betweenness",
        "top_betweenness_nodes": "Top nodes dễ nghẽn",
        "lcc_fraction": "Phần trăm LCC"
    }

    return metrics, bc if G.number_of_nodes() < 1000 else None


# ═══════════════════════════════════════════════════════════════
# 3. MÔ PHỎNG ĐỊNH TUYẾN
# ═══════════════════════════════════════════════════════════════

def xy_routing_mesh(r1, c1, r2, c2):
    """XY Deterministic Routing: đi X trước, Y sau"""
    path = [(r1, c1)]
    # Đi theo cột (X) trước
    step_c = 1 if c2 > c1 else -1
    for c in range(c1, c2, step_c):
        path.append((r1, c + step_c))
    # Đi theo hàng (Y) sau
    step_r = 1 if r2 > r1 else -1
    for r in range(r1, r2, step_r):
        path.append((r + step_r, c2))
    return path

def simulate_routing(G, routing_fn=None, num_pairs=100):
    """Mô phỏng routing giữa các cặp node ngẫu nhiên"""
    if G.is_directed():
        nodes = list(G.nodes())
    else:
        nodes = list(G.nodes())

    if len(nodes) < 2:
        return {}

    # Random pairs — dùng random.sample thay vì numpy.choice (tránh lỗi mixed types)
    import random as _random
    _rng = _random.Random(42)
    max_pairs = min(num_pairs, (len(nodes) * (len(nodes) - 1)) // 2)
    pairs = set()
    while len(pairs) < max_pairs:
        src, dst = _rng.sample(nodes, 2)
        if src != dst:
            pairs.add((src, dst) if (str(src) < str(dst)) else (dst, src))
    pairs = list(pairs)

    results = {
        "num_pairs": len(pairs),
        "shortest_path": {"avg_hops": 0, "max_hops": 0, "total": 0},
        "xy_routing": {"avg_hops": 0, "max_hops": 0, "total": 0} if routing_fn else None,
    }

    # Shortest path (baseline)
    sp_hops = []
    for src, dst in pairs:
        try:
            if G.is_directed():
                path = nx.shortest_path(G, src, dst)
            else:
                path = nx.shortest_path(G, src, dst)
            sp_hops.append(len(path) - 1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            sp_hops.append(-1)

    valid_sp = [h for h in sp_hops if h >= 0]
    if valid_sp:
        results["shortest_path"]["avg_hops"] = np.mean(valid_sp)
        results["shortest_path"]["max_hops"] = max(valid_sp)
        results["shortest_path"]["valid_pairs"] = len(valid_sp)

    # XY routing (chỉ cho mesh)
    if routing_fn and "mesh" in G.graph.get("topo_type", "").lower():
        xy_hops = []
        for src, dst in pairs:
            try:
                r1, c1 = src
                r2, c2 = dst
                path = xy_routing_mesh(r1, c1, r2, c2)
                xy_hops.append(len(path) - 1)
            except:
                xy_hops.append(-1)

        valid_xy = [h for h in xy_hops if h >= 0]
        if valid_xy:
            results["xy_routing"]["avg_hops"] = np.mean(valid_xy)
            results["xy_routing"]["max_hops"] = max(valid_xy)
            results["xy_routing"]["valid_pairs"] = len(valid_xy)

    # Tính path utilization (congestion indicator)
    edge_usage = defaultdict(int)
    for src, dst in pairs:
        try:
            path = nx.shortest_path(G, src, dst) if not G.is_directed() else nx.shortest_path(G, src, dst)
            for i in range(len(path) - 1):
                edge = tuple(sorted((path[i], path[i+1])))
                edge_usage[edge] += 1
        except:
            pass

    # Congestion variance
    if edge_usage:
        usages = list(edge_usage.values())
        results["edge_usage_mean"] = np.mean(usages)
        results["edge_usage_std"] = np.std(usages)
        results["edge_usage_max"] = max(usages)
        results["congestion_imbalance"] = results["edge_usage_std"] / (results["edge_usage_mean"] + 1)

    return results


# ═══════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def visualize_topology(G, ax, title=None):
    """Vẽ topology với node color theo degree"""
    pos = None
    name = G.graph.get("name", "Unknown")
    topo = G.graph.get("topo_type", "")

    if topo in ("mesh", "torus"):
        pos = {n: (n[1], -n[0]) for n in G.nodes()}
    elif topo == "fat_tree":
        pos = nx.multipartite_layout(G, subset_key="layer")
    elif topo == "butterfly":
        pos = nx.multipartite_layout(G, subset_key="stage")
    else:
        pos = nx.spring_layout(G, seed=42, k=3)

    degrees = [d for _, d in G.degree()]
    node_color = degrees

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80,
                           node_color=node_color, cmap=plt.cm.viridis,
                           alpha=0.85)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)

    # Label chỉ vài node cho gọn
    if G.number_of_nodes() <= 16:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    ax.set_title(title or name, fontsize=12, pad=10)
    ax.axis("off")
    return ax


def plot_degree_distribution(G, ax, title=None):
    """Vẽ degree distribution"""
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, bins=range(min(degrees), max(degrees)+2),
            alpha=0.7, color="steelblue", edgecolor="white", align="left")
    ax.axvline(np.mean(degrees), color="red", linestyle="--",
               label=f'TB={np.mean(degrees):.2f}')
    ax.set_xlabel("Bậc (Degree)")
    ax.set_ylabel("Số nodes")
    ax.set_title(title or "Degree Distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_betweenness(G, bc, ax, title=None):
    """Vẽ betweenness centrality"""
    values = list(bc.values())
    ax.hist(values, bins=30, alpha=0.7, color="darkorange", edgecolor="white")
    ax.axvline(np.mean(values), color="red", linestyle="--",
               label=f'TB={np.mean(values):.4f}')
    ax.set_xlabel("Betweenness Centrality")
    ax.set_ylabel("Số nodes")
    ax.set_title(title or "Betweenness Centrality", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_metric_comparison(all_metrics, ax, metric_key="avg_shortest_path"):
    """So sánh metric giữa các topology"""
    names = [m["name"] for m in all_metrics]
    values = [m.get(metric_key, 0) for m in all_metrics]

    colors = plt.cm.tab10(range(len(names)))
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.75)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        if isinstance(val, (int, float)):
            ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}" if isinstance(val, float) else f"{val}",
                    va="center", fontsize=9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(metric_key.replace("_", " ").title())
    ax.set_title(f"So sánh {metric_key.replace('_', ' ')}", fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


# ═══════════════════════════════════════════════════════════════
# 5. MAIN: CHẠY DEMO ĐẦY ĐỦ
# ═══════════════════════════════════════════════════════════════

def run_full_demo():
    print("=" * 72)
    print("GRAPH-NOC DEMO: Ứng dụng lý thuyết đồ thị cho tối ưu Network-on-Chip")
    print("=" * 72)

    # ── Build topologies ──
    print("\n[1] Xây dựng các NoC topologies...")
    topologies = {
        "mesh4x4": build_mesh(4, 4),
        "mesh8x8": build_mesh(8, 8),
        "torus4x4": build_torus(4, 4),
        "ring16": build_ring(16),
        "fattree_k4": build_fat_tree(4),
        "smallworld": build_smallworld_noc(36, 4, 0.15),
        "random": build_random_noc(36, 0.12),
    }

    for name, G in topologies.items():
        print(f"  {name:20s}: {G.graph['name']:25s} | nodes={G.number_of_nodes():3d}, edges={G.number_of_edges():3d}")

    # ── Analyze ──
    print("\n[2] Phân tích graph metrics...")
    all_metrics = []
    all_bc = {}

    for name, G in topologies.items():
        metrics, bc = analyze_noc_topology(G)
        metrics["name"] = G.graph["name"]
        all_metrics.append(metrics)
        if bc:
            all_bc[name] = bc

        print(f"\n  ┌─ {G.graph['name']} ─────────────────────")
        print(f"  │ Nodes: {metrics['num_nodes']}, Edges: {metrics['num_edges']}")
        print(f"  │ Bậc TB: {metrics['avg_degree']:.2f} (min={metrics['min_degree']}, max={metrics['max_degree']})")
        print(f"  │ Đường kính: {metrics.get('diameter', 'N/A')} hops")
        print(f"  │ Đường đi TB: {metrics.get('avg_shortest_path', 'N/A')} hops")
        print(f"  │ Liên thông: {metrics['is_connected']}")
        if "avg_betweenness" in metrics:
            print(f"  │ Betweenness TB: {metrics['avg_betweenness']:.4f}")
            print(f"  │ Top nodes dễ nghẽn: {metrics.get('top_betweenness_nodes', 'N/A')}")
        print(f"  └───────────────────────────────────")

    # ── Simulate Routing ──
    print("\n[3] Mô phỏng định tuyến (100 cặp ngẫu nhiên)...")
    routing_results = {}
    for name, G in topologies.items():
        # Chỉ mô phỏng trên graph đủ nhỏ
        if G.number_of_nodes() <= 100:
            res = simulate_routing(G, xy_routing_mesh, num_pairs=100)
            routing_results[name] = res
            print(f"  {G.graph['name']:25s}: SP={res['shortest_path']['avg_hops']:.2f} hops (max={res['shortest_path']['max_hops']})", end="")
            if "xy_routing" in res and res["xy_routing"]:
                print(f", XY={res['xy_routing']['avg_hops']:.2f} hops", end="")
            if "congestion_imbalance" in res:
                print(f", Congestion imbalance={res['congestion_imbalance']:.3f}", end="")
            print()

    # ── Visualize ──
    print("\n[4] Vẽ biểu đồ...")

    # Figure 1: Topology comparison
    n_topo = len(topologies)
    fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
    axes1 = axes1.flatten()
    for i, (name, G) in enumerate(topologies.items()):
        visualize_topology(G, axes1[i], G.graph["name"])
    for j in range(i + 1, len(axes1)):
        axes1[j].axis("off")
    plt.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, "01-topologies.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 01-topologies.png — Các NoC topologies")

    # Figure 2: Degree distributions
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 8))
    axes2 = axes2.flatten()
    for i, (name, G) in enumerate(topologies.items()):
        plot_degree_distribution(G, axes2[i], G.graph["name"])
    for j in range(i + 1, len(axes2)):
        axes2[j].axis("off")
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "02-degree-distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 02-degree-distributions.png — Phân phối bậc")

    # Figure 3: Betweenness
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
    axes3 = axes3.flatten()
    i = 0
    for name, bc in all_bc.items():
        if i < len(axes3):
            plot_betweenness(topologies[name], bc, axes3[i], topologies[name].graph["name"])
            i += 1
    for j in range(i, len(axes3)):
        axes3[j].axis("off")
    plt.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, "03-betweenness.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 03-betweenness.png — Betweenness Centrality")

    # Figure 4: Metric comparison
    fig4, axes4 = plt.subplots(2, 3, figsize=(16, 8))
    axes4 = axes4.flatten()
    metrics_list = ["avg_degree", "diameter", "avg_shortest_path", "avg_betweenness", "max_betweenness"]
    for i, m in enumerate(metrics_list):
        if i < len(axes4):
            # Filter metrics that exist
            valid = [mm for mm in all_metrics if m in mm and isinstance(mm.get(m), (int, float))]
            if valid:
                plot_metric_comparison(valid, axes4[i], m)
    for j in range(i + 1, len(axes4)):
        axes4[j].axis("off")
    plt.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR, "04-metric-comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 04-metric-comparison.png — So sánh metrics giữa các topology")

    # Figure 5: Routing comparison (SP vs XY trên Mesh)
    fig5, ax5 = plt.subplots(1, 1, figsize=(8, 5))
    mesh4 = topologies["mesh4x4"]
    res_mesh4 = routing_results.get("mesh4x4", {})
    labels = []
    sp_vals = []
    xy_vals = []
    for name, res in routing_results.items():
        topo = topologies[name].graph["name"]
        labels.append(topo)
        sp_vals.append(res["shortest_path"]["avg_hops"])
        if "xy_routing" in res and res["xy_routing"]:
            xy_vals.append(res["xy_routing"]["avg_hops"])
        else:
            xy_vals.append(0)

    x = np.arange(len(labels))
    w = 0.35
    ax5.bar(x - w/2, sp_vals, w, label="Shortest Path", alpha=0.75, color="steelblue")
    ax5.bar(x + w/2, xy_vals, w, label="XY Routing", alpha=0.75, color="coral")
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax5.set_ylabel("Số hops trung bình")
    ax5.set_title("So sánh định tuyến: Shortest Path vs XY Routing", fontsize=12)
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig5.savefig(os.path.join(OUT_DIR, "05-routing-comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ 05-routing-comparison.png — So sánh định tuyến")

    # ── Bảng tổng kết ──
    print("\n" + "=" * 72)
    print("BẢNG TỔNG KẾT GRAPH METRICS CHO NOC")
    print("=" * 72)
    header = f"{'Topology':25s} | {'Nodes':5s} | {'Bậc':6s} | {'Đ.kính':7s} | {'Đ.dài TB':8s} | {'BNess TB':8s}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        name = m["name"][:24]
        nodes = m["num_nodes"]
        degree = f"{m['avg_degree']:.2f}"
        diam = f"{m.get('diameter', 'N/A')}" if isinstance(m.get('diameter'), (int, float)) else str(m.get('diameter', 'N/A'))
        avgsp = f"{m.get('avg_shortest_path', 'N/A'):.3f}" if isinstance(m.get('avg_shortest_path'), (int, float)) else str(m.get('avg_shortest_path', 'N/A'))
        bness = f"{m.get('avg_betweenness', 'N/A'):.4f}" if isinstance(m.get('avg_betweenness'), (int, float)) else str(m.get('avg_betweenness', 'N/A'))
        print(f"{name:25s} | {nodes:5d} | {degree:6s} | {diam:7s} | {avgsp:8s} | {bness:8s}")
    print("-" * len(header))

    # ── Kết luận ──
    print("\n" + "=" * 72)
    print("NHẬN XÉT & ĐỊNH HƯỚNG NGHIÊN CỨU")
    print("=" * 72)
    print("""
  A. Graph Metrics quan trọng cho NoC:
      1. Đường kính (Diameter): → ảnh hưởng latency tối đa
      2. Đường đi TB: → ảnh hưởng latency trung bình
      3. Bậc (Degree): → ảnh hưởng cost (số cổng router)
      4. Betweenness: → ảnh hưởng congestion / load imbalance
      5. Connectivity: → ảnh hưởng fault tolerance

  B. Topology trade-offs:
      - Mesh: simple, low cost, scalable, moderate latency
      - Torus: lower diameter hơn mesh (wrap-around), nhưng wiring phức tạp
      - Fat-Tree: bandwidth cao, non-blocking, nhưng cost cao
      - Butterfly: latency thấp, nhưng không đều (dị bộ)
      - Small-World: có shortcut giảm latency, nhưng routing phức tạp

  C. Hướng ứng dụng Graph Neural Networks (GNN) cho NoC:
      1. Dự đoán latency/traffic: GNN học từ graph topology + traffic pattern
      2. Adaptive Routing: DRL + GNN để chọn đường đi tối ưu theo real-time state
      3. Topology Generation: Graph generative models cho NoC design space exploration
      4. Congestion Prediction: GNN phân tích distributed congestion pattern
      5. Fault Recovery: Graph-based rerouting khi node/link fail

  D. Paper references chính:
      - NOCTOPUS (2026): GNN pipeline cho NoC topology optimization & prediction
      - GraphNoC (FPT 2024): GNN cho application-specific FPGA NoC benchmarking
      - Slim NoC (ASPLOS '18): Graph theory (degree-diameter problem) cho low-diameter NoC
      - DRL+GNN routing: Kết hợp DRL + GNN cho routing optimization (generalizable)
    """)

    # ── Save results as JSON ──
    # Chuyển các metrics có thể serialize
    json_metrics = []
    for m in all_metrics:
        jm = {}
        for k, v in m.items():
            if isinstance(v, (int, float, str)):
                jm[k] = v
            elif isinstance(v, list):
                jm[k] = str(v)
        json_metrics.append(jm)

    with open(os.path.join(OUT_DIR, "graph-noc-results.json"), "w") as f:
        json.dump(json_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  📊 Kết quả đã lưu: {OUT_DIR}/graph-noc-results.json")

    # ── Bonus: Extended demo cho 8x8 mesh ──
    print("\n" + "~" * 72)
    print("BONUS: Mesh 8x8 detailed analysis")
    print("~" * 72)
    G8 = topologies["mesh8x8"]
    m8, _ = analyze_noc_topology(G8)
    print(f"  Nodes: {m8['num_nodes']}, Edges: {m8['num_edges']}")
    print(f"  Diameter: {m8.get('diameter', -1)} hops  (tối đa hops từ (0,0)->(7,7) = 14)")
    print(f"  Average shortest path: {m8.get('avg_shortest_path', -1):.3f} hops")
    print(f"  Top betweenness nodes: {m8.get('top_betweenness_nodes', [])}")
    print(f"  → Nodes trung tâm có betweenness cao nhất → dễ bottleneck")
    print(f"  → Gợi ý: dùng adaptive routing để phân tán traffic tránh congestion")

    print("\n✅ DEMO HOÀN THÀNH! Tất cả kết quả trong:", OUT_DIR)


if __name__ == "__main__":
    run_full_demo()
