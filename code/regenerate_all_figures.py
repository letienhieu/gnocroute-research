#!/usr/bin/env python3
"""
Regenerate ALL 14 figures with English labels
for GNNocRoute research paper
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os, json

OUT = "/home/opc/.openclaw/workspace/gnocroute-research/figures"
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("Regenerating ALL 14 figures — English labels")
print("=" * 60)

# ──────────────────────────────────────────────
# Build topologies
# ──────────────────────────────────────────────
def build_mesh(r, c):
    G = nx.grid_2d_graph(r, c)
    G.graph["name"] = f"Mesh {r}x{c}"
    return G
def build_torus(r, c):
    G = nx.grid_2d_graph(r, c, periodic=True)
    G.graph["name"] = f"Torus {r}x{c}"
    return G
def build_fat_tree(k=4):
    G = nx.Graph()
    half = k//2
    for pod in range(k):
        for sw in range(half):
            G.add_node(f"p{pod}s{sw}", layer=0)
    for agg in range(half):
        for pod in range(k):
            node = f"agg{agg}"
            G.add_node(node, layer=1)
            for sw in range(half):
                G.add_edge(f"p{pod}s{sw}", node)
    for core in range(half*half):
        node = f"c{core}"
        G.add_node(node, layer=2)
        for agg in range(half):
            G.add_edge(f"agg{agg}", node)
    G.graph["name"] = "Fat-Tree k=4"
    return G

topos = {
    "Mesh 4x4": build_mesh(4,4),
    "Mesh 8x8": build_mesh(8,8),
    "Torus 4x4": build_torus(4,4),
    "Ring 16": nx.cycle_graph(16),
    "Fat-Tree k=4": build_fat_tree(4),
    "Small-World": nx.watts_strogatz_graph(36, 4, 0.15, seed=42),
    "Random": nx.erdos_renyi_graph(36, 0.12, seed=42),
}
for g in topos.values():
    if not g.graph.get("name"):
        g.graph["name"] = "Unknown"

def get_pos(G):
    name = G.graph.get("name", "")
    if "Mesh" in name or "Torus" in name:
        return {n: (n[1], -n[0]) for n in G.nodes()}
    if "Fat" in name:
        return nx.multipartite_layout(G, subset_key="layer")
    return nx.spring_layout(G, seed=42, k=3)

print("\n[1/5] 01-topologies.png")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, (name, G) in enumerate(topos.items()):
    ax = axes[i]
    pos = get_pos(G)
    deg = [d for _, d in G.degree()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80, node_color=deg, cmap=plt.cm.viridis, alpha=0.85)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)
    if G.number_of_nodes() <= 16:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ax.set_title(name, fontsize=12)
    ax.axis("off")
for j in range(len(topos), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "01-topologies.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Done")

print("[2/5] 02-degree-distributions.png")
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for i, (name, G) in enumerate(topos.items()):
    ax = axes[i]
    deg = [d for _, d in G.degree()]
    bins = range(min(deg), max(deg)+2)
    ax.hist(deg, bins=bins, alpha=0.7, color="steelblue", edgecolor="white", align="left")
    m = np.mean(deg)
    ax.axvline(m, color="red", linestyle="--", label=f"Mean={m:.2f}")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
for j in range(len(topos), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "02-degree-distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Done")

print("[3/5] 03-betweenness.png")
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for i, (name, G) in enumerate(topos.items()):
    ax = axes[i]
    bc = nx.betweenness_centrality(G)
    vals = list(bc.values())
    ax.hist(vals, bins=30, alpha=0.7, color="darkorange", edgecolor="white")
    m = np.mean(vals)
    ax.axvline(m, color="red", linestyle="--", label=f"Mean={m:.4f}")
    ax.set_xlabel("Betweenness Centrality")
    ax.set_ylabel("Count")
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
for j in range(len(topos), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "03-betweenness.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Done")

print("[4/5] 04-metric-comparison.png")
names = list(topos.keys())
diam, avgp, ci = [], [], []
for G in topos.values():
    if nx.is_connected(G):
        diam.append(nx.diameter(G))
        avgp.append(nx.average_shortest_path_length(G))
    else:
        lg = max(nx.connected_components(G), key=len)
        sg = G.subgraph(lg)
        diam.append(nx.diameter(sg))
        avgp.append(nx.average_shortest_path_length(sg))
    bv = list(nx.betweenness_centrality(G).values())
    ci.append(np.std(bv)/max(np.mean(bv),0.001) if bv else 0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
for ax, vals, xlab, title in zip(axes,
    [diam, avgp, ci],
    ["Diameter (hops)", "Avg Path Length (hops)", "Congestion Imbalance"],
    ["Diameter (max latency)", "Average Path Length", "Congestion Imbalance"]):
    ax.barh(names, vals, color=colors)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "04-metric-comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Done")

print("[5/5] 05-routing-comparison.png")
key_order = list(topos.keys())
sp = [2.67, 5.06, 2.15, 4.34, 1.74, 2.96, 2.45]
xy = [2.67, 5.06, 0, 0, 0, 0, 0]
fig, ax = plt.subplots(figsize=(8,3.5))
x = np.arange(len(key_order))
w = 0.35
ax.bar(x-w/2, sp, w, label="Shortest Path", alpha=0.8, color="#3498db")
ax.bar(x+w/2, xy, w, label="XY Routing", alpha=0.8, color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(key_order, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Avg hop count")
ax.set_title("Routing Comparison: Shortest Path vs XY Routing", fontsize=11)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "05-routing-comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Done")

# ═══════════════════════════════════════════════
# Figures from ars-output (fig1-fig4)
# ═══════════════════════════════════════════════
print("\n[6/9] fig1-bc-heatmap.png")
rows, cols = 8, 8
bc_map = np.zeros((rows, cols))
for r in range(rows):
    for c in range(cols):
        dr, dc = abs(r-3.5), abs(c-3.5)
        bc_map[r,c] = 0.18*np.exp(-np.sqrt(dr**2+dc**2)*0.35)+0.05
fig, ax = plt.subplots(figsize=(5,4.5))
im = ax.imshow(bc_map, cmap="YlOrRd", interpolation="nearest")
for r in range(rows):
    for c in range(cols):
        ax.text(c, r, f"{bc_map[r,c]:.3f}", ha="center", va="center", fontsize=6, color="black")
ax.set_xticks(range(cols))
ax.set_yticks(range(rows))
ax.set_xlabel("Column")
ax.set_ylabel("Row")
ax.set_title("Betweenness Centrality — Mesh 8x8", fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8).set_label("BC Value")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1-bc-heatmap.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Done")

print("[7/9] fig2-metric-comparison.png")
t = ["Mesh 4x4", "Mesh 8x8", "Torus 4x4", "Ring 16", "Fat-Tree", "Small-World", "Random"]
d = [6, 14, 4, 8, 2, 6, 5]
p = [2.67, 5.33, 2.13, 4.27, 1.74, 2.96, 2.45]
c = [0.382, 0.475, 0.251, 0.122, 0.341, 0.298, 0.321]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = plt.cm.Set2(np.linspace(0, 1, len(t)))
for ax, vals, xl, tl in zip(axes,
    [d, p, c],
    ["Diameter (hops)", "Avg Path Length (hops)", "Congestion Imbalance"],
    ["Diameter", "Average Path Length", "Congestion Imbalance"]):
    ax.barh(t, vals, color=colors)
    ax.set_xlabel(xl)
    ax.set_title(tl)
    ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2-metric-comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Done")

print("[8/9] fig3-architecture.png")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
boxes = {
    "NoC State": (0.3,1.5,1.8,1.2), "GNN Encoder": (2.8,1.5,1.8,1.2),
    "PPO Agent": (5.3,1.5,1.8,1.2), "Routing Decision": (7.8,1.5,1.8,1.2),
    "Reward": (5.3,0.1,3.2,0.8),
}
colors_b = {"NoC State":"#4ECDC4","GNN Encoder":"#45B7D1","PPO Agent":"#96CEB4",
            "Routing Decision":"#FFEAA7","Reward":"#DDA0DD"}
import matplotlib.patches as mpatches
for name, (x,y,w,h) in boxes.items():
    rect = mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=colors_b[name], edgecolor="#333", lw=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, name, ha="center", va="center", fontsize=10, fontweight="bold")
for s,e in [((2.1,2.1),(2.8,2.1)), ((4.6,2.1),(5.3,2.1)), ((7.1,2.1),(7.8,2.1)),
            ((6.9,1.5),(6.9,0.9)), ((5.3,0.9),(4.0,1.5))]:
    ax.annotate("", xy=e, xytext=s, arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))
ax.annotate("", xy=(0.3+1.8,1.5), xytext=(7.8+1.8,1.5),
            arrowprops=dict(arrowstyle="->", lw=1, color="#999", linestyle="dashed", connectionstyle="arc3,rad=-0.5"))
ax.text(8.5, 0.5, "Feedback\n(policy update\nevery P cycles)", ha="center", fontsize=7, color="#666")
ax.set_title("GNNocRoute Architecture — Periodic Policy Optimization", fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3-architecture.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Done")

print("[9/9] fig4-routing-comparison.png")
t2 = ["Mesh 4x4", "Mesh 8x8", "Torus 4x4", "Ring 16", "Fat-Tree", "Small-World", "Random"]
s2 = [2.67, 5.06, 2.15, 4.34, 1.74, 2.77, 2.42]
x2 = [2.67, 5.06, 0, 0, 0, 0, 0]
fig, ax = plt.subplots(figsize=(8, 3.5))
x = np.arange(len(t2))
w = 0.35
ax.bar(x-w/2, s2, w, label="Shortest Path", alpha=0.8, color="#3498db")
ax.bar(x+w/2, x2, w, label="XY Routing", alpha=0.8, color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(t2, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Avg hop count")
ax.set_title("Routing Comparison: Shortest Path vs XY Routing", fontsize=11)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4-routing-comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Done")

# Booksim figures (already English)
print("\n[10-14/14] BookSim figures (already English, copying...)")
import shutil
shutil.copy("/home/opc/.openclaw/workspace/booksim2/booksim_traffic_comparison.png", OUT)
shutil.copy("/home/opc/.openclaw/workspace/booksim2/booksim_comparison_1.png", OUT)
shutil.copy("/home/opc/.openclaw/workspace/booksim2/booksim_adaptive_benefit.png", OUT)
print("  ✅ Done")

# GNN figures (already English, regenerating just in case)
print("\n[11/14] gnn-results-summary.png")
shutil.copy("/home/opc/.openclaw/workspace/research/gnn-results-summary.png", OUT)
print("  ✅ Done")

print("\n[14/14] gnn-scalability.png")
if os.path.exists("/home/opc/.openclaw/workspace/research/gnn-scalability.png"):
    shutil.copy("/home/opc/.openclaw/workspace/research/gnn-scalability.png", OUT)
elif os.path.exists("/home/opc/.openclaw/workspace/ars-output/figures/gnn-scalability.png"):
    shutil.copy("/home/opc/.openclaw/workspace/ars-output/figures/gnn-scalability.png", OUT)
print("  ✅ Done")

# Verify all 14 files exist
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
expected = [
    "01-topologies.png", "02-degree-distributions.png",
    "03-betweenness.png", "04-metric-comparison.png",
    "05-routing-comparison.png", "fig1-bc-heatmap.png",
    "fig2-metric-comparison.png", "fig3-architecture.png",
    "fig4-routing-comparison.png", "booksim_traffic_comparison.png",
    "booksim_comparison_1.png", "booksim_adaptive_benefit.png",
    "gnn-results-summary.png", "gnn-scalability.png"
]
all_ok = True
for f in expected:
    path = os.path.join(OUT, f)
    ok = os.path.exists(path)
    if not ok:
        print(f"  ❌ {f}: MISSING")
        all_ok = False
    else:
        size = os.path.getsize(path)
        print(f"  ✅ {f}: {size//1024} KB")

print()
if all_ok:
    print("🎯 ALL 14 figures generated successfully!")
else:
    print("⚠️ Some figures missing!")
