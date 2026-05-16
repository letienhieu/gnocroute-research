#!/usr/bin/env python3
"""
Phân tích 378 dense configs → Biểu đồ cho paper
+ Train GNN với contrastive loss
"""
import json, numpy as np, os, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/opc/.openclaw/workspace/papers/paper03-q1-jsa'
RESULTS = f'{BASE}/code/experiments/results'
FIGS = f'{BASE}/latex/figures'
os.makedirs(FIGS, exist_ok=True)

with open(f'{RESULTS}/dense_results.json') as f:
    data = json.load(f)

print(f'Loaded {len(data)} configs')

# ============================================================
# TASK 1: FIGURES
# ============================================================

# Figure 1: Latency vs Injection Rate — hotspot all topologies
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {'dor': '#e74c3c', 'adaptive_xy_yx': '#f39c12'}
markers = {'dor': 'o', 'adaptive_xy_yx': 's'}

for idx, (topo, ax) in enumerate(zip(['mesh44', 'mesh88', 'mesh1616'], axes)):
    for algo in ['dor', 'adaptive_xy_yx']:
        pts = [(r['inj_rate'], r['latency']) for r in data 
               if r['topology']==topo and r['traffic']=='hotspot' 
               and r['algorithm']==algo and r['latency']]
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=markers[algo], color=colors[algo], 
                   linewidth=2.5, markersize=5, label=f'{"Adaptive" if algo=="adaptive_xy_yx" else "XY (DOR)"}')
    
    ax.set_xlabel('Injection Rate', fontsize=11)
    ax.set_ylabel('Avg Latency (cycles)', fontsize=11)
    ax.set_title(f'{topo.upper()}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.35)

plt.tight_layout()
plt.savefig(f'{FIGS}/fig-latency-all-topos.png', dpi=200, bbox_inches='tight')
plt.close()
print('✅ Figure 1: Latency all topologies')

# Figure 2: Improvement heatmap
fig, ax = plt.subplots(figsize=(10, 5))
topos = ['mesh44', 'mesh88', 'mesh1616']
inj_rates = [0.01, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3]

improvement = np.zeros((len(topos), len(inj_rates)))
for i, topo in enumerate(topos):
    for j, inj in enumerate(inj_rates):
        xy = next((r['latency'] for r in data if r['topology']==topo and r['traffic']=='hotspot'
                   and r['algorithm']=='dor' and r['inj_rate']==inj and r['latency']), None)
        ad = next((r['latency'] for r in data if r['topology']==topo and r['traffic']=='hotspot'
                   and r['algorithm']=='adaptive_xy_yx' and r['inj_rate']==inj and r['latency']), None)
        if xy and ad: improvement[i, j] = (xy - ad) / xy * 100

im = ax.imshow(improvement, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
ax.set_xticks(range(len(inj_rates)))
ax.set_yticks(range(len(topos)))
ax.set_xticklabels([str(x) for x in inj_rates], rotation=45)
ax.set_yticklabels(['Mesh 4×4', 'Mesh 8×8', 'Mesh 16×16'])
ax.set_xlabel('Injection Rate')
ax.set_ylabel('Topology')
ax.set_title('Adaptive Routing Improvement over XY (%) — Hotspot', fontweight='bold')

for i in range(len(topos)):
    for j in range(len(inj_rates)):
        val = improvement[i, j]
        color = 'white' if abs(val) > 20 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

plt.colorbar(im, label='Improvement (%)')
plt.tight_layout()
plt.savefig(f'{FIGS}/fig-improvement-heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print('✅ Figure 2: Improvement heatmap')

# Figure 3: Saturation analysis
fig, ax = plt.subplots(figsize=(10, 5))
for topo in topos:
    for algo, label in [('dor', 'XY'), ('adaptive_xy_yx', 'Adaptive')]:
        pts = [(r['inj_rate'], r['latency']) for r in data 
               if r['topology']==topo and r['traffic']=='hotspot' 
               and r['algorithm']==algo and r['latency']]
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            # Find saturation point (latency > 3x baseline)
            base = ys[0]
            sat_idx = next((i for i, y in enumerate(ys) if y > 3*base), len(ys)-1)
            ax.plot(xs[:sat_idx+1], ys[:sat_idx+1], 'o-', 
                   label=f'{topo}_{label}', linewidth=2)
            if sat_idx < len(xs)-1:
                ax.plot(xs[sat_idx:], ys[sat_idx:], ':o', alpha=0.4)

ax.set_xlabel('Injection Rate', fontsize=11)
ax.set_ylabel('Avg Latency (cycles)', fontsize=11)
ax.set_title('Saturation Analysis — Hotspot Traffic', fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/fig-saturation.png', dpi=200, bbox_inches='tight')
plt.close()
print('✅ Figure 3: Saturation analysis')

print(f'\n✅ All figures saved to {FIGS}/')
