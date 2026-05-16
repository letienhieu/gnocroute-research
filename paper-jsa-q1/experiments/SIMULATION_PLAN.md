# Simulation Campaign — GNNocRoute-DRL (JSA Q1)

## Objective

Generate **actual GNN-PPO simulation results** to replace all "projection/estimation" in the paper's Results section. Current baseline-only results have been verified; GNN-PPO runs still missing.

## Experiment Space (1,350 runs)

| Factor | Levels | Count |
|--------|--------|-------|
| Topologies | Mesh 4×4, Mesh 8×8, Torus 4×4 (k=4,n=2) | 3 |
| Traffic | uniform, transpose, hotspot | 3 |
| Algorithms | dor, adaptive_xy_yx, min_adapt, valiant, gnn_ppo_route_4x4 (hoặc 8x8) | 5 (mesh) / 4 (torus) |
| Injection rates | 0.01, 0.02, 0.05, 0.1, 0.2, 0.3 | 6 |
| Seeds | 0,1,2,3,4 | 5 |

## Metrics

- Latency (cycles) — **mean over last sample**
- Accepted flit rate
- Hop count
- Saturation point

## Output Files

- `results/gnn_*/` — raw per-experiment text
- `results/all_results.csv` — aggregated, parsed
- `results/analysis_summary.csv` — means + CI + %improvement
- `figures/` — paper-quality plots

