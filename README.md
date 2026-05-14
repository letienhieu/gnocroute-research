# GNNocRoute Research Repository

**Title:** Topology-Aware Periodic Routing Optimization for Network-on-Chip: A Feasibility Study using Graph Neural Networks

**Author:** Le Tien Hieu, IEEE Member — VNU-ITI, Vietnam National University, Hanoi

## Structure
```
code/
  gnn-experiments/     PyTorch Geometric + NetworkX experiments
  noc-simulations/     BookSim2 batch scripts
data/                  (raw data - see results/)
figures/               Generated figures
paper/                 LaTeX source + PDF
results/               CSV + JSON results
```

## Reproducibility
- All experiments use fixed seeds where applicable
- Each result file has timestamp
- Configurations fully specified in scripts
- BookSim2 commit: (see git submodule)

## Experiments
1. Graph analysis (NetworkX) — 7 topologies
2. GNN embedding correlation (PyTorch Geometric)
3. BookSim2 routing simulation — 4 algorithms × 3 topologies × 3 traffic patterns × 9 IR
