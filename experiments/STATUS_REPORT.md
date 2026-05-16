# GNNocRoute-DRL: Experiment Progress & Results

## Status (2026-05-16 16:30)

### Hướng A: Port Score Routing
- [x] GNN training (train_gnn_port_score_v3.py): 100% minimal port accuracy
- [x] C++ routing function (gnn_port_score_route_4x4_mesh)
- [x] Registered + BookSim2 compiled
- [ ] Experiments running (~130/397 done)

### Hướng B: Faulty Topology
- [x] BookSim2 link_failures bug fixed
- [x] Experiment script (run_faulty_experiments.py)
- [ ] Experiments running (~40/1215 done)

## Key Architecture

### Port Score GNN (v3)
- GATv2 encoder → 32-dim node embeddings
- 4 separate port decoders (E/W/S/N)
- Training: KL divergence + MSE across 7 traffic patterns
- Score table: float[16][16][4] precomputed

### Routing Function
- At each hop: find minimal ports (toward destination)
- Among minimal: pick highest GNN score
- Congestion override: if best port buffer > 12, try second best
- VC-based deadlock freedom (2 VC classes)

### Fault Handling
- GNN port-score: automatic fallback when best port unreachable
- Planar adapt: built-in IsFaultyOutput checking
- DOR: no fault awareness (baseline for comparison)

## Results Summary (PENDING)

Will be populated once experiments complete.
