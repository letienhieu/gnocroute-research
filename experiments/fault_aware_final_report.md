# GNN-PortScore v4: Fault-Aware Training — Final Report

## Model
- **Architecture:** GATv2, 12-dim input (7 spatial + 5 fault-aware)
- **Training:** 300 epochs, best val_loss=3.42
- **Accuracy:** fault-free 100%, fault-avail 96.7% (3 fails) / 92.5% (5 fails)

## Routing (C++)
- Precomputed 16×16×4 score table
- Minimal-only routing (deadlock-free by construction):
  1. GNN score → best non-faulty minimal port
  2. Congestion-based reroute among minimal ports
  3. XY/YX DOR with fault avoidance
  4. DOR without fault checking (same behavior as standard XY)

## Results (144 runs)

### Stability
- **GNNv4**: 0 unstable cases outside hotspot saturation ✅
- **DOR**: 0 unstable cases outside hotspot saturation ✅
- **PlanAdapt**: unstable on transpose+f=7, uniform+f=7

### Latency (excluding hotspot saturation)
| Algo | f=0 | f=2 | f=4 | f=7 | Max Deg |
|------|-----|-----|-----|-----|---------|
| DOR | 19.53 | 19.53 | 20.03 | 19.04 | +2.5% |
| **GNNv4** | 20.22 | 20.26 | 21.01 | 25.09 | +24.1% |
| PlanAdapt | 20.29 | 24.90 | 65.15 | 72.44 | **+257%** |

### Throughput
| Algo | Baseline | f=7 | Degradation |
|------|----------|-----|-------------|
| DOR | 0.0259 | 0.0258 | -0.5% |
| **GNNv4** | 0.0262 | 0.0256 | **-2.2%** |
| PlanAdapt | 0.0267 | 0.0168 | **-37.1%** |

### Per-traffic at 15% failures (low rate)
| Traffic | DOR | GNNv4 | PlanAdapt |
|---------|-----|-------|-----------|
| Uniform | 19.43 | 19.47 | 22.00 |
| Transpose | 19.50 | 19.55 | 22.20 |
| Hotspot | 17.18 | 17.43 | 17.59 |

## Key Insight
**Minimal-only constraint is essential.** Earlier versions with non-minimal fallback ("any non-faulty port") created routing cycles and deadlocks. The correct approach is to stay minimal at all costs — if a minimal port is faulty, try other minimal ports via XY/YX DOR. If none available, accept the stuck packet (same as DOR).

## Files
- `code/train_gnn_port_score_fault_aware_v4.py` — training + export
- `booksim2/src/gnn_port_score_route_4x4_v4.h` — C++ routing header
- `experiments/gnn_port_scores_v4.npy` — score table
- `experiments/gnn_port_score_v4_model.pt` — model weights
- `experiments/run_faulty_v4_experiments.py` — benchmark
- `experiments/results_faulty_v4/` — 144 results
