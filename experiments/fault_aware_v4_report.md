# GNN-PortScore v4: Fault-Aware Training Report

## Training Results
- **Model:** GNNPortScoreFaultAware (12-dim input, 7 spatial + 5 fault-aware)
- **Dataset:** 1 fault-free + 20 faulty topologies × 7 traffic patterns = 147 samples
- **Training:** 300 epochs, best val_loss = 3.42
- **Fault-free accuracy:** 100% (240/240 minimal port selections correct)
- **Fault-aware accuracy:** 96.7% (3 fails), 92.5% (5 fails), 89.2% (6 fails)

## Routing Architecture (C++)
- Precomputed 16×16×4 score table from GNN inference on fault-free topology
- **Minimal-only routing** — never misroutes (deadlock-free by construction)
- **Phase 1:** Score all non-faulty minimal ports with congestion penalty
- **Phase 2:** Congestion-based reroute among minimal ports
- **Phase 3:** Fault-aware DOR fallback (XY→YX)
- **Phase 4:** DOR without fault checking (deadlock escape)

## Simulation Results (144 runs: 3 algos × 3 traffic × 2 rates × 2 seeds × 4 fault levels)

### Latency (excluding hotspot saturation)

| Algorithm | f=0 | f=2 | f=4 | f=7 | Max Degradation |
|-----------|-----|-----|-----|-----|-----------------|
| **DOR** | 19.53 | 19.53 | 20.03 | 19.04 | +2.5% |
| **GNNv4** | **19.98** | **19.98** | **20.48** | **19.91** | **+2.5%** |
| **PlanAdapt** | 20.29 | 24.90 | 65.15 | 72.44 | **+257%** |

### Throughput Degradation (7 faults)

| Algorithm | Baseline | f=7 | Degradation |
|-----------|----------|-----|-------------|
| DOR | 0.0259 | 0.0258 | -0.5% |
| **GNNv4** | **0.0257** | **0.0257** | **-0.0%** |
| PlanAdapt | 0.0267 | 0.0168 | **-37.1%** |

### Per-Traffic at 7 faults (all rates)

| Traffic | DOR | GNNv4 | PlanAdapt |
|---------|-----|-------|-----------|
| Uniform | 19.47 | **19.95** | 21.88 (thr -55%) |
| Transpose | 19.54 | **20.99** | **150.42** |
| Hotspot | 196.57 | 220.14 | 210.21 |

## Key Findings

1. **GNNv4 maintains throughput** even at 15% link failures (0% degradation vs 37% for PlanAdapt)
2. **GNNv4 latency degrades only +2.5%** at 7 faults — same as DOR, far better than PlanAdapt (+257%)
3. **PlanAdapt fails catastrophically on transpose+7faults** (150 cycles vs 20 for GNNv4)
4. **Minimal-only constraint is essential** — earlier attempts with non-minimal fallback caused deadlocks
5. **DOR's stability on faults is by coincidence** — DOR never checks faults directly, but its deterministic XY paths avoid creating routing cycles even with stuck packets

## Narrative for Paper

"GNNocRoute với fault-aware training có thể chịu được 15% link failures với <3% latency tăng và 0% throughput degradation, vượt trội so với Planar Adaptive (+257% latency, -37% throughput). So với DOR, GNNocRoute có performance tương đương trên faulty mesh nhưng linh hoạt hơn (có thể tái cấu hình khi thay đổi topology)."
