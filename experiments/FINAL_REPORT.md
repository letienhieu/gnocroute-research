
# GNNocRoute-DRL: Experimental Results Report

## Summary

| Item | Value |
|------|-------|
| Total experiments | 450 (Mesh 4×4, 5 algos, 3 traffics, 6 rates, 5 seeds) |
| Completed (100 samples) | 260 |
| Partial results | 180 (old sim_count=1, or aborted at congestion) |
| Total data files | 440 |

## Data Coverage (sim_count=100)

| Rate | dor | adaptive | min_adapt | valiant | gnn_ppo_route_4x4 |
|------|-----|----------|-----------|---------|-------------------|
| 0.01 | ✓ 6/15 | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 |
| 0.02 | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 |
| 0.05 | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 |
| 0.1  | ✓ 15/15 | ✓ 15/15 | ✓ 15/15 | ✓ 10/15 | ✓ 15/15 |
| 0.2  | ✓ 5/15  | ✓ 5/15  | ✓ 5/15  | ✗ 0/15 | ✗ 0/15 |
| 0.3  | ✗ 0/15 | ✗ 0/15 | ✗ 0/15 | ✗ 0/15 | ✗ 0/15 |

## Key Findings

1. **GNN-PPO under uniform traffic**: Comparable to XY (19.8 vs 19.6 at rate 0.1)
2. **GNN-PPO under transpose**: 115% worse than XY (44.0 vs 20.5 at rate 0.1)
3. **Hotspot saturation**: All algorithms saturate at rate ≥ 0.05
4. **GNN encoder**: Achieves |r| = 0.978 correlation with betweenness centrality
5. **Precomputed table limitation**: The offline routing table doesn't generalize to unseen traffic patterns

## Paper Updates

- **Section 02** (Related Work): Updated 
- **Section 04** (Experiments): Updated simulation params, algorithm descriptions
- **Section 05** (Results): Replaced projections with actual data
- **Section 06** (Discussion): Updated to discuss offline vs online tradeoffs
- **Section 07** (Conclusion): Updated claims, added honest assessment
- **Abstract**: Updated key numbers

## Files

| File | Description |
|------|-------------|
| `SIMULATION_PLAN.md` | Campaign design |
| `results/all_results.csv` | Aggregated results (438 rows) |
| `results/*.txt` | Raw BookSim2 outputs (440 files) |
| `analyze_results.py` | Analysis pipeline |
| `run_focused.py` | Batch experiment runner |
| `latex/04-experiments.tex` | Updated experiments section |
| `latex/05-results.tex` | Updated results section |
| `latex/06-discussion.tex` | Updated discussion section |
| `latex/07-conclusion.tex` | Updated conclusion section |

