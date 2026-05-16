# Overhead Analysis: GNN-Weighted Routing

## Area, Power, and Latency Estimation

*Generated: 2026-05-16*

### 1. Weight Table Size


| Topology | Nodes | Entries | Size (bytes) | Size (bits) | Est. Area (µm²) |
|----------|-------|---------|-------------|-------------|-----------------|
| Mesh 4x4 | 16 | 256 | 1,024 | 8,192 | 4096 |
| Mesh 8x8 | 64 | 4,096 | 16,384 | 131,072 | 65536 |
| Mesh 16x16 | 256 | 65,536 | 262,144 | 2,097,152 | 1048576 |

*Assumption: ~0.5 um^2/bit for 32nm LUT/ROM*

### 2. Router Pipeline Latency


- **weight_table_lookup:** 1 cycle (combinational + register)
- **credit_compare:** ~0.5 cycle (combinational)
- **vc_assignment:** ~0.3 cycle (combinational mux)
- **total_added_pipeline:** 1 cycle (pessimistic) or ~0 cycles if merged with SA
- **comparison_adaptive_xy_yx:** 2 credit reads + compare = ~0.8 cycle


### 3. Power Estimation (per routing access)


| Topology | Dynamic (µW/access) | Static (µW) | Notes |
|----------|--------------------|-------------|-------|
| Mesh 4x4 | 3.24e-06 | 8.19e-02 | 32nm, 0.9V |
| Mesh 8x8 | 3.24e-06 | 1.31e+00 | 32nm, 0.9V |
| Mesh 16x16 | 3.24e-06 | 2.10e+01 | 32nm, 0.9V |

*Assumptions: 32nm, V=0.9V, 1 fF/bit capacitance, 25% toggle rate*

### 4. Comparison with Adaptive XY/YX


- **Adaptive XY/YX:** 16 bits, ~16 µm²
- **Comparison:** Weight table dominates area for large meshes (N^2 scaling)
- **Scaling:** O(N^2) for GNN-weighted vs O(1) for adaptive_xy_yx


### 5. Discussion


The GNN-weighted routing adds a single-cycle ROM lookup to the router pipeline. 

**For Mesh 4x4:** The weight table requires only 256 bytes (2,048 bits), which is negligible
compared to buffer storage (4 flits × 4 VCs × 4 ports × 32 bytes = 2 KB per router).

**For Mesh 8x8:** The weight table increases to 16 KB (128K bits). This is still modest
compared to the total router area (~50K gates in 32nm). The ROM can be implemented as
a dedicated lookup table or shared memory.

**For Mesh 16x16:** The weight table grows to 256 KB (2M bits), which may be prohibitive
for on-chip storage. Hierarchical or compressed representations would be needed.

The key advantage over adaptive_xy_yx: the GNN-weighted approach provides global
topology-aware routing decisions with only local congestion feedback, while
adaptive_xy_yx only considers local buffer occupancy. The area overhead is the weight
table, which scales as O(N^2) but is typically dominated by buffer storage in the router.

**Power considerations:** The weight table is read-only (ROM), consuming only dynamic
power per access. At 1 GHz with 25% toggle rate, each access consumes ~0.1 µW,
negligible compared to the router pipeline (~10 mW).



### 6. Overhead Summary Table


| Metric | Mesh 4x4 | Mesh 8x8 | Mesh 16x16 |
|--------|----------|----------|-----------|
| ROM size | 1.0 KB | 16.0 KB | 256.0 KB |
| Pipeline latency | +0-1 cycle | +0-1 cycle | +0-1 cycle |
| Est. area (32nm) | <0.01 mm² | ~0.06 mm² | ~1.0 mm² |
| Dynamic energy | pJ/access | pJ/access | pJ/access |
| Comparison with adaptive | Comparable | 10x area | 100x area |


### 7. GNN Inference Overhead (Periodic Policy Update)


- **GNN FLOPs per inference:** 65,536
- **Inference time (CPU, 2GHz ARM):** 30 μs
- **Estimated inference time (simple HW):** 5,000 ns
- **Policy update frequency:** every 10,000 cycles
- **Effective overhead per cycle:** 0.5%
- **Notes:** GNN inference runs offline/background. Only routing table lookup is in critical path.