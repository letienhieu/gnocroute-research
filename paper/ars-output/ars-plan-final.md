# ARS Lite — Research Plan

> **Chủ đề:** Ứng dụng lý thuyết đồ thị (Graph Theory) và Graph Neural Networks (GNN)
> cho tối ưu hóa cơ chế định tuyến thích nghi trong Network-on-Chip (NoC)
>
> **Tác giả:** Lê Tiến Hiếu — NCS MS: 25218003 — Viện CNTT, ĐHQGHN
> **Ngày:** 14/05/2026
> **Demo cơ sở:** Graph-NOC với NetworkX (7 topologies, centrality analysis, XY routing benchmark)

---

## Vòng 1 — Xác định vấn đề

### Research Question (RQ)

> **Làm thế nào để thiết kế một cơ chế định tuyến thích nghi cho Network-on-Chip dựa trên biểu diễn đồ thị cấu trúc liên kết kết hợp với Graph Neural Networks (GNN) nhằm giảm tắc nghẽn (congestion) và cải thiện độ trễ (latency) so với các giải pháp định tuyến tĩnh hiện tại?**

**RQ con:**
1. RQ1 — Mô hình đồ thị nào (directed/weighted/dynamic) biểu diễn trạng thái NoC một cách hiệu quả nhất cho bài toán định tuyến?
2. RQ2 — Kiến trúc GNN nào (GCN, GAT, MPNN) phù hợp để học đặc trưng tắc nghẽn từ cấu trúc liên kết NoC?
3. RQ3 — Cơ chế kết hợp GNN encoder với Deep Reinforcement Learning (DRL) agent cho quyết định định tuyến theo thời gian thực có khả thi trên phần cứng NoC không?

### Why Now

1. **Kết quả từ demo Graph-NOC (đã chạy):**
   - Betweenness centrality phân bố không đều: Mesh 4×4 có node trung tâm BC ≈ 0.25, Fat-Tree có aggregation switch BC ≈ 0.42 → đây là các bottleneck tiềm năng.
   - XY routing (deterministic) cho mesh 8×8 đạt hop count tối ưu nhưng congestion imbalance lên đến 0.475 — tức gần một nửa số link chịu tải không đồng đều.
   - Điều này chứng minh *deterministic routing không đủ* cho workload hiện đại.

2. **Bối cảnh công nghệ 2024–2026:**
   - NoC đã trở thành interconnect chuẩn cho multi-core SoC, AI accelerators (Google TPU, NVIDIA Grace Hopper).
   - GNN đã trưởng thành (PyTorch Geometric, DGL) và bắt đầu xuất hiện trong research về mạng truyền thông.
   - Các nghiên cứu gần đây (2023–2025) cho thấy GNN outperforms MLP trong việc học cấu trúc topology.

3. **Khoảng trống:**
   - Chưa có công trình nào kết hợp GNN encoder + DRL cho adaptive routing trên NoC một cách có hệ thống với đánh giá trên nhiều topology chuẩn.

### Scope

| Dimension | Scope | Out of scope |
|-----------|-------|-------------|
| Topology | Mesh, Torus, Fat-Tree, Small-World (4 loại tiêu biểu) | Ring, Random (đã demo nhưng ít ứng dụng) |
| Routing | Adaptive routing (congestion-aware, GNN-DRL driven) | Circuit switching, wormhole flow control details |
| Metric | Latency, throughput, congestion imbalance, energy | Area overhead, thermal |
| Validation | Simulation (Noxim/BookSim2 + PyTorch Geometric) | FPGA/RTL implementation |
| GNN type | GCN, GAT, MPNN (so sánh chọn 1) | Graph Transformers (quá nặng cho NoC) |

---

## Vòng 2 — Literature Gap

### Existing Work

| Hướng nghiên cứu | Công trình tiêu biểu | Hạn chế |
|------------------|---------------------|---------|
| **Adaptive routing cổ điển** | DyAD (2004), NoC-Deflection (2021), Region-based routing | Dựa trên threshold/local info, không học được long-term pattern |
| **DRL cho NoC routing** | MAAR (2022, IEEE TC), GARNN (2023) | Dùng MLP cho state encoding, bỏ qua cấu trúc topology |
| **GNN cho mạng truyền thông** | GNN for routing in computer networks (Rusek 2024), GNN-RL for SDN (2023) | Chỉ áp dụng cho WAN/SDN, không cho NoC (latency constraint khác) |
| **Graph theory + NoC** | Centrality-based mapping (2020), deadlock-free routing from graph coloring | Chỉ dùng graph static properties, không real-time adaptive |
| **GNN cho Chip Design** | Google Circuit GNN (Nature 2021), PRICE (2024) | Hướng placement/routing vật lý, không phải routing logic |

### Gap Analysis

```
Existing works:
├── Adaptive routing cổ điển
│   └── Thiếu khả năng học, phụ thuộc threshold cứng
├── DRL routing
│   └── MLP state encoding → mất thông tin cấu trúc
├── GNN for WAN/SDN
│   └── Latency constraint khác, không áp dụng trực tiếp cho NoC
└── Graph theory for NoC
    └── Static analysis, không real-time
                      │
                      ▼
          GNN + DRL cho adaptive NoC routing
          (kết hợp graph structure learning + RL decision)
```

**Gap chính:**
1. Chưa có nghiên cứu nào thiết kế GNN encoder *chuyên biệt cho NoC topology* kết hợp với DRL agent.
2. Các DRL routing hiện tại dùng MLP để encode state → mất thông tin connectivity và spatial locality.
3. Chưa có benchmark so sánh có hệ thống giữa GNN variants trên multiple NoC topologies.

### Contribution Claim

1. **Mô hình hóa:** Đề xuất biểu diễn NoC state dưới dạng *dynamic weighted graph* trong đó node features bao gồm: buffer occupancy, injection rate; edge features bao gồm: link utilization, latency.

2. **Kiến trúc:** Thiết kế *GNN-Encoder + DRL-Agent* — GNN encoder học spatial-temporal embedding của NoC, DRL agent (PPO/DQN) đưa ra quyết định routing dựa trên embedding này.

3. **Đánh giá:** Xây dựng framework mô phỏng tích hợp Noxim/BookSim2 + PyTorch Geometric; benchmark trên 4 topologies với SPLASH-2/PARSEC traces + synthetic traffic.

4. **So sánh:** Đối sánh với 4 baselines: XY routing, DyAD, MAAR (DRL+MLP), Regional Adaptive.

---

## Vòng 3 — Methodology

### Method Choice

#### Kiến trúc đề xuất

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│  NoC State   │───▶│  GNN Encoder     │───▶│  DRL Agent   │───▶ Routing decision
│  (Graph)     │    │  (GCN/GAT/MPNN)  │    │  (PPO/DQN)   │
└──────────────┘    └──────────────────┘    └──────────────┘
       ▲                                            │
       │            ┌──────────────────┐            │
       └────────────┤ Reward Function  │◄───────────┘
                    │(latency + bal.)  │
                    └──────────────────┘
```

#### Lựa chọn cụ thể

| Thành phần | Lựa chọn | Lý do |
|-----------|---------|-------|
| **Graph representation** | Directed weighted multigraph | NoC links có direction, buffer state biến thiên theo thời gian |
| **GNN variant** | **GAT (Graph Attention)** — thử nghiệm so sánh với GCN, MPNN | Attention weights cho phép model tập trung vào node đang congestion |
| **Message passing** | 2–3 layers, 64–128 hidden dim | NoC topology không quá sâu (Mesh 4×4 có diameter=6) |
| **RL algorithm** | **PPO** (so sánh với DQN nếu action space nhỏ) | PPO ổn định, sample-efficient, phù hợp routing decision |
| **Action space** | Output port selection (5 actions: N/S/E/W/local — cho 2D Mesh) | Phù hợp adaptive routing tiêu chuẩn |
| **Reward** | R = -α·latency - β·congestion_imbalance - γ·hop_penalty | Đa mục tiêu: latency + balance + efficiency |

#### Data flow cụ thể

1. **t = 0:** Noxim/BookSim2 khởi tạo topology, traffic pattern.
2. **t → t+1:** State = graph với node features (buffer_occ, inj_rate) và edge features (link_util, pkt_latency).
3. **GNN Encoder** nhận graph → output node embeddings (128-dim).
4. **DRL Agent** nhận embedding của current node → chọn output port.
5. **Simulator** thực thi routing → tính reward → update state.
6. **Train:** PPO update policy dựa trên trajectory.

### Validation Strategy

| Level | Phương pháp | Metric |
|-------|------------|--------|
| **1. Synthetic** | Uniform random, transpose, bit-reversal traffic | Avg. latency, saturation throughput |
| **2. Benchmark** | SPLASH-2 (FFT, LU, Barnes), PARSEC (blackscholes, dedup) | Completion time, energy-delay product |
| **3. Ablation** | Replace GNN→MLP, GCN→GAT, remove node/edge features | Delta performance |
| **4. Sensitivity** | Vary injection rate, topology size (4×4 → 8×8 → 16×16) | Scaling behavior |

#### Baselines

1. **XY routing** — deterministic, baseline tối thiểu
2. **DyAD** — adaptive routing cổ điển (threshold-based)
3. **MAAR** — DRL + MLP (SOTA 2022)
4. **Regional Adaptive** — regional congestion-aware (heuristic)

### Risks

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|-----------|
| **GNN inference latency quá cao** cho real-time routing decision | Medium | High | Quantization, knowledge distillation, early exit; benchmark inference time per decision target: <100 cycles |
| **Training không converge** với sparse reward | Medium | High | Reward shaping, curriculum learning (bắt đầu với topology nhỏ) |
| **Overfitting** trên synthetic traffic | Medium | Medium | Cross-traffic evaluation, regularization, diverse training traces |
| **Deadlock** khi adaptive routing | Low | Critical | Deadlock-free subchannel mechanism; routing algorithm đảm bảo acyclic |
| **So sánh không công bằng** với baseline | Low | Medium | Thống nhất simulator, warm-up cycles, confidence interval (95%) |

---

## Vòng 4 — Tổng kết & Research Plan

### Paper Structure (dự kiến)

```
Title: Graph Neural Network-Enhanced Adaptive Routing for Network-on-Chip

1. Introduction
   - NoC bottleneck problem, motivation từ demo kết quả
   - Research Question + Contribution

2. Background & Related Work
   - NoC routing adaptive (DyAD → DRL-based)
   - GNN fundamentals (GCN, GAT, MPNN)
   - Gap synthesis

3. Proposed Method: GNN-DRL Router
   3.1 Graph representation of NoC
   3.2 GNN encoder architecture
   3.3 DRL agent (PPO) design
   3.4 Training procedure

4. Experimental Setup
   4.1 Simulation framework (Noxim + PyTorch Geometric)
   4.2 Topologies, traffic patterns, baselines
   4.3 Evaluation metrics

5. Results & Analysis
   5.1 Overall performance comparison
   5.2 Ablation study
   5.3 Scalability analysis
   5.4 Inference overhead

6. Discussion
   - Trade-offs, limitations, practical feasibility

7. Conclusion & Future Work
```

### Timeline (dự kiến 6 tháng)

| Giai đoạn | Thời gian | Milestone |
|-----------|----------|-----------|
| **GĐ 1: Framework** | Tháng 1–2 | Tích hợp Noxim/BookSim2 + PyG; implement graph representation |
| **GĐ 2: Prototype** | Tháng 2–3 | Implement GNN encoder (GCN, GAT, MPNN); DRL agent (PPO) |
| **GĐ 3: Training** | Tháng 3–4 | Train trên synthetic traffic; tuning hyperparameters |
| **GĐ 4: Evaluation** | Tháng 4–5 | Benchmark trên SPLASH-2/PARSEC; ablation experiments |
| **GĐ 5: Writing** | Tháng 5–6 | Draft paper + submission preparation |

### Target Venues

1. **Primary:** IEEE Transactions on Computers (TC) / IEEE Access / JSA
2. **Conference:** NoCArc (workshop at MICRO), ISPASS, DATE
3. **Fallback:** VNU Journal of Computer Science and Communication Engineering

### Next Steps (immediate)

- [ ] **A1:** Xác nhận simulation framework — Noxim (SystemC) vs Booksim2 (C++) vs gem5. Noxim có Python API, dễ tích hợp PyG nhất.
- [ ] **A2:** Implement graph state collector module trong Noxim — export topology + node/edge state ra Python object.
- [ ] **A3:** Thiết kế GNN encoder với PyTorch Geometric — bắt đầu với GCN 2-layer, 64-dim.
- [ ] **A4:** Implement PPO agent dùng Stable-Baselines3 hoặc custom implementation.
- [ ] **A5:** Chạy proof-of-concept trên Mesh 4×4 với uniform traffic — verify convergence.

---

## Phụ lục — Kết quả từ Demo Graph-NOC (đã chạy)

| Topology | Nodes | BC Max | Congestion Imbalance | Ghi chú |
|----------|-------|--------|---------------------|---------|
| Mesh 4×4 | 16 | 0.25 (node 5,6,9,10) | 0.382 | XY routing imbalance rõ rệt |
| Mesh 8×8 | 64 | 0.18 (center cluster) | **0.475** | Imbalance cao nhất — cần adaptive routing nhất |
| Torus 4×4 | 16 | 0.13 (distributed) | 0.251 | Torus tự cân bằng hơn |
| Ring 8 | 8 | 0.21 (all equal) | 0.122 | Ring đơn giản, ít congestion |
| Fat-Tree 4-ary | 32 | **0.42** (aggregation switch) | 0.341 | Aggregation switch là bottleneck rõ |
| Small-World | 16 | 0.31 (hub nodes) | 0.298 | Hub-dependent |
| Random | 16 | 0.28 | 0.321 | Không có cấu trúc |

> **Kết luận từ demo:** Mesh 8×8 và Fat-Tree có imbalance cao nhất — đây là target topology chính cho đề xuất GNN-DRL. Congestion imbalance 0.475 (Mesh 8×8) là baseline cần cải thiện.

---

*Tài liệu này được xây dựng theo phương pháp ARS Lite — Socratic dialogue.*
*Hoàn thành ngày 14/05/2026.*
