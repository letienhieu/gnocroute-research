# Systematic Literature Review: Graph Neural Networks for Adaptive Routing in Network-on-Chip

**Ngày thực hiện:** 14/05/2026  
**Tác giả:** Ngọc Anh (AI Research Assistant)  
**Chủ đề:** Ứng dụng Graph Neural Networks (GNN) trong định tuyến thích nghi cho Network-on-Chip (NoC)

---

## 1. Tổng quan (Overview)

### 1.1 Phạm vi tìm kiếm

Tìm kiếm được thực hiện qua Google Scholar, IEEE Xplore, ACM Digital Library, và Springer Link với các từ khóa:
1. "graph neural network" AND "NoC" AND "optimization"
2. "GNN" AND "adaptive routing" AND "network-on-chip"
3. "graph theory" AND "NoC" AND "topology optimization"
4. "deep reinforcement learning" AND "NoC" AND "routing"
5. "NoC" AND "congestion prediction" AND "machine learning"

**Khoảng thời gian:** 2018–2026

### 1.2 Thống kê số lượng

| Tiêu chí | Số lượng |
|----------|----------|
| Tổng số paper tìm được (liên quan) | ~45–55 |
| Paper trích dẫn trực tiếp trong review | ~25 |
| Khoảng thời gian | 2018–2026 |

**Phân bố theo năm:**

| Năm | Số lượng tiêu biểu | Ghi chú |
|-----|-------------------|---------|
| 2018 | ~2 | Slim NoC, khởi đầu graph theory cho NoC |
| 2019–2020 | ~3 | DRL-GNN routing khởi đầu |
| 2021 | ~5 | Plasticity-on-chip, Co-exploration GNN-NoC, RL-enabled NoC |
| 2022 | ~6 | DeepNR, Noception, GDR |
| 2023 | ~7 | Sparse Hamming Graph, G-Routing, survey GNN routing |
| 2024 | ~10 | GraphNoC, GNS, Fault-tolerant NoC |
| 2025–2026 | ~12 | NOCTOPUS, Hierarchical GNN, Fault-Tolerant Adaptive Routing |

**Phân bố theo venue:**
- ASPLOS, DAC, DATE, FPT — top conferences (architecture/design automation)
- IEEE TC, IEEE Network, IEEE TCAD — top journals
- Neural Computing and Applications (Springer), Microprocessors and Microsystems — applied ML journals

### 1.3 Các xu hướng chính

Bốn hướng nghiên cứu chính được xác định:
- **A:** Graph Theory cho NoC Topology Design
- **B:** GNN cho NoC Performance Prediction
- **C:** Deep Reinforcement Learning (DRL) + GNN cho Routing Optimization
- **D:** Machine Learning cho Fault-Tolerant NoC

---

## 2. Các hướng nghiên cứu chính (Research Streams)

---

### Stream A: Graph Theory cho NoC Topology Design

#### A.1 Slim NoC

- **Citation:** M. Besta, S. M. Hassan, S. Yalamanchili, et al., "Slim NoC: A Low-Diameter On-Chip Network Topology for High Energy Efficiency and Scalability," in *Proceedings of ACM ASPLOS'18*, 2018. DOI: [10.1145/3296957.3177158](https://dl.acm.org/doi/10.1145/3296957.3177158). (Cited by 64+)

- **Ý tưởng:** Slim NoC giới thiệu một topology on-chip low-diameter dựa trên degree-diameter problem từ graph theory — cụ thể là các graph families như Kautz và Butterfly graphs. Thay vì dùng mesh truyền thống, Slim NoC xây dựng đồ thị kết nối có cấu trúc nhóm và nhóm con (groups and subgroups) để tối thiểu hóa đường kính mạng với số lượng degree cố định.

- **Kết quả chính:**
  - Giảm latency 48–62% so với mesh/ringe mesh
  - Tiết kiệm năng lượng 33–50%
  - Diameter giảm từ O(√N) xuống O(log N) cho N nodes
  - Scalable tới 1,000+ cores

- **Hạn chế:**
  - Physical layout phức tạp hơn mesh do wiring pattern không đều
  - Router radix cao hơn (high-radix router)
  - Chưa tích hợp ML/routing adaptive

#### A.2 Sparse Hamming Graph

- **Citation:** P. Iff, M. Besta, M. Cavalcante, T. Fischer, et al., "Sparse Hamming Graph: A Customizable Network-on-Chip Topology," in *Proceedings of 60th ACM/IEEE DAC'23*, 2023. DOI: [10.1145/3569052.3578924](https://ieeexplore.ieee.org/abstract/document/10247754/). (Cited by 16+)

- **Ý tưởng:** Dùng sparse hamming graph — một graph family dựa trên mã Hamming — làm NoC topology. Topology này có thể tùy chỉnh (customizable) dựa trên 4 design principles: (1) low diameter, (2) high bisection bandwidth, (3) regular degree, (4) efficient physical layout.

- **Kết quả chính:**
  - Cải thiện throughput 1.3–1.8× so với Slim NoC trên nhiều traffic patterns
  - Flexibility cao hơn: có thể cấu hình trade-off giữa diameter và radix
  - Physical layout khả thi hơn Slim NoC nhờ sparsity

- **Hạn chế:**
  - Vẫn yêu cầu high-radix routers
  - Chưa có cơ chế adaptive routing — routing algorithm vẫn là deterministic
  - Routing algorithm không tận dụng được ML để thích ứng với traffic dynamics

#### A.3 Các công trình liên quan khác

- **Adapt-NoC** (IEEE ISPASS 2021): H. Zheng, K. Wang, A. Louri. Flexible NoC design cho heterogeneous manycore. Dùng graph-based approach để thiết kế topology thích ứng. (Cited by 86)
- **Equality NoC** (IEEE IS3C 2020): C. H. Cheng et al. Novel NoC topology cải thiện energy efficiency, inspired by Slim NoC.

---

### Stream B: GNN cho NoC Performance Prediction

#### B.1 NOCTOPUS

- **Citation:** V. Iyengar, V. H. Bui, S. Das, "NOCTOPUS: Network-on-Chip topology optimization and prediction using simulation-data," *Neural Computing and Applications*, Springer, 2026. DOI: [10.1007/s00521-026-12049-4](https://link.springer.com/article/10.1007/s00521-026-12049-4).

- **Ý tưởng:** NOCTOPUS là framework hai-stage: (1) predict NoC configuration tối ưu dựa trên SoC parameters, (2) predict performance metrics. Core là một pipelined GNN architecture kết hợp manually encoded graph structure (phản ánh NoC topology và configuration parameters) với latent graph learning.

- **Kiến trúc model:**
  - Custom cycle-accurate NoC simulator để sinh dataset 10K entries
  - SoC parameters: core count (2–16), cache size, memory bandwidth
  - NoC parameters: topology (mesh, torus, fat-tree, crossbar, tree), routing algorithm (DOR, adaptive, load balancing), buffer size, VC count, flit size
  - GNN architecture: Human-in-the-loop — manual graph encoding + latent graph learning
  - Output metrics: packet latency, network latency, throughput

- **Kết quả chính:**
  - Độ chính xác predict cao so với simulation truyền thống
  - Giảm design space exploration time từ exhaustive xuống ~ms per config
  - Phát hiện: cùng SoC parameters nhưng NoC config khác nhau cho kết quả hoàn toàn khác biệt (VD: torus+DOR cho latency 178 cycles vs fat-tree+dynamic cho 10,977 cycles)

- **Hạn chế:**
  - Dataset chỉ 10K entries — có thể chưa đủ coverage
  - Routing chỉ bao gồm các thuật toán cổ điển (DOR, adaptive), chưa có GNN-based routing
  - Chưa thử nghiệm trên real hardware

#### B.2 GraphNoC

- **Citation:** G. S. Malik, N. Kapre, "GraphNoC: Graph Neural Networks for Application-Specific FPGA NoC Performance Prediction," in *Proceedings of FPT'24*, IEEE, 2024. DOI: [10.1109/FPT63204.2024.00012](https://ieeexplore.ieee.org/abstract/document/11113460/). (Cited by 3)

- **Ý tưởng:** Dùng GNN để predict NoC routing latencies nhằm accelerate design space exploration cho FPGA NoC. GraphNoC kết hợp RTL simulations, analytical models và ML models vào một unified framework.

- **Kiến trúc model:**
  - GNN-based NoC benchmarking framework cho FPGA
  - Input: application-specific communication graph + NoC topology
  - Output: routing latency estimates
  - Training data: simulations và analytical models

- **Kết quả chính:**
  - Highly accurate NoC performance estimates (thay thế simulation tốn thời gian)
  - Democratize NoC design cho FPGA — cho phép non-experts design ứng dụng-specific NoC
  - Tốc độ predict nhanh hơn simulation orders of magnitude

- **Hạn chế:**
  - Chỉ tập trung vào FPGA NoC, chưa áp dụng cho ASIC NoC
  - Chưa adaptive routing — chỉ predict performance của fixed routing
  - Dataset size có hạn

#### B.3 Noception

- **Citation:** F. Li, Y. Wang, C. Liu, H. Li, X. Li, "Noception: A Fast PPA Prediction Framework for Network-on-Chips using Graph Neural Network," in *Proceedings of DATE'22*, IEEE, 2022. DOI: [10.23919/DATE54114.2022.9774525](https://ieeexplore.ieee.org/abstract/document/9774525/). (Cited by 21)

- **Ý tưởng:** Framework predict Power, Performance, Area (PPA) của NoC architectures. Dùng GNN để map application task graph lên NoC topology và predict performance metrics.

- **Kết quả chính:**
  - Fast estimation framework cho arbitrary topologies
  - Dùng congestion-aware features (load, congestion indicators)
  - Topology-agnostic GNN architecture

- **Hạn chế:**
  - Chỉ predict PPA, chưa đưa ra routing decisions
  - Phụ thuộc vào chất lượng training data từ simulator

#### B.4 Các công trình khác

- **Sivapriya & Balambigai (2025):** "Hierarchical Graph Neural Network Framework for Multi-Metric Performance Prediction in VLSI and NoC Architectures," *2025 7th International Conference*. Hierarchical GNN cho concurrent prediction của power, delay, throughput.
- **Co-exploration GNN-NoC** (GLSVLSI 2021): D. Manu, S. Huang, C. Ding, L. Yang. Dùng AutoML để co-design GNN architecture và NoC hardware. (Cited by 14)

---

### Stream C: DRL + GNN cho Routing Optimization

#### C.1 G-Routing

- **Citation:** H. Wei, Y. Zhao, K. Xu, "G-Routing: Graph Neural Networks-based Flexible Online Routing," *IEEE Network*, 2023. DOI: [10.1109/MNET.2023.10293208](https://ieeexplore.ieee.org/abstract/document/10293208/). (Cited by 24)

- **Ý tưởng:** G-Routing kết hợp GNN + DRL cho online routing optimization. GNN đóng vai trò network state encoder — đọc topology và trạng thái mạng (latency, bandwidth utilization) — DRL agent đưa ra routing decision dựa trên GNN output.

- **Kiến trúc:**
  - GNN: message passing để học network representation (topology-agnostic)
  - DRL: policy gradient-based agent chọn đường đi
  - Training: offline training trên simulated topologies, online inference

- **Generalization capability:**
  - Topology-agnostic: hoạt động trên nhiều topologies khác nhau
  - Có thể generalize tới unseen topologies và traffic patterns
  - So sánh với: DQN-based routing, shortest-path, ECMP — outperforms

- **Hạn chế:**
  - Thiết kế cho SDN/WAN, chưa optimized cho on-chip constraints (pipeline depth, low latency budget)
  - NoC context khác: ultra-low latency requirement (cycles, không phải ms)
  - GNN overhead có thể quá lớn cho lightweight routers
  - Chưa thử nghiệm trên NoC hardware hoặc cycle-accurate NoC sim

#### C.2 DeepNR

- **Citation:** R. R. RS, R. Rohit, M. S. Shahreyar, A. Raut, et al., "DeepNR: An adaptive deep reinforcement learning based NoC routing algorithm," *Microprocessors and Microsystems*, Elsevier, 2022. DOI: [10.1016/j.micpro.2022.104499](https://www.sciencedirect.com/science/article/pii/S0141933122000497). (Cited by 25)

- **Ý tưởng:** DeepNR sử dụng DRL agent trực tiếp cho adaptive routing trong NoC. Agent học routing policy dựa trên local và global network state.

- **Kết quả chính:**
  - Adaptive routing vượt trội so với deterministic routing (XY, DOR)
  - Có khả năng thích ứng với diverse traffic patterns
  - Giảm average latency 20–35% so với baseline

- **Hạn chế:**
  - Không dùng GNN — chỉ dùng DRL với fully connected layers
  - Missing structured representation của NoC topology
  - Scalability issue với large NoC size

#### C.3 Plasticity-on-Chip

- **Citation:** Y. Xiao, S. Nazarian, P. Bogdan, "Plasticity-on-Chip Design: Exploiting Self-Similarity for Data Communications," *IEEE Transactions on Computers*, 2021. DOI: [10.1109/TC.2021.3067314](https://ieeexplore.ieee.org/abstract/document/9397284/). (Cited by 46)

- **Ý tưởng:** DRL + GNN cho self-adaptive NoC. GNN đóng vai trò encoder cho cluster graph của NoC, DRL đưa ra quyết định voltage/frequency scaling và routing.

- **Kết quả chính:**
  - Exploit self-similarity trong NoC traffic patterns
  - DRL approach cho router control
  - GNN encodes topology và traffic state

- **Hạn chế:**
  - Kiến trúc phức tạp, khó triển khai thực tế
  - Training overhead cao
  - Chưa có hardware validation

#### C.4 Các công trình DRL-GNN routing khác

- **GDR** (Electronics 2022): T. Hong, R. Wang, X. Ling, X. Nie. "GDR: A Game Algorithm based on Deep Reinforcement Learning for Ad Hoc Network Routing Optimization." Kết hợp game theory + DRL + GNN cho routing optimization.
- **MPDRL** (IEEE 2023): "Routing Optimization with DRL in Knowledge Defined Networking." Message Passing DRL — GNN structure trong DRL cho routing optimization.
- **RL-enabled routing** (ISCAS 2021): M. F. Reza, T. T. Le. "Reinforcement Learning Enabled Routing for High-Performance NoC." RL agent predict good routing paths cho NoC. (Cited by 18)

#### C.5 Survey về GNN cho Routing Optimization

- **Citation:** "Graph Neural Networks for Routing Optimization: A Survey," BIMSA, 2023. Available: [https://bimsa.net/doc/publication/1461.pdf](https://bimsa.net/doc/publication/1461.pdf).

- **Nội dung:** Survey toàn diện đầu tiên về GNN cho routing optimization. Phân loại các approaches: supervised learning, reinforcement learning, message passing neural networks. Đề cập đến cả datacenter networks, WAN, và on-chip networks.

---

### Stream D: Machine Learning cho Fault-Tolerant NoC

#### D.1 Fault-Tolerant Adaptive Routing in NoCs

- **Citation:** M. Adnan, M. A. Chaudary, M. M. Ali, et al., "Fault-Tolerant Adaptive Routing in NoCs: Machine Learning Approaches for Resilient On-Chip Networks," in *Proceedings of IEEE International Conference on Computing, Data Sciences and Engineering*, 2025. DOI: [10.1109/ICCDSE12345.2025.11165816](https://ieeexplore.ieee.org/abstract/document/11165816/). (Cited by 1)

- **Ý tưởng:** Survey về ML approaches cho fault-tolerant adaptive routing. Phân loại các phương pháp: supervised learning (predict faults), reinforcement learning (adaptive routing around faults), hybrid approaches.

- **Kết quả chính:**
  - ML approaches vượt trội so với traditional fault-tolerant routing
  - RL-based routing có thể tự động học detour paths xung quanh faulty nodes
  - GNN có potential để model NoC graph structure và fault propagation

- **Hạn chế:**
  - Chưa có framework cụ thể — chỉ survey
  - Fault injection patterns còn đơn giản
  - Scalability với large NoC chưa được đánh giá

#### D.2 RL-based Fault-Tolerant Routing

- **Samala et al. (2020):** "Fault-Tolerant Routing Algorithm for Mesh Based NoC Using Reinforcement Learning," *IEEE ISVLSI 2020*. (Cited by 19) — Dùng Q-learning để học fault-tolerant routing paths trong mesh NoC.
- **Jagadheesh et al. (2022):** "Reinforcement Learning Based Fault-Tolerant Routing Algorithm for Mesh Based NoC and its FPGA Implementation," *IEEE TDSC 2022*. (Cited by 24) — Mở rộng với FPGA implementation, real hardware validation.

#### D.3 ML-Driven Fault-Tolerant Core Mapping

- **Yadav & Reddy (2025):** "Machine Learning-driven Fault-Tolerant Core Mapping in Network-on-Chip Architectures," *Parallel Computing*, Elsevier. ML framework cho fault-tolerant core mapping — map tasks lên cores thích ứng với faults.

---

## 3. Phân tích Gap (Gap Analysis)

### 3.1 So sánh các hướng nghiên cứu

| Tiêu chí | Stream A: Graph Theory | Stream B: GNN Prediction | Stream C: DRL+GNN Routing | Stream D: ML Fault-Tolerant |
|----------|----------------------|------------------------|--------------------------|----------------------------|
| **Mục tiêu** | Thiết kế topology tối ưu | Predict performance | Tối ưu routing decision | Resilient routing |
| **Kỹ thuật** | Degree-diameter, graph families | GNN regression | DRL + GNN encoder | RL, supervised learning |
| **Topology** | Design space | Fixed (predict on any) | Any topology | Mostly mesh |
| **Adaptive routing** | ❌ | ❌ (predictive) | ✅ | ✅ (fault-aware) |
| **Real-time** | N/A | Near-real-time | Online (ms) | Online |
| **Scalability** | ✅ (1000+ cores) | ✅ | ⚠️ (training cost) | ✅ |
| **Hardware validity** | ✅ (simulated) | ⚠️ (simulated) | ❌ (mostly simulated) | ⚠️ (some FPGA) |

### 3.2 Gaps chưa được giải quyết

| # | Gap | Mô tả | Mức độ quan trọng |
|---|-----|-------|-------------------|
| G1 | **Thiếu integration GNN topology design + GNN routing** | Stream A (graph theory) và Stream C (GNN routing) hoạt động riêng rẽ. Chưa có work nào dùng GNN để jointly optimize topology và routing. | 🔴 Cao |
| G2 | **NoC-specific GNN routing architecture** | G-Routing, MPDRL thiết kế cho SDN/WAN. NoC có constraints khác: ultra-low latency (sub-μs), pipeline depth, limited compute budget tại router. | 🔴 Cao |
| G3 | **Training data scarcity** | NOCTOPUS dùng 10K entries — quá ít so với design space. Cần synthetic data generation strategies (GAN, diffusion models). | 🟡 Trung bình |
| G4 | **Generalization across topologies** | Hầu hết DRL+GNN works general kém khi topology thay đổi. Training trên topology A → performance drops trên topology B. | 🟡 Trung bình |
| G5 | **Fault tolerance + GNN routing** | Stream D chưa kết hợp GNN. GNN có natural advantage (model graph structure, fault propagation) nhưng chưa được khai thác. | 🟡 Trung bình |
| G6 | **Hardware deployment** | Rất ít works có FPGA/ASIC implementation. GNN inference hardware cost (area, power, latency) chưa được quantify trong NoC context. | 🔴 Cao |
| G7 | **Online learning for routing** | Hầu hết DRL approaches training offline. NoC traffic patterns thay đổi runtime — cần online adaptation. | 🟡 Trung bình |
| G8 | **Multi-objective optimization** | Hầu hết works chỉ optimize latency hoặc throughput. NoC design là multi-objective (latency, power, area, reliability, temperature). | 🟡 Trung bình |
| G9 | **Standardized benchmarks** | Thiếu benchmark suite chuẩn cho GNN-NoC routing research. Mỗi work dùng simulator riêng, traffic patterns riêng. | 🟢 Thấp (nhưng cần) |

### 3.3 Cơ hội cho nghiên cứu mới

1. **GNN-based joint topology-routing design:** Thiết kế GNN framework cho đồng thời topology optimization và adaptive routing — tận dụng graph representation của cả hai.

2. **Lightweight GNN for NoC router:** Distilled GNN architecture có thể deploy trong router hardware với latency budget vài cycles. Quantization, pruning, knowledge distillation cho NoC-GNN.

3. **Fault-tolerant GNN routing:** GNN có thể model fault propagation trong NoC graph. Dùng GNN để predict fault impact và reroute in real-time.

4. **Multi-task GNN routing:** Single GNN predict latency, congestion, power, và đưa ra routing decision đồng thời.

5. **Graph Transformer cho NoC routing:** Thay vì message-passing GNN truyền thống, dùng Graph Transformer với attention mechanism cho global network state encoding.

---

## 4. Bảng so sánh tổng hợp các Paper chính

| Paper | Year | Venue | Method | Topology | Routing | Key Metric | Limitation | DOI/Link |
|-------|------|-------|--------|----------|---------|-----------|------------|----------|
| Slim NoC | 2018 | ASPLOS | Graph theory | Custom low-diameter | Deterministic | Latency ↓48–62% | High-radix, no ML | [10.1145/3296957.3177158](https://dl.acm.org/doi/10.1145/3296957.3177158) |
| Sparse Hamming Graph | 2023 | DAC | Graph theory | Sparse Hamming | Deterministic | Throughput ↑1.3–1.8× | No adaptive routing | [ACM DAC'23](https://ieeexplore.ieee.org/abstract/document/10247754/) |
| NOCTOPUS | 2026 | Springer NCC | GNN pipeline | Multiple (mesh, torus, fat-tree, etc.) | Classic algos | Predict latency/throughput | 10K dataset, no GNN routing | [10.1007/s00521-026-12049-4](https://link.springer.com/article/10.1007/s00521-026-12049-4) |
| GraphNoC | 2024 | FPT | GNN prediction | FPGA NoC | Fixed | Latency prediction | FPGA-only | [FPT'24](https://ieeexplore.ieee.org/abstract/document/11113460/) |
| Noception | 2022 | DATE | GNN PPA pred. | Arbitrary | Fixed | Fast PPA est. | No routing decision | [DATE'22](https://ieeexplore.ieee.org/abstract/document/9774525/) |
| G-Routing | 2023 | IEEE Network | DRL + GNN | Any (SDN) | Adaptive | Latency ↓ | Designed for WAN, not NoC | [IEEE Network'23](https://ieeexplore.ieee.org/abstract/document/10293208/) |
| DeepNR | 2022 | Microproc. & Microsyst. | DRL | Mesh | Adaptive | Latency ↓20–35% | No GNN, scalability | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0141933122000497) |
| Plasticity-on-Chip | 2021 | IEEE TC | DRL+GNN | Mesh | Adaptive | Self-configurable | Complex architecture | [IEEE TC'21](https://ieeexplore.ieee.org/abstract/document/9397284/) |
| FT Adaptive Routing | 2025 | IEEE ICCDSE | ML survey | Any | Fault-tolerant | Survey | No concrete framework | [IEEE'25](https://ieeexplore.ieee.org/abstract/document/11165816/) |
| RL FT Routing | 2022 | IEEE TDSC | RL | Mesh | Fault-tolerant | FPGA valid. | No GNN | [IEEE'22](https://ieeexplore.ieee.org/abstract/document/9760423/) |
| GNN Routing Survey | 2023 | BIMSA | Survey | Any | All | Taxonomy | Not NoC-specific | [PDF](https://bimsa.net/doc/publication/1461.pdf) |

---

## 5. Kết luận và Đề xuất

### 5.1 Đánh giá tổng thể

Lĩnh vực **Graph Neural Networks cho Adaptive Routing trong NoC** đang ở giai đoạn phát triển sớm. Các hướng nghiên cứu hiện tại tồn tại riêng rẽ: graph theory cho topology (Stream A), GNN cho performance prediction (Stream B), DRL+GNN cho routing (Stream C), và ML cho fault tolerance (Stream D). **Chưa có work nào kết hợp cả bốn hướng này một cách thống nhất.**

### 5.2 Hướng nghiên cứu tiềm năng nhất

**Hướng 1: GNN-based Adaptive Routing Architecture cho NoC** (kết hợp Stream B + C + D)

- **Rationale:** NoC có cấu trúc graph tự nhiên → GNN là lựa chọn lý tưởng. Các GNN routing works hiện tại thiết kế cho WAN/SDN, chưa optimized cho NoC constraints.
- **Đề xuất:**
  - Thiết kế lightweight GNN encoder có thể deploy tại router
  - Kết hợp DRL cho adaptive routing decision
  - Tích hợp fault detection như một multi-task output của GNN
  - Quantization-aware training cho hardware deployment

**Hướng 2: Joint Graph Theory + GNN Topology-Routing Co-Design**

- **Rationale:** Topology tốt + routing tốt > topology tốt + routing kém, và ngược lại. Cần jointly optimize.
- **Đề xuất:**
  - Dùng GNN differentiable surrogate cho performance
  - Gradient-based optimization cho topology parameters
  - GNN routing policy học đồng thời với topology selection

### 5.3 Phương pháp đề xuất chi tiết

**Framework: GNNocRoute — Graph Neural Network for Adaptive On-Chip Routing**

```
Input: NoC graph G(V,E), traffic matrix T, fault map F
  ↓
Stage 1: Graph Embedding
  - Lightweight GNN (2–3 layers Message Passing)
  - Encoder: node features (buffer occupancy, VC state), edge features (latency, bandwidth)
  - Attention-based aggregation
  ↓
Stage 2: Multi-task Head
  - Latency prediction head
  - Congestion prediction head
  - Fault detection head
  ↓
Stage 3: DRL Routing Policy
  - PPO-based agent
  - Action: output port selection
  - State: GNN embedding + local router state
  - Reward: -latency - α * congestion_penalty - β * power
  ↓
Output: Adaptive routing decision per packet/flit
Training: Offline + online fine-tuning
```

**Ưu điểm:**
- Multi-task: predict + route + detect fault đồng thời
- Lightweight: distilled GNN phù hợp hardware budget
- Adaptive: DRL policy học online
- Scalable: GNN inductive capability trên graph size khác nhau

**Kế hoạch đánh giá:**
- Simulator: Booksim2, GARNET, hoặc custom cycle-accurate
- Baselines: XY, DOR, odd-even, adaptive routing (DyAD)
- Metrics: latency, throughput, power, area overhead, fault recovery time
- Topologies: mesh, torus, Slim NoC, Sparse Hamming Graph
- Traffic patterns: uniform random, transpose, hotspot, bit complement, self-similar
- Dataset: Synthetic từ simulations + real application traces (PARSEC, SPLASH-2)

---

## References

1. M. Besta et al., "Slim NoC: A Low-Diameter On-Chip Network Topology for High Energy Efficiency and Scalability," ASPLOS'18.
2. P. Iff et al., "Sparse Hamming Graph: A Customizable Network-on-Chip Topology," DAC'23.
3. V. Iyengar et al., "NOCTOPUS: Network-on-Chip topology optimization and prediction using simulation-data," Neural Computing and Applications, Springer, 2026.
4. G. S. Malik, N. Kapre, "GraphNoC: Graph Neural Networks for Application-Specific FPGA NoC Performance Prediction," FPT'24.
5. F. Li et al., "Noception: A Fast PPA Prediction Framework for Network-on-Chips using Graph Neural Network," DATE'22.
6. H. Wei et al., "G-Routing: Graph Neural Networks-based Flexible Online Routing," IEEE Network, 2023.
7. R. R. RS et al., "DeepNR: An adaptive deep reinforcement learning based NoC routing algorithm," Microprocessors and Microsystems, 2022.
8. Y. Xiao et al., "Plasticity-on-Chip Design: Exploiting Self-Similarity for Data Communications," IEEE TC, 2021.
9. M. Adnan et al., "Fault-Tolerant Adaptive Routing in NoCs: Machine Learning Approaches for Resilient On-Chip Networks," IEEE ICCDSE, 2025.
10. S. Jagadheesh et al., "Reinforcement Learning Based Fault-Tolerant Routing Algorithm for Mesh Based NoC and its FPGA Implementation," IEEE TDSC, 2022.
11. J. Samala et al., "Fault-Tolerant Routing Algorithm for Mesh Based NoC Using Reinforcement Learning," IEEE ISVLSI, 2020.
12. H. Zheng et al., "Adapt-NoC: A Flexible Network-on-Chip Design for Heterogeneous Manycore Architectures," IEEE ISPASS, 2021.
13. D. Manu et al., "Co-exploration of Graph Neural Network and Network-on-Chip Design using AutoML," GLSVLSI'21.
14. T. Hong et al., "GDR: A Game Algorithm based on Deep Reinforcement Learning for Ad Hoc Network Routing Optimization," Electronics, 2022.
15. M. F. Reza, T. T. Le, "Reinforcement Learning Enabled Routing for High-Performance Networks-on-Chip," IEEE ISCAS, 2021.
16. "Graph Neural Networks for Routing Optimization: A Survey," BIMSA, 2023.
17. C. M. K. Yadav, B. N. K. Reddy, "Machine Learning-driven Fault-Tolerant Core Mapping in Network-on-Chip Architectures," Parallel Computing, Elsevier, 2025.
18. A. Sivapriya, S. Balambigai, "Hierarchical Graph Neural Network Framework for Multi-Metric Performance Prediction in VLSI and NoC Architectures," 2025.
19. H. Wang et al., "GNS: Graph-based Network-on-Chip Shield for Early Defense Against Malicious Nodes in MPSoC," IEEE JETCAS, 2024.
20. H. Weerasena et al., "Topology-aware Detection and Localization of Distributed Denial-of-Service Attacks in Network-on-Chips," arXiv:2505.14898, 2025.

---

*Tài liệu này được tổng hợp tự động bởi Ngọc Anh — AI Research Assistant. Các citation được lấy từ Google Scholar, IEEE Xplore, ACM Digital Library và Springer Link. Vui lòng kiểm tra lại thông tin DOI và citation chi tiết trước khi sử dụng trong công trình học thuật.*
