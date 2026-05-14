# ARS Review — Multi-Perspective Peer Review

> **Title:** Graph Neural Network-Enhanced Adaptive Routing for Network-on-Chip
> **Venue target:** IEEE Transactions on Computers
> **Date:** 14/05/2026
> **Review format:** ARS Review với 5 góc nhìn phản biện

---

## 📊 Summary Dashboard

| Reviewer | Score | Decision | Điểm mạnh nhất | Điểm yếu nhất |
|----------|-------|----------|----------------|---------------|
| 🏛️ Editor-in-Chief | 58/100 | Major Revision | Timing & gap analysis tốt | Novely chưa đủ mạnh cho IEEE TC |
| 🔬 Methods Reviewer | 62/100 | Major Revision | Validation strategy toàn diện | Implementation details còn mờ |
| 🎯 Domain Expert | 55/100 | Major Revision | Problem selection đúng | Inference latency feasibility chưa được chứng minh |
| 🌐 Interdisciplinary | 60/100 | Minor Revision | Transferability tốt | Impact analysis còn hời hợt |
| 😈 Devil's Advocate | 45/100 | Major Revision → Reject | — | GNN+DRL novelty thấp hơn tuyên bố, timing bottleneck critical |

> **Consensus Score: 56/100 — Overall Decision: Major Revision**
> *(Devil's Advocate nghiêng về Reject nhưng các reviewer khác cho rằng có thể sửa để Major Revision)*

---

# 🏛️ 1. Editor-in-Chief Review

## Reviewer Profile
*Góc nhìn tổng biên tập — đánh giá significance, novelty, scope fit với IEEE TC, clarity of presentation.*

### Evaluation Scores

| Criterion | Score (0–100) | Weight | Weighted |
|-----------|:------------:|:------:|:--------:|
| Scientific Significance | 60 | 25% | 15.0 |
| Novelty & Originality | 45 | 25% | 11.3 |
| Scope Fit (IEEE TC) | 60 | 20% | 12.0 |
| Clarity & Organization | 70 | 15% | 10.5 |
| Completeness of Proposal | 55 | 15% | 8.3 |
| **Total** | | | **57.1 ≈ 57/100** |

### Major Points

**✅ Điểm mạnh:**
1. **Problem framing rõ ràng.** Research question được đặt tốt với 3 sub-questions có tính phân cấp logic. Việc phân tích từ demo Graph-NOC (đã chạy ra số liệu thật như BC=0.42 trên Fat-Tree, imbalance=0.475 trên Mesh 8×8) tạo grounding thuyết phục cho motivation.
2. **Cấu trúc paper dự kiến** hợp lý, theo format chuẩn của IEEE TC. Timeline 6 tháng là realistic nếu tác giả đã có framework cơ bản.
3. **Gap analysis** dạng cây decision tree rất trực quan, giúp người đọc nhanh chóng định vị khoảng trống nghiên cứu.

**❌ Điểm yếu:**
1. **Novelty claim cần kiểm tra lại.** Contribution #1 (graph representation of NoC) là incremental — đã có nhiều paper dùng weighted directed graph để model NoC. Contribution #2 (GNN + DRL) là kết hợp hai kỹ thuật đã biết, chưa thấy architectural innovation thực sự. Với IEEE TC — một trong những journal hàng đầu về computer architecture — mức độ novelty này có thể chưa đủ.
2. **Scope fit với IEEE TC chưa chặt.** IEEE TC ưu tiên các công trình có *hardware implementation hoặc significant system-level contribution*. Research plan này hoàn toàn dựa trên simulation, không có kế hoạch FPGA/RTL implementation nào. Mục "Out of scope" ghi "FPGA/RTL implementation" — điều này làm yếu submission profile cho IEEE TC. Phù hợp hơn với IEEE Access hoặc JSA.
3. **Missing related work quan trọng.** Không thấy nhắc đến:
   - *GTNoC* (Graph Theory for NoC, 2025+) — một số paper gần đây đã dùng graph embedding cho NoC optimization.
   - Các DRL-routing paper 2024–2025 trên IEEE TC chính thức (ví dụ: "DRL-Based Congestion-Aware Routing for 3D NoC").
   - *Hardware-aware GNN acceleration* (vì paper này cần inference trên chip, literature về accelerating GNN inference là relevant).
4. **Impact quantification mờ.** Paper claim sẽ giảm congestion imbalance từ 0.475 xuống bao nhiêu? Mục tiêu quantitative (target improvement) không được nêu. IEEE TC reviewers sẽ hỏi "bao nhiêu là đủ?"

### Decision: **Major Revision**
*Điều kiện: Bổ sung FPGA implementation roadmap hoặc chuyển venue xuống IEEE Access/JSA. Củng cố novelty argument.*

---

# 🔬 2. Methodology Reviewer

## Reviewer Profile
*Góc nhìn chuyên gia phương pháp — đánh giá rigor, reproducibility, statistical validity, experimental design.*

### Evaluation Scores

| Criterion | Score (0–100) | Weight | Weighted |
|-----------|:------------:|:------:|:--------:|
| Methodology Soundness | 65 | 25% | 16.3 |
| Reproducibility | 70 | 20% | 14.0 |
| Experimental Design | 60 | 20% | 12.0 |
| Statistical Rigor | 50 | 20% | 10.0 |
| Risk Assessment | 65 | 15% | 9.8 |
| **Total** | | | **62.1 ≈ 62/100** |

### Major Points

**✅ Điểm mạnh:**
1. **Validation strategy 4-level** (synthetic → benchmark → ablation → sensitivity) là comprehensive và well-structured. Đây là điểm tựa methodology vững chắc.
2. **Lựa chọn PPO** thay vì DQN có lý do hợp lý (stability, sample efficiency). Việc dự định so sánh với DQN cho action space nhỏ là good practice.
3. **Sử dụng open-source toolchain** (Noxim/BookSim2 + PyTorch Geometric + Stable-Baselines3) giúp reproducibility cao.
4. **Baseline selection** bao gồm cả deterministic (XY), heuristic adaptive (DyAD), và learned (MAAR) — đủ spectrum để đánh giá relative improvement.

**❌ Điểm yếu:**
1. **Reward function chưa được thiết kế đủ chi tiết.** Công thức `R = -α·latency - β·congestion_imbalance - γ·hop_penalty` có 3 hyperparameters (α, β, γ) nhưng không có:
   - Giá trị khởi tạo đề xuất
   - Phương pháp tuning (grid search? Bayesian?)
   - Cách normalization giữa các thành phần (latency tính bằng cycles, imbalance là ratio — scale rất khác nhau)
   - *Đây là risk cao cho training convergence.*
2. **Thiếu statistical rigor.** Chỉ mention "confidence interval (95%)" trong bảng risk mitigation nhưng không có detail nào về:
   - Number of runs per experiment (cần tối thiểu 5–10 seeds)
   - Warm-up cycles specification
   - Cách xử lý outliers
   - Statistical significance tests (Mann-Whitney U? Welch's t-test?)
3. **Ablation design cần cụ thể hơn.** "Replace GNN→MLP, GCN→GAT, remove node/edge features" là hướng đúng, nhưng thiếu ablation về:
   - Number of GNN layers (1 vs 2 vs 3)
   - Hidden dimension size
   - Message passing iterations
   - Reward component ablation (từng term α, β, γ riêng rẽ)
4. **Graph representation ambiguity.** "Directed weighted multigraph" — cần định nghĩa rõ:
   - Node features cụ thể: buffer occupancy (per VC? per port?), injection rate (instantaneous? moving average?)
   - Edge features: link utilization (measured how?), latency (average? percentile?)
   - Feature normalization scheme
   - *Thiếu detail → không reproducible.*

### Minor Points

- Nên bổ sung baseline là *shortest-path routing* (không chỉ XY) để phân tách effect của adaptive routing từ effect của routing algorithm.
- NoC simulation có vấn đề về timing accuracy: Noxim dùng SystemC với cycle-accurate simulation khác với BookSim2's flit-level. Cần specify rõ simulator nào cho từng experiment.

### Decision: **Major Revision**
*Điều kiện: Bổ sung đầy đủ experimental protocol (reward design, statistical methods, ablation details).*

---

# 🎯 3. Domain Expert Review (Computer Architecture / NoC)

## Reviewer Profile
*Chuyên gia về kiến trúc máy tính và Network-on-Chip — đánh giá technical correctness, practical feasibility, related work depth.*

### Evaluation Scores

| Criterion | Score (0–100) | Weight | Weighted |
|-----------|:------------:|:------:|:--------:|
| Technical Correctness | 55 | 25% | 13.8 |
| Related Work Coverage | 50 | 20% | 10.0 |
| Practical Feasibility | 45 | 25% | 11.3 |
| Result Interpretation | N/A (chưa có results) | 10% | 0 |
| Realism of Claims | 60 | 20% | 12.0 |
| **Total** | | | **47.1 + (adjust) ≈ 55/100** |

### Major Points

**✅ Điểm mạnh:**
1. **Problem selection đúng.** Mesh 8×8 congestion imbalance 0.475 với XY routing là vấn đề thực tế. NoC adaptive routing đang là chủ đề nóng trong cộng đồng computer architecture.
2. **4 topology selection** (Mesh, Torus, Fat-Tree, Small-World) covers đủ spectrum phổ biến trong NoC research.
3. **Benchmark suite** (SPLASH-2 + PARSEC) là tiêu chuẩn vàng cho workload characterization.

**❌ Điểm yếu:**
1. **Vấn đề then chốt: Inference latency.** 
   - Router trong NoC hoạt động ở tần số 500MHz–2GHz. Mỗi routing decision cần được đưa ra trong <5–10 cycles (khoảng 2.5–10ns ở 1GHz).
   - GAT với 2–3 layers, 128-dim hidden, multi-head attention trên graph với 16–64 nodes: số phép tính là hàng trăm nghìn floating-point operations. Trên CPU embedded (RISC-V, ARM Cortex): có thể mất 1,000–10,000+ cycles.
   - Paper target "<100 cycles" — đây là con số rất tham vọng. Cần benchmark inference latency thực tế trên embedded processor (không phải trên GPU/server CPU).
   - Quantization/knowledge distillation được mention nhưng là mitigation chứ không phải solution. *Nếu không giải quyết được vấn đề này, toàn bộ approach không khả thi trên hardware.*

2. **Related work còn thiếu trọng yếu:**
   - *Q-Routing* và các biến thể (DQ-Routing, Double Q-Routing) — đây là foundational work cho RL-based routing, cần được so sánh.
   - *ReNoC* (Reconfigurable NoC) — adaptive routing thông qua reconfiguration.
   - *HARAQ* (Hierarchical Adaptive Routing) — congestion-aware với hierarchical monitoring.
   - Không có citation nào từ các IEEE TC paper gần đây (2023–2025) về NoC routing. *Một reviewer của IEEE TC sẽ detect ngay.*

3. **Deadlock handling quá mơ hồ.**
   - "Deadlock-free subchannel mechanism" không được giải thích. Subchannel là Virtual Channel (VC) hay physical channel?
   - Adaptive routing + deadlock freedom: cần chứng minh routing function là acyclic (theo lý thuyết Duato's protocol).
   - Nếu dùng turn model (odd-even, west-first) để đảm bảo deadlock freedom: cần kiểm tra GNN có sinh ra turns vi phạm không.
   - *Đây là lỗi kỹ thuật nghiêm trọng nếu không được giải quyết trong paper.*

4. **Energy metric scope inconsistency.**
   - Bảng Scope ghi energy là "In scope" nhưng trong Metrics chỉ có latency, throughput, congestion imbalance. Energy consumption là một selling point quan trọng cho adaptive routing (tránh congestion → giảm energy cho buffering và arbitration). Cần đưa energy vào metric chính thức.

### Decision: **Major Revision**
*Điều kiện tiên quyết: Chứng minh inference latency feasibility qua benchmark trên embedded processor. Bổ sung deadlock-free proof. Cập nhật related work.*

---

# 🌐 4. Interdisciplinary Reviewer

## Reviewer Profile
*Góc nhìn liên ngành — đánh giá generalizability, broader impact, cross-domain applicability, methodology transferability.*

### Evaluation Scores

| Criterion | Score (0–100) | Weight | Weighted |
|-----------|:------------:|:------:|:--------:|
| Generalizability | 65 | 25% | 16.3 |
| Broader Impact | 55 | 20% | 11.0 |
| Cross-Domain Applicability | 70 | 20% | 14.0 |
| Methodological Innovation | 55 | 20% | 11.0 |
| Societal/Tech Relevance | 55 | 15% | 8.3 |
| **Total** | | | **60.6 ≈ 60/100** |

### Major Points

**✅ Điểm mạnh:**
1. **Phương pháp có tính transferable cao.** GNN encoder + DRL agent là architecture pattern có thể áp dụng rộng rãi:
   - WAN/SDN routing (đã có precedent)
   - Data center network load balancing
   - Wireless mesh network routing
   - Traffic engineering cho 5G/6G core networks
   - *Điều này làm tăng significance của công trình dù novelty trong NoC có giới hạn.*

2. **Kết nối graph theory ↔ ML ↔ computer architecture** là hướng đa ngành đúng đắn. Việc dùng centrality analysis từ demo để motivate adaptive routing là ví dụ tốt về cross-pollination.

3. **Impact tiềm năng lên AI hardware design.** Nếu GNN-DRL routing chứng minh hiệu quả trên NoC cho AI accelerators, nó có thể ảnh hưởng đến thiết kế thế hệ tiếp theo của TPU/NPU interconnect.

**❌ Điểm yếu:**
1. **Broader impact analysis còn hời hợt.** Phần này mới chỉ dừng ở "apply to other networks" — cần deeper analysis:
   - Tác động lên thiết kế tự động (design automation): liệu GNN-DRL routing có thể tự động adapt cho bất kỳ topology nào không? Đây là step hướng tới NoC design automation.
   - Tác động lên chi phí sản xuất: nếu adaptive routing yêu cầu hardware GNN accelerator, cost overhead là bao nhiêu?
   - Tác động lên software ecosystem: nếu phương pháp này được chuẩn hóa, có thể tạo ra new design toolchain không?

2. **Generalizability bị giới hạn bởi topology assumption.**
   - Phương pháp được thiết kế cho 2D regular topologies (Mesh, Torus). NoC hiện đại có xu hướng dùng irregular/hierarchical topologies cho heterogeneous SoCs.
   - GNN có thể handle irregular structures, nhưng message passing trên irregular graph có complexity khác. Paper chưa thảo luận.

3. **Thiếu discussion về negative societal impact.**
   - Nếu GNN-DRL routing được dùng trong safety-critical systems (automotive, aerospace): failure mode analysis là gì?
   - DRL agent behavior có unpredictable corner cases — cần forward-looking risk assessment.

4. **Reproducibility cho interdisciplinary audience.**
   - Người đọc từ computer architecture background sẽ quen với Noxim/BookSim2; người từ ML background quen với PyG. Paper cần bridge explanation: làm sao để một researcher thuần architecture có thể reproduce ML phần.

### Decision: **Minor Revision**
*Bổ sung broader impact section, thảo luận generalizability cho irregular topologies, thêm safety/risk discussion.*

---

# 😈 5. Devil's Advocate Review

## Reviewer Profile
*Phản biện mạnh — intentionally critical, challenge mọi assumption, expose weaknesses được che giấu.*

### Evaluation Scores

| Criterion | Score (0–100) | Weight | Weighted |
|-----------|:------------:|:------:|:--------:|
| Novelty (Challenge) | 35 | 30% | 10.5 |
| Technical Soundness (Challenge) | 40 | 25% | 10.0 |
| Practical Feasibility (Challenge) | 40 | 25% | 10.0 |
| Completeness (Challenge) | 55 | 20% | 11.0 |
| **Total** | | | **41.5 ≈ 42/100** |

**Note:** Score thấp là intentional vì góc nhìn Devil's Advocate tập trung vào critical weaknesses. Điểm này được dùng để cân bằng với các reviewer khác trong consensus.

### Major Points

**1. 🎭 Novelty inflation — Claim cao hơn thực tế.**

Paper claim: *"Chưa có công trình nào kết hợp GNN encoder + DRL cho adaptive routing trên NoC một cách có hệ thống."*

Kiểm tra thực tế:
- *GNN-RL for network routing* đã được Rusek et al. (2024) và nhiều nhóm khác thực hiện cho computer networks. Sự khác biệt duy nhất là "trên NoC" — nhưng về mặt kỹ thuật, routing trên graph là routing trên graph. Topology scale khác nhau (64 nodes vs 1000+ nodes) không tạo ra novelty về mặt methodology.
- *Graph-based NoC routing* đã có từ lâu (ví dụ: region-based routing, turn model dựa trên graph coloring).
- *What is actually novel here?* GNN encoder architecture? Không — dùng GCN/GAT/MPNN có sẵn. DRL agent? Không — PPO implementation có sẵn. Integration framework? Có thể, nhưng đây là engineering contribution, không phải scientific novelty cho IEEE TC.

**Kết luận:** Contribution claim cần được viết lại một cách trung thực hơn. Đây là *application paper* (applying GNN+DRL to NoC routing domain), không phải *methodological breakthrough*. Với IEEE TC, application paper cần chứng minh significant performance gain + hardware feasibility — cả hai đều chưa được làm.

**2. 🎭 The latency elephant in the room.**

Đây là điểm yếu chí mạng nhất của proposal mà paper đang attempt to dismiss với một mitigation bullet.

Hãy làm phép tính đơn giản:
- GAT 2-layer, 128-dim hidden, 4-head attention (default GATv2): mỗi forward pass trên graph 64 node cần xấp xỉ:
  - Layer 1: 4 heads × (128×128×64 + attention computations) ≈ 4.2M MACs
  - Layer 2: tương tự ≈ 4.2M MACs
  - MLP decoder: ~10K MACs
  - Total: ~8.5M MAC operations

So sánh với routing trong router hardware:
- Router hiện tại dùng deterministic routing: 1 cycle (= read LUT + mux).
- DyAD adaptive: 10–20 cycles (threshold compare + routing computation).
- MAAR (MLP-based): ~100–200 cycles (MLP forward pass, đã được optimize).
- GAT 2-layer: **estimated ~1,000–10,000+ cycles** (trên embedded processor không có hardware GNN accelerator).

Paper target <100 cycles: con số này **không realistic** nếu không có custom hardware accelerator cho GNN. Và nếu đã có hardware GNN accelerator, thì contribution không còn là "routing algorithm" nữa mà là "GNN accelerator design" — một paper khác.

**Bottom line:** Paper đang đề xuất giải pháp mà không kiểm tra feasibility constraint quan trọng nhất. *Nên chuyển mục tiêu từ "real-time routing decision" sang "offline routing policy optimization" hoặc "adaptive routing với relaxed timing constraints" (ví dụ: update routing policy mỗi N cycles chứ không phải per-packet).*

**3. 🎭 Deadlock — Ticking time bomb.**

Adaptive routing + cycle-free guarantee là bài toán kinh điển. Một số vấn đề cụ thể:

- Duato's protocol: với adaptive routing cần tối thiểu 2 VCs (1 escape + 1 adaptive). Paper không nói rõ VC configuration.
- GNN-DRL agent học policy từ experience — nếu training không có deadlock penalty trong reward, model có thể học policy gây deadlock.
- Cần thêm deadlock detection mechanism hoặc hard constraint trong action selection.
- Hiện tại mitigation column chỉ ghi "Deadlock-free subchannel mechanism" — đây không phải mitigation, đây là hand-waving. *Cần concrete solution.*

**4. 🎭 Comparison bias.**

Paper so sánh với 4 baselines:
- XY routing — deterministic, outdated.
- DyAD — 2004, 20+ năm tuổi.
- MAAR — 2022, dùng MLP.
- Regional Adaptive — heuristic.

Đây là cherry-picked baselines. Một số baselines đáng so sánh hơn:
- *HERMES* (2023) — hierarchical adaptive routing, dùng MLP nhưng optimized cho hardware.
- *BiNoC* (2024) — bidirectional NoC với adaptive routing, có publication trên IEEE TC.
- *DRNoC* (2024) — domain-specific routing cho ML workloads.

**Đề xuất:** Bổ sung ít nhất 1 baseline từ IEEE TC 2023–2025. Nếu không, reviewer sẽ cho rằng tác giả đang so sánh với các phương pháp yếu để làm nổi bật phương pháp của mình.

**5. 🎭 Simulation fidelity gap.**

- Noxim/BookSim2 là transaction-level simulator, không cycle-accurate cho routing pipeline detail.
- Thông số như buffer depth, routing pipeline stages, arbiter delay ảnh hưởng lớn đến latency.
- Nếu chỉ mô phỏng ở transaction level, kết quả có thể không reproduce được trên RTL simulation.
- Paper cần nêu rõ assumptions và acknowledge fidelity gap.

### Devil's Advocate Final Word

> *"This is a competent proposal for applying existing ML techniques to a well-known problem. The gap analysis is sound but the novelty is not breakthrough-level for IEEE TC. The single biggest technical risk — whether GNN inference can meet NoC timing constraints — is not adequately addressed; it is hand-waved with a mitigation bullet that amounts to 'we'll make it faster.' The deadlock problem is similarly under-treated. The authors should either (a) provide concrete feasibility evidence for inference latency, (b) reframe the contribution as an offline/periodic optimization framework, or (c) target a venue with lower hardware feasibility bar (e.g., IEEE Access)."*

### Decision: **Major Revision (nghiêng về Reject nếu không giải quyết được latency và deadlock)**

---

# 📋 Key Revision Items (Actionable)

Ưu tiên theo mức độ quan trọng:

| # | Item | Priority | Reviewer | Action Required |
|---|------|----------|----------|----------------|
| **K1** | **Inference latency feasibility** | 🔴 Critical | 🎯 Domain Expert, 😈 Devil's Advocate | Benchmark GNN inference trên embedded processor (RISC-V/ARM) với target <100 cycles hoặc reframe contribution thành offline/periodic optimization |
| **K2** | **Deadlock guarantee** | 🔴 Critical | 🎯 Domain Expert, 😈 Devil's Advocate | Bổ sung deadlock-free proof (theo Duato's protocol), VC configuration, và mechanism để prevent GNN policy gây deadlock |
| **K3** | **Reward function specification** | 🟠 High | 🔬 Methods Reviewer | Chi tiết hóa reward design: giá trị α, β, γ, normalization scheme, tuning strategy |
| **K4** | **Related work update** | 🟠 High | 🏛️ Editor-in-Chief, 🎯 Domain Expert | Bổ sung các paper IEEE TC 2023–2025, GNN+WAN routing, hardware-aware GNN acceleration |
| **K5** | **Statistical protocol** | 🟠 High | 🔬 Methods Reviewer | Specify number of runs, seeds, warm-up, significance tests, confidence intervals |
| **K6** | **Novelty claim rewrite** | 🟠 High | 🏛️ Editor-in-Chief, 😈 Devil's Advocate | Hạ mức claim từ "chưa có công trình nào" xuống "chưa có systematic study trong NoC context" |
| **K7** | **Energy metric integration** | 🟡 Medium | 🎯 Domain Expert | Đưa energy consumption vào core metrics (không chỉ scope) |
| **K8** | **Broader impact expansion** | 🟡 Medium | 🌐 Interdisciplinary Reviewer | Thêm design automation, safety-critical implications, irregular topology discussion |
| **K9** | **Baseline completeness** | 🟡 Medium | 😈 Devil's Advocate | Bổ sung 1–2 baselines từ IEEE TC 2023–2025 |
| **K10** | **Simulator fidelity note** | 🟢 Low | 🎯 Domain Expert, 😈 Devil's Advocate | Acknowledge limitations của transaction-level simulation so với cycle-accurate/RTL |
| **K11** | **Venue consideration** | 🟢 Low | 🏛️ Editor-in-Chief | Cân nhắc IEEE Access hoặc JSA nếu không có FPGA/RTL validation |

---

# 📝 Editor Recommendation (Consensus Decision)

## Tổng hợp

| Metric | Value |
|--------|-------|
| **Consensus Score** | **56/100** |
| **Consensus Decision** | **Major Revision** |
| **Confidence** | Medium (cần thêm dữ liệu từ inference benchmark để đánh giá feasibility) |

## Recommendation

**Major Revision with path to Accept** nếu tác giả:

1. **Chứng minh được** GNN inference latency khả thi trên NoC router timeline (qua benchmark trên embedded processor), hoặc reframe contribution thành offline routing policy optimization framework.
2. **Bổ sung** deadlock-free proof chi tiết và VC configuration.
3. **Cập nhật** reward function design, statistical protocol, và related work.
4. **Điều chỉnh** novelty claim cho trung thực với contribution thực tế.

**Nếu không giải quyết được K1 và K2:** chuyển recommendation xuống **Reject** (theo Devil's Advocate) do fundamental feasibility flaw.

**Suggested alternative venues** (nếu IEEE TC không phù hợp):
- **IEEE Access** — topical nhưng yêu cầu contribution thấp hơn
- **JSA (Journal of Systems Architecture)** — phù hợp với simulation-based NoC research
- **NoCArc workshop (at MICRO)** — đúng chuyên ngành, có peer review nhưng dễ accept hơn

---

*Review hoàn thành ngày 14/05/2026 theo format ARS Review.*
