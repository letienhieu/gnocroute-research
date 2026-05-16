# BÁO CÁO CẢI THIỆN GNN-PPO MODEL CHO PAPER JSA Q1

**Ngày:** 2026-05-16  
**Thực hiện:** Ngọc Anh (subagent)  
**Mục tiêu:** Improve GNN-PPO model để outperfom baselines trên BookSim2

---

## TÓM TẮT PHÁT HIỆN

### ✅ FIX #1: Deadlock Bug — VC Isolation (QUAN TRỌNG NHẤT)
**Vấn đề:** Routing function `gnn_ppo_route_4x4_mesh()` không phân tách VC giữa 
XY và YX path, gây **turn-cycle deadlock** trên transpose traffic.

**Trước khi fix:**
- Transpose @0.30: ~747 cycles (33x tệ hơn XY)
- Uniform @mọi rate: 999/SAT

**Sau khi fix (VC isolation theo pattern của adaptive_xy_yx_mesh):**
- Transpose @0.30: ~22.4 cycles (= XY)
- Uniform @0.10: ~19.7 cycles (= XY)

**Fix:** Tham khảo `adaptive_xy_yx_mesh` trong BookSim2:
```
available_vcs = (vcEnd - vcBegin + 1) / 2
XY → vcBegin..vcBegin+available_vcs-1
YX → vcBegin+available_vcs..vcEnd
```
Áp dụng cho cả 3 header files: 4x4, 8x8, 16x16.

### ✅ FIX #2: Training Pipeline Disconnected
**Vấn đề:** Python training code dùng synthetic environment với 
`np.random.normal()` — không liên kết với BookSim2 thực tế.  
`eval_gnnocrout.py` **ước tính** kết quả GNN (comment: "5% improvement") 
thay vì chạy BookSim2 thật.

**Fix:** Xây dựng pipeline end-to-end:
1. `train_routing_table.py` — GNN training với GATv2 + pairwise decoder
2. `update_and_test_routing_table.py` — Update header + recompile + benchmark

### ✅ FIX #3: Routing Table Generation
**Vấn đề:** File header `.h` chứa routing table static hand-crafted.

**Giải pháp:** GNN training pipeline với:
- **Encoder:** 3-layer GATv2 (7 features → 64 hidden → 32 embedding)
- **Decoder:** Pairwise MLP (src_embed || dst_embed → logit → sigmoid)
- **Node features:** tọa độ (x,y), degree, betweenness centrality, corner/edge/center flags
- **Loss:** Proxy loss (max link utilization + variance + hotspot penalty)

### ⚠️ PHÁT HIỆN QUAN TRỌNG: Giới Hạn Của Binary XY/YX

Trên mesh 4x4, **mọi routing thuần XY, YX, hoặc kết hợp XY/YX đều cho 
kết quả gần như giống hệt nhau** (~19-20 cycles) trên hầu hết traffic patterns.

**Tại sao?** 
- Mesh 4x4 chỉ có 16 nodes → ít lựa chọn routing
- XY và YX đều là shortest-path → hop count bằng nhau
- VC buffer (8 slots) đủ lớn cho congestion ở injection rate thấp-trung bình

**Khác biệt chỉ xuất hiện ở:**
- Transpose @0.40+: XY=258, GNN-YX=24 (YX tốt hơn do tránh diagonal)
- Hotspot @0.10+: tất cả đều saturate

---

## KẾT LUẬN & KHUYẾN NGHỊ

### Static Binary XY/YX Table KHÔNG ĐỦ cho JSA Q1

Phương pháp "GNN generates binary XY/YX selection" có giới hạn cố hữu:
1. **Chỉ 2 action** (XY hoặc YX) — không học được routing phức tạp
2. **Static per (src,dst)** — không adapt theo congestion
3. **Bằng shortest-path** — hop count không đổi

### Đề xuất kiến trúc mới cho Q1 paper

#### Option A: GNN + Non-Minimal Adaptive Routing (Khả thi nhất)
- Cho phép non-minimal routing khi congestion cao
- GNN học khi nào nên detour
- Cần: escape VC chain (như min_adapt) + adaptive threshold

#### Option B: Per-hop GNN Inference
- GNN chạy real-time tại mỗi router hop
- Input: local congestion state (buffer fill, credit count)
- Output: best output port với congestion awareness
- Cần: GNN model đủ nhỏ để inference trong 1 cycle

#### Option C: GNN-Weighted Adaptive Routing
- GNN precompute "routing preference weights" cho mỗi node
- Runtime: combine GNN weights + local congestion → port selection
- Giống min_adapt nhưng với GNN-weighted port priorities

### Kiến trúc khuyến nghị (Option C - thực tế nhất)

```
┌─────────────────┐     ┌─────────────────────┐
│  GNN Encoder    │     │  BookSim2 Runtime    │
│  (offline)      │     │  (per-hop)           │
│                 │     │                      │
│ Topology Graph  │     │ Congestion state     │
│ → Node Embeds   │     │ + GNN weights        │
│ → Weight Matrix │────→│ → Adaptive routing   │
└─────────────────┘     └─────────────────────┘
```

---

## FILE ĐÃ TẠO / SỬA

| File | Mô tả |
|------|-------|
| `booksim2/src/gnn_ppo_route_4x4.h` | Fixed: VC isolation + improved routing table |
| `booksim2/src/gnn_ppo_route_8x8.h` | Fixed: VC isolation |
| `booksim2/src/gnn_ppo_route_16x16.h` | Fixed: VC isolation |
| `paper03-q1-jsa/code/train_routing_table.py` | GNN training pipeline với GATv2 |
| `paper03-q1-jsa/code/update_and_test_routing_table.py` | Pipeline update header + recompile + benchmark |
| `paper03-q1-jsa/experiments/final_benchmark.py` | Comprehensive benchmark script |
| `paper03-q1-jsa/experiments/improvement_log.md` | Chi tiết các fix và kết quả |
| `paper03-q1-jsa/experiments/final_report.md` | Báo cáo tổng kết này |
