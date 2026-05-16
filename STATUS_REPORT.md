# STATUS REPORT — GNNocRoute-DRL Paper
## Target: Journal of Systems Architecture (Elsevier, Q1)

**Date:** 2026-05-16
**Author:** Lê Tiến Hiếu (NCS 25218003, Viện CNTT - ĐHQGHN)

---

## 1. Tình trạng từng Section

| Section | File | Status | Pages | Ghi chú |
|---------|------|--------|-------|---------|
| Abstract | `main.tex` | ✅ OK | — | Đã fix inconsistencies |
| 1. Introduction | `01-introduction.tex` | ✅ OK | 1–2 | Đã fix timing description |
| 2. Related Work | `02-related-work.tex` | ✅ OK | 2–3 | Đã fix order-of-magnitude claim |
| 3. Proposed Method | `03-method.tex` | ✅ OK | 3–4 | Đã clarify two-sublevel timing |
| 4. Experimental Setup | `04-experiments.tex` | ✅ OK | 4 | Đầy đủ params |
| 5. Results | `05-results.tex` | ✅ OK | 4–6 | Figures + tables OK |
| 6. Discussion | `06-discussion.tex` | ✅ OK | 6–7 | Đã fix Mesh 16×16 reference |
| 7. Conclusion | `07-conclusion.tex` | ✅ OK | 7–8 | Enumerate → fixed |
| References | `references.bib` | ✅ OK | 8–9 | 36 entries, đã bổ sung 3 thiếu |
| **Tổng** | | **✅ OK** | **9 pages** | |

---

## 2. Lỗi đã phát hiện và sửa

### 2.1 Lỗi nghiêm trọng đã sửa

| # | Mô tả | File | Fix |
|---|-------|------|-----|
| 1 | `\ocite{*}` → `\nocite{*}` | `main.tex` | Typo: chữ "o" thay vì "n" |
| 2 | `\usepackage{cite}` xung đột với `natbib` | `main.tex` | Xoá `\usepackage{cite}` (elsarticle.cls tự load natbib) |
| 3 | Abstract: "378 dense configurations" — không khớp với Results (252) | `main.tex` | Sửa thành "252 configurations across 7 injection rates" |
| 4 | "three-order-of-magnitude" — sai số: 10^6/10 = 5 orders | `01-introduction.tex`, `02-related-work.tex` | Sửa thành "approximately five-order-of-magnitude" |
| 5 | GNN inference timing mâu thuẫn: P=5,000 cycles vs 10^6 cycles | `03-method.tex` | Rewrite: tách DRL policy update (mỗi P cycles) khỏi GNN re-encoding (mỗi 10^5–10^6 cycles) |
| 6 | Introduction timing description không khớp với method | `01-introduction.tex` | Cập nhật để nhất quán |
| 7 | Mesh 16×16 — không có trong Experimental Setup | `06-discussion.tex` | Sửa thành "larger topologies (e.g., Mesh 16×16)" |
| 8 | Thiếu 3 references: `amd2023`, `apple2023`, `duato1993` | `references.bib` | Đã thêm đầy đủ |

### 2.2 BibTeX Warnings (17, không ảnh hưởng)

- Thiếu trường `pages` trong 17 entries (inproceedings và article không có pages) — không gây lỗi, cosmetic

### 2.3 Overfull hboxes (nhẹ, formatting)

- Table `\resizebox{\columnwidth}{!}` gây overfull hbox ~25pt ở table simparams — chấp nhận được với 2-column format
- Một số equations hơi rộng — không nghiêm trọng

---

## 3. PDF Compilation

| Hạng mục | Trạng thái |
|----------|-----------|
| LaTeX engine | pdfTeX 3.14159265 (TeX Live 2020) |
| Document class | `elsarticle` v3.5 (2026/01/09) — JSA format |
| Compile lần 1 | ✅ Thành công (9 pages, ~784 KB) |
| Cross-references | ✅ Tất cả resolved (0 undefined) |
| Bibliography | ✅ 36 entries, đúng format elsarticle-num |
| Figures | ✅ 5 figures included (fig3-architecture, fig-latency-all-topos, fig-improvement-heatmap, fig4-congestion-imbalance, fig-saturation) |
| Errors | ❌ 0 (zero) — sạch |
| Output file | `latex/main.pdf` → sao chép vào `submission/GNNocRoute_JSA_v4.pdf` |

---

## 4. Pre-Submission Checklist (Elsevier JSA Guidelines)

### 4.1 Format Requirements

| Mục | Yêu cầu | Status | Ghi chú |
|-----|---------|--------|---------|
| Document class | `elsarticle.cls` | ✅ | v3.5, option [5p,twocolumn] |
| Max pages | Không giới hạn cứng, thường 8–12 | ✅ | 9 pages |
| Abstract | ≤ 250 words | ✅ | ~220 words |
| Keywords | 5–7 từ khóa | ✅ | 5 keywords |
| References format | elsarticle-num (numbered) | ✅ | `.bst` file included |
| Figures | 300 DPI, PNG/PDF/EPS | ✅ | PNG files |
| Equation formatting | AMS math | ✅ | amsmath, amssymb |
| Sections | IMRaD structure | ✅ | Intro → Related → Method → Experiments → Results → Discussion → Conclusion |

### 4.2 Content Checklist

| Mục | Trạng thái | Ghi chú |
|-----|-----------|---------|
| Problem clearly stated | ✅ | Congestion imbalance in deterministic routing |
| Novel contribution claimed | ✅ | GNN + DRL with periodic optimization |
| Method reproducibility | ✅ | BookSim2, parameters in Table |
| Results match claims | ✅ | Numbers consistent across sections |
| Limitations acknowledged | ✅ | Section 6.4 — synthetic traffic, low-load degradation, power models |
| Future work identified | ✅ | Section 7 (Future Work) |
| References comprehensive | ✅ | 36 entries, covers adaptive routing, DRL, GNN |
| No overclaiming | ✅ | Claims conservative: "up to 27.3%" with context |

### 4.3 Elsevier Submission Requirements

| Mục | Trạng thái | Ghi chú |
|-----|-----------|---------|
| Cover letter | ✅ Có sẵn | `submission/cover_letter.md` — cần update title |
| Author info | ✅ | Single author: Le Tien Hieu, VNU-ITI |
| ORCID | Nên có | ORCID của Thầy Hiếu: 0009-0000-6896-0292 |
| Highlights | ❌ Chưa có | Elsevier yêu cầu 3–5 bullet points (85 chars max) |
| Graphical abstract | ❌ Chưa có | Optional nhưng khuyến khích |
| Data availability | ❌ Chưa có statement | Cần thêm "Data availability" section |
| CRediT authorship | ✅ | Single author, n/a |
| Declaration of competing interest | ❌ Chưa có | Cần thêm |
| Funding statement | ✅ Có | Trong Acknowledgments |
| Manuscript number | N/A | Sẽ có sau khi submit |

### 4.4 Figures Quality Check

| Figure | File | In paper? | Size | Notes |
|--------|------|-----------|------|-------|
| fig3-architecture.png | ✅ | ✅ Fig 1 | 55 KB | Architecture diagram |
| fig-latency-all-topos.png | ✅ | ✅ Fig 2 | 155 KB | Latency vs injection rate |
| fig-improvement-heatmap.png | ✅ | ✅ Fig 3 | 111 KB | Improvement heatmap |
| fig4-congestion-imbalance.png | ✅ | ✅ Fig 4 | 113 KB | GNN vs BC correlation |
| fig-saturation.png | ✅ | ✅ Fig 5 | 57 KB | Saturation throughput |
| fig1-latency-comparison.png | ✅ | ❌ Unreferenced | 194 KB | Có thể dùng để thay thế |
| fig2-improvement-heatmap.png | ✅ | ❌ Unreferenced | 71 KB | Duplicate of fig-improvement-heatmap? |
| fig2-ppo-improvement.png | ✅ | ❌ Unreferenced | 83 KB | Not used |
| fig3-latency-vs-injection.png | ✅ | ❌ Unreferenced | 130 KB | Not used |
| fig5-architecture.png | ✅ | ❌ Unreferenced | 55 KB | Duplicate of fig3? |
| fig5-ppo-results.png | ✅ | ❌ Unreferenced | 165 KB | Not used |
| fig5-training-curve.png | ✅ | ❌ Unreferenced | 82 KB | Not used |
| fig6-training-convergence.png | ✅ | ❌ Unreferenced | 74 KB | Not used |

**Khuyến nghị:** Clean up figures folder — giữ lại 5 figures đang dùng, xoá hoặc archive 8 file không dùng đến.

---

## 5. Các bước còn lại trước khi submit

### Critical (cần làm ngay)

1. **Clean up figures**: Xoá 8 file PNG không dùng đến trong `latex/figures/`
2. **Highlights**: Tạo file `highlights.md` (3–5 bullet, ≤85 chars mỗi bullet)
3. **Data availability statement**: Thêm vào conclusion hoặc cuối paper
4. **Declaration of competing interest**: Thêm

### Important (nên làm)

5. **Update cover letter title**: Cover letter viết "GNNocRoute" nhưng paper title là "GNNocRoute-DRL"
6. **Check ORCID**: Thêm vào author block
7. **Graphical abstract**: Optional, nên chuẩn bị
8. **Double-check references**: Verify các citation có context phù hợp (doi nếu có)

### Nice-to-have

9. **Fix BibTeX empty pages warnings**: Thêm pages field cho các proceedings entries
10. **Add DOIs** to references (Elsevier format thường yêu cầu)
11. **Check for reprint permission**: Nếu dùng figure từ nguồn khác

---

## 6. Summary

**LaTeX compilation:** ✅ Clean — 0 errors, 9 pages, all cross-references resolved

**Content:** Paper mô tả GNNocRoute-DRL framework — GNN encoder (GATv2) + Periodic Policy Optimization cho NoC adaptive routing. Kết quả chính: GNN-BC correlation |r|=0.978, adaptive routing giảm 27.3% latency hotspot trên Mesh 4×4. Kết quả thực nghiệm từ 252 BookSim2 configurations (3 topologies × 4 algorithms × 3 traffic patterns × 7 rates × 5 seeds).

**Pre-submission readiness:** ~75%. Critical issues (highlights, data availability, competing interests) cần xử lý trước khi submit.

**Số lỗi đã sửa trong phiên này:** 8 lỗi (2 critical LaTeX, 4 nội dung, 2 references)
