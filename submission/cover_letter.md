# Cover Letter — Journal of Systems Architecture (Elsevier)

**Date:** May 17, 2026

**Manuscript Title:** GNNocRoute-DRL: Topology-Aware Adaptive Routing for Network-on-Chip using Graph Neural Networks and Deep Reinforcement Learning

**Corresponding Author:** Le Tien Hieu, Institute of Information Technology, Vietnam National University, Hanoi

Dear Editor,

We are pleased to submit our manuscript entitled "GNNocRoute-DRL: Topology-Aware Adaptive Routing for Network-on-Chip using Graph Neural Networks and Deep Reinforcement Learning" for consideration in the Journal of Systems Architecture.

## Why This Work Matters

Network-on-Chip (NoC) routing is a fundamental problem in computer architecture. While Graph Neural Networks (GNNs) have shown promise for topology-aware routing in wide-area networks, their application to on-chip networks has been limited by the five-orders-of-magnitude gap between GNN inference latency and NoC timing constraints. Our work bridges this gap through a novel port-scoring decoupling architecture.

## Three Key Contributions

### 1. Single-Cycle GNN-Aware Routing
We precompute GNN-derived port scores into weight tables (4 KB for 4×4 mesh), enabling topology-aware routing decisions in a single cycle with only 0.13% of the router power budget. This makes GNN-based NoC routing practical for the first time.

### 2. Demonstrated Fault Tolerance
Under 15% random link failures, our fault-aware variant degrades latency by only 2.5% with zero throughput loss. In contrast, Planar Adaptive routing—a classic adaptive algorithm—suffers 257% latency degradation and 37.1% throughput loss under identical conditions.

### 3. Zero-Shot Generalization
Our model trained exclusively on 4×4 meshes achieves 99.78% minimal-port accuracy when transferred to 8×8 meshes without retraining, outperforming XY routing by up to 25% under hotspot traffic.

## Comparison with State of the Art

- **1-cycle inference** vs. N cycles (MAAR, DeepNR) or 10³ cycles (GARN)
- **r=0.978 topology correlation**—the first GNN-based routing validated against graph theory
- **Fault tolerance** absent in all prior DRL-based NoC routing approaches

## Novelty for JSA

This work targets JSA's systems architecture focus by addressing the practical hardware implications of AI-based routing: area, power, timing, and fault tolerance. The port-scoring architecture is a system-level innovation that makes GNN-enhanced routing implementable in real NoC hardware.

We believe our contributions are significant and timely for the on-chip interconnection network community. We look forward to your review.

Sincerely,

Le Tien Hieu, M.Sc.
PhD Candidate, Institute of Information Technology
Vietnam National University, Hanoi
