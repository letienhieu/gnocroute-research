Dear Editor-in-Chief,
Journal of Systems Architecture (JSA)

Subject: Submission of Original Research Article – "GNNocRoute-DRL: Topology-Aware Adaptive Routing for Network-on-Chip using Graph Neural Networks and Deep Reinforcement Learning"

Dear Editor-in-Chief,

I am pleased to submit our manuscript entitled "GNNocRoute-DRL: Topology-Aware Adaptive Routing for Network-on-Chip using Graph Neural Networks and Deep Reinforcement Learning" for consideration for publication in the Journal of Systems Architecture.

While recent literature has increasingly explored Machine Learning for Network-on-Chip (NoC) routing, the community faces a critical deployment wall: the massive computational overhead of per-packet neural network inference completely violates the nanosecond-scale latency and stringent power constraints of physical NoC routers.

In this manuscript, we present a framework that breaks this barrier. GNNocRoute-DRL uniquely decouples complex offline topology encoding from real-time online routing. By projecting Graph Neural Network (GNN) embeddings into precomputed directional port scores, we condense the intelligence of a multi-layer GATv2 architecture into a lightweight 4 KB weight table.

We believe this work is highly aligned with JSA's focus on practical architectural innovations, driven by three major empirical breakthroughs detailed in our manuscript:

1. 1-Cycle Hardware Feasibility: Unlike state-of-the-art MLP/GNN routers (e.g., MAAR, DeepNR, GARN) that require hundreds or thousands of cycles per hop, GNNocRoute executes routing decisions in a single-cycle table lookup, consuming merely 0.13% of the router's power budget.

2. Unprecedented Fault Tolerance: Because our GNN inherently learns global structural dependencies, a fault-aware variant of GNNocRoute sustains only a 2.5% latency degradation under a severe 15% random link failure rate, directly circumventing the catastrophic +257% latency surge experienced by traditional Planar Adaptive routing.

3. Zero-Shot Scalability: We demonstrate that our topology-aware encodings are structurally invariant, achieving 99.78% routing accuracy when transferred zero-shot from 4x4 to 8x8 mesh scales, outperforming XY routing by up to 25% under hotspot traffic.

This manuscript represents original work, has not been published previously, and is not under consideration by any other journal. All authors have critically reviewed and approved the final manuscript.

Thank you very much for your time and for managing the editorial process of our submission. We look forward to the opportunity of receiving feedback from your esteemed reviewers.

Sincerely,

Le Tien Hieu
Institute of Information Technology (VNU-ITI)
Vietnam National University, Hanoi, Vietnam
Email: letienhieu@mcs.edu.vn
