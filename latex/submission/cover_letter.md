# Cover Letter — Journal of Systems Architecture (Elsevier)

## To: Editor-in-Chief, Journal of Systems Architecture

### Subject: Submission of Original Research Article

**Title:** "GNNocRoute: Topology-Aware Periodic Routing Optimization for Network-on-Chip using Graph Neural Networks"

Dear Editor-in-Chief,

I am writing to submit our manuscript entitled "GNNocRoute: Topology-Aware Periodic Routing Optimization for Network-on-Chip using Graph Neural Networks" for consideration for publication in the *Journal of Systems Architecture*.

As multi-core System-on-Chips scale to unprecedented complexities, the Network-on-Chip (NoC) communication backbone faces critical congestion challenges. While recent literature has explored Deep Reinforcement Learning (DRL) for adaptive routing, these methods predominantly suffer from two severe limitations: the inability of standard Multi-Layer Perceptrons to inherently capture spatial topology, and the sheer impracticality of per-packet neural network inference under strict nanosecond-scale NoC timing budgets (often exceeding 1,000,000 cycles per decision).

In this manuscript, we present GNNocRoute, a framework that directly resolves these physical and architectural bottlenecks. Our key contributions include:

* **Topology-Awareness via GNNs:** We empirically demonstrate that Graph Neural Network (GNN) embeddings strongly and naturally correlate with structural betweenness centrality (Pearson r up to 0.978), allowing the routing agent to proactively identify topological bottlenecks without explicit graph computations.
* **Periodic Policy Optimization:** To bridge the massive gap between inference latency and router clock budgets, we propose a decoupled, periodic update architecture. This guarantees real-time, deadlock-free per-packet routing via Duato's protocol while the heavier GNN-DRL optimization occurs asynchronously.
* **Comprehensive Trade-off Characterization:** Moving beyond conventional high-load-only evaluations, we present a rigorous BookSim2 cycle-accurate simulation across the full operational spectrum (Injection Rates from 0.001 to 0.50). While our approach reduces hotspot latency by up to 27% under heavy loads, we transparently analyze the heuristic degradation (up to 44% latency overhead) under low-load regimes. This critical trade-off highlights the limitations of purely adaptive routing at low loads and establishes a foundational argument for dynamic, load-aware hybrid routing in future many-core architectures.

We believe that our empirical approach—transforming a theoretical machine learning concept into a physically feasible architectural framework, complete with a candid analysis of its operational trade-offs—aligns perfectly with the rigorous, systems-oriented focus of the *Journal of Systems Architecture*.

This manuscript is original, has not been published before, and is not currently being considered for publication elsewhere. All authors have reviewed and approved the manuscript and its submission to your journal.

Thank you very much for your time and for managing the editorial process of our submission. We look forward to your feedback.

Sincerely,

**Le Tien Hieu**
Institute of Information Technology (VNU-ITI)
Vietnam National University, Hanoi, Vietnam
Email: letienhieu@mcs.edu.vn
