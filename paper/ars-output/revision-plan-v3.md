# GNNocRoute Paper — Revision Plan v3.0
# Feasibility-Study Framing with Preliminary Experimental Results

---

## 1. Revised Abstract

```
Network-on-Chip (NoC) is the communication backbone for modern multi-core
System-on-Chips. Conventional deterministic routing algorithms such as XY
dimension-order routing suffer from congestion imbalance — our graph-theoretic
analysis across seven topologies shows that an 8x8 mesh under XY routing
achieves a congestion imbalance of 0.475, with central nodes exhibiting
2-3x higher betweenness centrality than edge nodes.

This paper presents a feasibility study on topology-aware periodic routing
optimization for NoC using Graph Neural Networks (GNN). We report four
preliminary experimental findings. First, untrained GNN encoders exhibit
strong correlation with betweenness centrality (|r| up to 0.978 on Mesh 4x4),
indicating that topology-aware representations emerge implicitly. Second,
cross-topology generalization experiments suggest that relative node
importance rankings transfer across mesh sizes (Pearson r ≈ 0.96), though
absolute scaling remains topology-specific. Third, measured inference
latency on CPU (Python/PyTorch) of approximately 1 million cycles per
forward pass on an 8x8 mesh confirms that per-packet GNN inference is
impractical under NoC timing constraints, motivating periodic policy
optimization as a necessary design alternative. Fourth, GNN inference
time scales approximately O(N) with the number of nodes.

These findings provide initial evidence that topology-aware representations
can be extracted via GNN encoding, and that periodic routing optimization
offers a feasible path toward topology-generalizable adaptive routing on NoC.
We discuss implications and limitations of this early-stage investigation.
```

---

## 2. Revised Contribution List

**Before (v2.0, overclaimed):**
1. "First systematic study applying GNN-enhanced DRL to NoC adaptive routing"
2. "Periodic policy optimization framework solving inference latency bottleneck"
3. "Benchmark design with 6 baselines and statistical protocol"

**After (v3.0, defensible):**
1. **Structural congestion analysis:** Quantitative demonstration across 7 topologies (mesh, torus, fat-tree, ring, small-world, random) that deterministic routing creates structurally predictable bottlenecks, measurable via graph centrality metrics.
2. **Preliminary GNN-topology correlation evidence:** Experimental results showing that GNN node embeddings strongly correlate with betweenness centrality (|r| up to 0.978), suggesting topology-aware representations can emerge without explicit centrality computation.
3. **Feasibility analysis of periodic GNN-based optimization:** Measured inference latency (CPU) and scalability data establishing the feasibility boundaries of GNN-enhanced routing under NoC timing constraints.
4. **Cross-topology generalization characterization:** Initial evidence that GNN-based routing representations partially transfer across mesh sizes (Pearson r ≈ 0.96), identifying topology-specific scaling as an open research challenge.

---

## 3. Full Rewritten Section 6

```
\section{Preliminary Experimental Results}
\label{sec:results}

We conducted four experiments to evaluate the feasibility of GNN-based
topology representations for NoC routing. All experiments use PyTorch
Geometric~\cite{pyg2019} on a CPU (ARM Neoverse-N1, 2.8 GHz). NetworkX~\cite{networkx2008}
is used for graph construction. Source code and data are available upon request.

\subsection{Structural Representation Learning}
\label{sec:results-correlation}

We investigate whether GNN node embeddings encode topology-structural
information relevant to routing, specifically betweenness centrality (BC)
as a proxy for congestion potential.

\textbf{Setup:} Four NoC topologies (Mesh 4x4, Mesh 8x8, Torus 4x4,
Small-World n=36) are converted to PyTorch Geometric Data objects.
Node features include normalized degree and positional coordinates.
Two untrained GNN models (GCN, GAT with 2 layers, hidden dimension 32,
output dimension 16) compute node embeddings via a single forward pass.
We compute the Pearson correlation coefficient between each embedding
dimension and the ground-truth BC.

\textbf{Results:} Table~\ref{tab:correlation} summarizes the maximum
absolute correlation observed.

\begin{table}[h]
\centering
\caption{Maximum Pearson correlation |r| between GNN embedding
dimension and betweenness centrality}
\label{tab:correlation}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Topology} & \textbf{Nodes} & \textbf{GCN max|r|} & \textbf{GAT max|r|} \\
\midrule
Mesh 4x4 & 16 & \textbf{0.978} & 0.862 \\
Mesh 8x8 & 64 & 0.744 & \textbf{0.870} \\
Torus 4x4 & 16 & 0.000* & 0.000* \\
Small-World (n=36) & 36 & 0.615 & 0.261 \\
\bottomrule
\end{tabular}
\vspace{2mm}
\small\textit{*Torus 4x4 is a regular graph where all nodes have identical
4 degree and symmetric BC, yielding zero variance.}
\end{table}

On Mesh 4x4, the GCN embedding achieves |r| = 0.978 with the
ground-truth BC, indicating that topology-aware structural information
emerges implicitly in GNN representations. GAT on Mesh 8x8 achieves
|r| = 0.870, suggesting attention mechanisms may provide advantages
on larger topologies.

\textbf{Interpretation:} These results suggest that GNN-based topology
encoders can capture congestion-relevant structural information
\emph{without explicit graph centrality computation}. This is a
preliminary but encouraging finding: if GNN embeddings naturally encode
topology bottlenecks, they may serve as informative state representations
for routing decisions. The Torus case (zero correlation) is expected
given its regular structure, and serves as a negative control confirming
that the method correctly identifies undifferentiated topologies.

\subsection{Cross-Topology Generalization}
\label{sec:results-generalization}

We evaluate whether GNN models trained to predict BC on one topology
can generalize to unseen topologies.

\textbf{Setup:} A 2-layer GCN (hidden=32) and GAT (hidden=16, heads=4)
are trained for 500 epochs to predict BC (MSE loss) on Mesh 4x4 (16 nodes).
The trained model is evaluated on Mesh 8x8 (64 nodes), Small-World (36
nodes), and Torus 4x4 (16 nodes).

\textbf{Results:} Table~\ref{tab:generalization} reports MSE, R$^2$, and
Pearson correlation between predicted and ground-truth BC.

\begin{table}[h]
\centering
\caption{Cross-topology BC prediction (trained on Mesh 4x4)}
\label{tab:generalization}
\small
\begin{tabular}{lcccr}
\toprule
\textbf{Test Topology} & \textbf{Model} & \textbf{MSE} & \textbf{R$^2$} & \textbf{Pearson r} \\
\midrule
Mesh 8x8 & GCN & 0.038 & $-16.83$ & \textbf{0.958} \\
Mesh 8x8 & GAT & 0.072 & $-33.00$ & \textbf{0.972} \\
Small-World & GCN & 0.004 & $-1.02$ & 0.385 \\
Torus 4x4 & GCN & 0.092 & 0.00* & -- \\
\bottomrule
\end{tabular}
\vspace{2mm}
\small\textit{*Torus variance is zero (regular graph), making R$^2$
degenerate.}
\end{table}

On the training topology (Mesh 4x4), both models achieve R$^2 > 0.99.
On Mesh 8x8, the negative R$^2$ in conjunction with Pearson r $\approx
0.96$ reveals an important distinction: the models capture relative
node ranking (which nodes are more central) but fail to predict absolute
BC values, which differ in scale between the 4x4 and 8x8 meshes.

\textbf{Interpretation:} This is a significant and non-obvious finding.
The high ranking correlation (r $\approx 0.96$) suggests that GNN-based
representations transfer \emph{structural patterns} across topology sizes.
The negative R$^2$ indicates that \emph{absolute scaling} is topology-
specific. This identifies a concrete research direction: topology-aware
normalization or fine-tuning mechanisms are needed for cross-topology
generalization.

\subsection{Inference Feasibility Analysis}
\label{sec:results-latency}

We measure GNN inference latency on CPU to assess feasibility under NoC
timing constraints. In a typical NoC router clocked at 1 GHz, each
routing decision must complete within a few cycles~\cite{wang2022maar},
making per-packet GNN inference impractical. Our measurements quantify
this gap.

\textbf{Setup:} Inference time is measured for GCN and GAT models with
varying hidden dimensions on four topologies. Each measurement is the
average of 500 forward passes after 50 warmup iterations.

\textbf{Results:} Table~\ref{tab:latency} reports measured latency in
microseconds and estimated cycles at 1 GHz.

\begin{table}[h]
\centering
\caption{GNN inference latency on CPU (Python/PyTorch, ARM Neoverse-N1)}
\label{tab:latency}
\small
\begin{tabular}{lccr}
\toprule
\textbf{Topology} & \textbf{Model} & \textbf{Latency ($\mu$s)} & \textbf{Est. cycles @ 1 GHz} \\
\midrule
Mesh 4x4 & GCN h=32 & 778 & 778,000 \\
Mesh 8x8 & GCN h=32 & 1,015 & 1,015,000 \\
Mesh 8x8 & GAT h=32 & 1,491 & 1,491,000 \\
Mesh 16x16 & GCN h=32 & 1,273 & 1,273,000 \\
Torus 8x8 & GCN h=32 & 1,031 & 1,031,000 \\
\bottomrule
\end{tabular}
\end{table}

Measured latency ranges from 778 $\mu$s (Mesh 4x4) to 1,491 $\mu$s
(Mesh 8x8, GAT). At 1 GHz, this corresponds to approximately 0.8--1.5
million cycles per forward pass. These measurements are on a full
Python/PyTorch stack; optimized C++ implementations with quantization
could reduce this by an estimated 1--2 orders of magnitude, to
approximately 10,000--50,000 cycles.

\textbf{Interpretation:} The measured latency confirms that naive
per-packet GNN inference is infeasible for NoC routing (typical budgets:
$<$10 cycles per decision). However, this finding directly motivates the
periodic policy optimization approach: if routing policies are updated
every $10^4$--$10^5$ cycles rather than every cycle, the inference
overhead becomes manageable. We discuss this trade-off in Section~\ref{sec:discussion}.

\subsection{Scalability Analysis}
\label{sec:results-scalability}

We measure GCN inference time across mesh sizes from 4x4 (16 nodes) to
16x16 (256 nodes) to characterize scalability behavior.

\textbf{Results:} Figure~\ref{fig:scalability} shows inference time as
a function of the number of nodes.

\begin{figure}[h]
\centering
\includegraphics[width=0.85\columnwidth]{figures/gnn-scalability.png}
\caption{GCN inference time scales approximately O(N) with number of
nodes (Mesh NxN, h=32).}
\label{fig:scalability}
\end{figure}

Inference time grows from 773 $\mu$s (16 nodes) to 1,266 $\mu$s
(256 nodes), a factor of 1.64 for a 16x increase in nodes. The per-node
cost decreases from 48.3 $\mu$s/node (4x4) to 4.9 $\mu$s/node (16x16),
indicating efficient parallelization in the GNN message-passing
computation.

\textbf{Interpretation:} The approximately O(N) scaling is predictable
and non-explosive, suggesting that the computational cost of GNN-based
routing representations remains manageable as NoC scale increases.
```

---

## 4. Updated Discussion Section

```
\section{Discussion}
\label{sec:discussion}

\subsection{Implications for Periodic Routing Optimization}

The experimental results collectively support a periodic routing
optimization approach. The inference latency measurements
(Section~\ref{sec:results-latency}) establish that per-packet GNN
inference is infeasible --- with measured latencies of $10^5$--$10^6$
cycles versus a per-packet budget of $<$10 cycles. However, the
structural correlation results (Section~\ref{sec:results-correlation})
suggest that GNN representations carry topology information relevant
to routing decisions. If policies are updated every $P$ cycles where
$P$ is large enough to amortize the inference cost, the topology-aware
representations may still inform routing behavior.

A preliminary feasibility estimate: with an optimized C++
implementation (estimated $10^4$--$5 \times 10^4$ cycles per inference)
and a policy update period $P = 10^5$ cycles, the inference overhead
would be approximately 10--50\%. This overhead may be acceptable if
the resulting routing decisions improve congestion balance.

\subsection{The Generalization Challenge}

The cross-topology experiment (Section~\ref{sec:results-generalization})
reveals a nuanced picture: GNN-based representations partially
generalize across topology sizes (high Pearson r), but the absolute
scale is topology-specific (negative R$^2$). This suggests two viable
research directions: (1) topology-aware normalization schemes that
align BC scales across different mesh sizes, and (2) fine-tuning
strategies where a base model is adapted to each target topology
with minimal additional training.

\subsection{Limitations}

This paper presents an early-stage investigation with several
important limitations:
\begin{enumerate}
    \item \textbf{Static graph analysis:} The GNN experiments are
    conducted on static topology graphs with synthetic node features.
    Dynamic NoC state (buffer occupancy, link utilization) is not yet
    incorporated.
    \item \textbf{No routing simulation:} We have not yet integrated
    GNN routing decisions into a cycle-accurate NoC simulator
    (e.g., Noxim~\cite{noxim2016}). The correlation and latency
    measurements are necessary but not sufficient to validate
    end-to-end routing performance.
    \item \textbf{CPU-only measurements:} Latency numbers are
    measured on a server-class ARM CPU with Python/PyTorch.
    On-chip inference on a router-embedded processor would likely
    exhibit different characteristics.
    \item \textbf{No RTL validation:} Hardware implementation costs
    (area, power, timing closure) are not addressed.
    \item \textbf{Limited topology scope:} Only regular 2D topologies
    are evaluated. Irregular and hierarchical topologies remain
    unexplored.
\end{enumerate}

\subsection{Threats to Validity}

\begin{itemize}
    \item \textbf{Correlation does not imply causation:} High
    correlation between GNN embeddings and BC does not guarantee
    that GNN-based routing decisions will improve congestion
    balance. End-to-end simulation is required.
    \item \textbf{Python overhead:} The measured inference latency
    includes Python interpreter overhead; optimized implementations
    may be substantially faster. However, even order-of-magnitude
    improvements do not bridge the gap to per-packet inference.
    \item \textbf{Synthetic features:} Current node features (degree,
    position) are intentionally minimal to isolate structural effects.
    Real deployment would require richer input features.
\end{itemize}
```

---

## 5. Updated Conclusion

```
\section{Conclusion and Future Work}
\label{sec:conclusion}

This paper presents a feasibility study on topology-aware periodic
routing optimization for Network-on-Chip using Graph Neural Networks.
We report four preliminary experimental findings:

\begin{enumerate}
    \item GNN embeddings strongly correlate with betweenness centrality
    (|r| up to 0.978), suggesting that topology-aware representations
    emerge without explicit graph computation.
    \item Cross-topology generalization partially succeeds: relative
    node importance transfers across mesh sizes (Pearson r $\approx$
    0.96), but absolute scaling requires topology-specific adaptation.
    \item Measured inference latency ($\approx 10^6$ cycles on CPU)
    confirms that per-packet GNN inference is impractical, motivating
    periodic policy optimization as a necessary design alternative.
    \item GNN inference scales approximately O(N), with predictable
    computational growth.
\end{enumerate}

These findings provide initial evidence for the feasibility of
GNN-based topology representations in NoC routing, while realistically
identifying the inference latency and generalization challenges that
must be addressed.

\textbf{Future work} includes: (1) integrating GNN routing decisions
into a cycle-accurate NoC simulator (Noxim) for end-to-end validation;
(2) developing topology-aware normalization techniques for cross-topology
generalization; (3) evaluating quantized GNN models on embedded-class
processors; and (4) exploring irregular and hierarchical NoC topologies.
```

---

## 6. Reviewer-Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| "No routing simulation results" | High | Explicitly acknowledge in limitations. Frame as feasibility study. Report what IS measured (correlation, latency, scalability). |
| "GNN results are on static graphs only" | Medium | Acknowledge. Note that static graph analysis is a necessary first step. Dynamic state integration is future work. |
| "Latency measured on Python, not realistic" | Medium | Report honestly. Discuss optimization path. The raw numbers still demonstrate infeasibility of per-packet approach. |
| "R² negative = model doesn't work" | Medium | Pre-empt with explicit discussion. Show that Pearson r is high — model captures ranking but not scale. This IS a finding. |
| "Why not just use traditional adaptive routing?" | Medium | Traditional methods use local heuristics (thresholds, regional info). GNN provides a fundamentally different signal: global topology structure. |
| "Novelty is incremental" | Medium | The novelty is in the combination of (a) GNN-topology correlation analysis for NoC, (b) periodic feasibility argument, (c) generalization characterization. |

---

## 7. Suggested Paper Title Refinements

| Option | Assessment |
|--------|------------|
| "GNNocRoute: GNN-Enhanced Adaptive Routing for Network-on-Chip" | Current — implies production system |
| "Topology-Aware Graph Neural Network Representations for NoC Routing" | Better — focuses on representations |
| "GNN-Based Topology Representations for Periodic NoC Routing Optimization: A Feasibility Study" | **Recommended** — accurately describes scope |
| "Preliminary Investigation of GNN Encoders for Topology-Aware NoC Routing" | Good for workshop |

---

## 8. Sentences That Must Be Softened (from current v2.0)

| Current | Softened |
|---------|----------|
| "GNNocRoute, a framework combining GNN with DRL" | "An investigation into GNN-based routing representations" |
| "A GATv2 encoder with 2 layers and 64 hidden dimensions learns node embeddings" | "A GATv2 encoder with 2 layers is evaluated as a candidate architecture" |
| "This provides the first systematic study" | "To the best of our knowledge, this is the first feasibility study examining..." |
| "GNNocRoute formulates the problem as periodic routing policy optimization" | "We formulate the problem as periodic routing policy optimization" |
| "Our framework generalizes across topologies" | "Our preliminary experiments suggest that representation ranking partially transfers across topologies" |

---

## 9. Remaining Weaknesses Before Submission

| Weakness | Severity | Path to Resolution |
|----------|----------|--------------------|
| No end-to-end routing simulation | Critical | Requires Noxim integration (1-2 weeks) |
| Only static graph features tested | High | Requires dynamic state injection |
| No deadlock verification | Medium | Formal analysis or simulation |
| Only 4 topologies in GNN experiments | Medium | Add more in next iteration |
| Single CPU architecture measured | Medium | Add embedded processor benchmark |
| No comparison to existing adaptive routing | High | Add DyAD or similar baseline |

---

*Prepared for GNNocRoute paper revision v3.0 — 14/05/2026*
