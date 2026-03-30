# RaBitQ with IVF Index Integration

## Prerequisites Refresher

### Inverted File (IVF) Index

An **Inverted File** index partitions the dataset into clusters (buckets) for sub-linear search. Instead of scanning all $N$ vectors, only vectors in nearby clusters are examined.

| Term | Meaning |
|:--|:--|
| **Cluster** | A group of similar vectors, represented by a centroid |
| **Centroid** | The mean of all vectors in a cluster |
| **KMeans** | Algorithm to partition $N$ vectors into $C$ clusters by minimizing total within-cluster variance |
| **Bucket** | The list of vectors assigned to a cluster |

### Re-Ranking

**Re-ranking** computes the **exact** distance for a subset of candidates after an initial approximate screening. It is the most expensive step — minimizing the number of re-ranked candidates directly improves query throughput.

## Main Content

### IVF Structure: Coarse Partitioning

RaBitQ uses a standard IVF structure as its outer layer:

1. **KMeans clustering:** Partition $N$ data vectors into $C$ clusters (typically $C = \sqrt{N}$ to $4\sqrt{N}$)
2. **Per-cluster storage:** Each cluster stores its centroid, rotation matrix $P$, and quantized codes

| Parameter | Meaning | Typical Value |
|:--|:--|:--|
| $N$ | Total data vectors | $10^6$ to $10^9$ |
| $C$ | Number of clusters | $1024$ to $65536$ |
| $N/C$ | Average vectors per cluster | $\sim 1000$ |
| $N_{\text{probe}}$ | Clusters searched per query | $1$ to $128$ |

### Per-Cluster Normalization

Each cluster uses its **own centroid** for normalization (not a global centroid). For data vector $\mathbf{o}_r$ assigned to cluster $c$ with centroid $\mathbf{c}_c$:

$$
\mathbf{o} = \frac{\mathbf{o}_r - \mathbf{c}_c}{\|\mathbf{o}_r - \mathbf{c}_c\|}
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{c}_c$ | Centroid of cluster $c$ |
| $\|\mathbf{o}_r - \mathbf{c}_c\|$ | Distance from data to its cluster centroid (stored) |

**Why per-cluster?** Vectors within a cluster are close together, so centering on the cluster centroid produces small residuals. This makes the normalized vectors more evenly distributed on the hypersphere, improving quantization quality. The full distance formula (from US-009) applies within each cluster:

$$
\|\mathbf{o}_r - \mathbf{q}_r\|^2 = \|\mathbf{o}_r - \mathbf{c}_c\|^2 + \|\mathbf{q}_r - \mathbf{c}_c\|^2 - 2\|\mathbf{o}_r - \mathbf{c}_c\| \cdot \|\mathbf{q}_r - \mathbf{c}_c\| \cdot \langle \mathbf{o}, \mathbf{q}\rangle
$$

### Index Phase

For each cluster $c$:

1. Compute centroid $\mathbf{c}_c$ via KMeans
2. Sample random orthogonal matrix $P_c$ (one per cluster, or shared globally)
3. For each vector $\mathbf{o}_r$ in cluster $c$:
   - Normalize: $\mathbf{o} = (\mathbf{o}_r - \mathbf{c}_c)/\|\mathbf{o}_r - \mathbf{c}_c\|$
   - Inverse-transform: $\mathbf{v} = P_c^{-1}\mathbf{o}$
   - Extract binary code: $\bar{\mathbf{x}}_b[i] = \mathbf{1}[v_i > 0]$
   - Compute quality: $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle = \|\mathbf{v}\|_1/\sqrt{D}$
   - Store: $\bar{\mathbf{x}}_b$, $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$, $\|\mathbf{o}_r - \mathbf{c}_c\|$
4. For FAST_SCAN mode: pack codes into 32-vector batches (US-013)

**Index time** (GIST, 1M vectors, 960-dim, 32 threads): RaBitQ 117s, PQ 105s, OPQ 291s — comparable to PQ.

### Query Phase: The Complete Search Algorithm

```
SEARCH(query q_r, N_probe, k, ε₀):

1. PROBE SELECTION:
   - Compute distance from q_r to all C cluster centroids
   - Select N_probe nearest clusters

2. FOR EACH probed cluster c:
   a. QUERY NORMALIZATION:
      q = (q_r - c_c) / ||q_r - c_c||
      q' = P_c⁻¹ q                        [O(D²), once per cluster]
      q̄_u = Quantize(q', B_q=4)           [O(D)]

   b. DISTANCE ESTIMATION (SCAN or FAST_SCAN):
      For each data vector o in cluster c:
        - Compute ⟨x̄, q̄⟩ via bitwise or SIMD  [O(B_q·D/64)]
        - Estimate: est_ip = ⟨x̄, q̄⟩ / ⟨ō,o⟩
        - Compute error bound:
          err = √((1-⟨ō,o⟩²)/⟨ō,o⟩²) · ε₀/√(D-1)
        - Compute estimated distance:
          d̂ = ||o_r-c_c||² + ||q_r-c_c||² - 2·||o_r-c_c||·||q_r-c_c||·est_ip
        - Compute distance lower bound:
          d_lb = ... using (est_ip + err) ...

   c. CANDIDATE FILTERING:
      If d_lb < current k-th distance:
        Add to candidate heap

3. RE-RANKING:
   For each candidate in heap (ordered by estimated distance):
     - Compute exact distance d_exact
     - Update top-k heap with d_exact
     - If d_lb of remaining candidates > current k-th exact distance:
       STOP (all remaining candidates are provably too far)

4. RETURN top-k by exact distance
```

### Error-Bound-Based Re-Ranking (RaBitQ's Key Innovation)

The confidence interval from US-011 (Eq. 16) gives a **per-vector lower bound** on the true inner product:

$$
\langle \mathbf{o}, \mathbf{q}\rangle \geq \frac{\langle \bar{\mathbf{o}}, \mathbf{q}\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle} - \sqrt{\frac{1 - \langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2}} \cdot \frac{\varepsilon_0}{\sqrt{D-1}}
$$

with failure probability $\leq 2e^{-c_0 \varepsilon_0^2}$.

This translates to an **upper bound on the true distance**, enabling the search to **provably skip** candidates whose best-case distance still exceeds the current top-$k$ threshold.

| Re-ranking aspect | RaBitQ | PQ |
|:--|:--|:--|
| **Decision basis** | Theoretical confidence interval | Empirical threshold (top-$N_{\text{cand}}$) |
| **Guarantee** | Provable: miss probability $\leq 2e^{-c_0 \varepsilon_0^2}$ | No guarantee |
| **Adaptivity** | Per-vector (uses stored $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$) | Global (same threshold for all) |
| **Parameter** | $\varepsilon_0$ (confidence level) | $N_{\text{cand}}$ (re-rank count) |

**$\varepsilon_0 = 1.9$ in practice** — this provides near-perfect confidence across tested datasets, validated empirically in Section 5.2.4 of the paper.

### Why RaBitQ Is More Robust Than PQ

PQ's estimator is biased (US-011), meaning it systematically underestimates inner products. On certain datasets (e.g., MSong with 420 dimensions), this bias compounds:

> "PQ cannot achieve $\geq 60\%$ recall even with re-ranking applied." — RaBitQ paper, Section 5.2.3

RaBitQ's unbiased estimator with provable error bounds works consistently across **all tested datasets**, because the $O(1/\sqrt{D})$ bound holds regardless of data distribution.

## Worked Example

### IVF-RaBitQ Search with Re-ranking

**Setup:** $N = 10^6$, $D = 128$, $C = 1024$, $N_{\text{probe}} = 8$, $k = 10$, $\varepsilon_0 = 1.9$.

**Step 1: Probe selection.** Find 8 nearest cluster centroids to $\mathbf{q}_r$. Each cluster has $\sim 1000$ vectors.

**Step 2: Distance estimation.** For each of $8 \times 1000 = 8000$ candidates:
- Compute estimated inner product via bitwise ops: $\sim 8$ uint64 operations
- Compute error bound: $\text{err} = 0.75 \cdot 1.9/\sqrt{127} \approx 0.126$

**Step 3: Candidate filtering.** Suppose the 10th-best estimated distance is $d_{10} = 15.2$. Candidates with distance lower bound $> 15.2$ are immediately discarded. With error bound of $\pm 0.126$ on the inner product, this typically eliminates $\sim 90\%$ of candidates.

**Step 4: Re-ranking.** The remaining $\sim 800$ candidates get exact distance computation. Exact distances update the top-10 heap. Early stopping: once remaining candidates' lower bounds exceed the updated $d_{10}$, stop.

**Result:** Scanned 8000 codes via fast bitwise ops, re-ranked only $\sim 800$ with exact distances, returned top-10 with high confidence.

## Comparison: IVF-PQ vs IVF-RaBitQ

| Aspect | IVF-PQ (PQx4fs) | IVF-RaBitQ |
|:--|:--|:--|
| **Code length** | $2D$ bits typical (e.g., 256 bits for $D=128$) | $D$ bits ($128$ bits for $D=128$) |
| **Estimator** | Biased | Unbiased |
| **Error bound** | None | $O(1/\sqrt{D})$ with probability |
| **Re-ranking** | Fixed top-$N_{\text{cand}}$ | Adaptive per-vector bounds |
| **Memory per vector** | $2D/8$ bytes | $D/8 + 8$ bytes (code + 2 floats) |
| **Recall stability** | Fails on some datasets | Stable across all tested |
| **Single-vector speed** | LUT lookup | Bitwise (3× faster) |
| **Batch speed** | SIMD FastScan | SIMD FastScan (comparable) |
| **Index time** | Comparable | Comparable |

### Performance Summary (from paper)

| Dataset | $D$ | $N$ | RaBitQ Recall | PQ Recall | Notes |
|:--|:--|:--|:--|:--|:--|
| SIFT1M | 128 | 1M | High | High | Both work well |
| GIST | 960 | 1M | High | Moderate | RaBitQ benefits from high $D$ |
| DEEP | 96 | 10M | High | High | Both work well |
| MSong | 420 | 1M | High | $< 60\%$ | PQ catastrophic failure |

RaBitQ's advantage grows on datasets where PQ's bias causes problems — particularly higher-dimensional spaces and distributions where PQ's sub-codebook assumption breaks down.

## Key Takeaways

- **IVF + RaBitQ** combines coarse clustering (KMeans) with fine-grained binary quantization
- **Per-cluster normalization** uses each cluster's centroid, improving quantization quality for local residuals
- **Error-bound-based re-ranking** is RaBitQ's key innovation — the confidence interval (Eq. 16) enables **provable**, **per-vector**, **adaptive** candidate filtering
- $\varepsilon_0 = 1.9$ provides near-perfect confidence in practice
- RaBitQ uses **half the code length** of PQ while achieving equal or better recall
- RaBitQ is **robust across all tested datasets**, including ones where PQ fails catastrophically ($< 60\%$ recall on MSong)
- The search algorithm naturally supports both **SCAN** (bitwise, small clusters) and **FAST\_SCAN** (SIMD, large clusters) modes

## Sources

- [RaBitQ paper — arXiv:2405.12497](https://arxiv.org/abs/2405.12497) — Sections 4, 5
- [RaBitQ paper — arXiv HTML](https://arxiv.org/html/2405.12497) — Algorithm description, experimental results
- [RaBitQ reference implementation — ivf_rabitq.h](https://github.com/gaoj0017/RaBitQ/blob/main/src/ivf_rabitq.h) — IVF + RaBitQ search
- [RaBitQ reference implementation — index.cpp](https://github.com/gaoj0017/RaBitQ/blob/main/src/index.cpp) — Index construction
- [RaBitQ reference implementation — search.cpp](https://github.com/gaoj0017/RaBitQ/blob/main/src/search.cpp) — Search entry point
- [RaBitQ GitHub repo](https://github.com/gaoj0017/RaBitQ)
- [RaBitQ Library](https://vectordb-ntu.github.io/RaBitQ-Library/)
