# RaBitQ Reference Implementation Analysis

## Overview

This document analyzes the [RaBitQ reference C++ implementation](https://github.com/gaoj0017/RaBitQ) to bridge theory (US-009 through US-014) with practical code. The implementation has four key files:

| File | Purpose | Lines |
|:--|:--|:--|
| `src/ivf_rabitq.h` | Core: IVFRN class, search, scan, fast_scan | ~300 |
| `src/space.h` | Bitwise distance: quantize, transpose, popcount | ~150 |
| `src/fast_scan.h` | SIMD batch: pack_LUT, pack_codes, accumulate | ~200 |
| `src/search.cpp` | Entry point: load, rotate queries, run search | ~100 |

Supporting: `data/rabitq.py` (index phase), `matrix.h` (I/O), `utils.h` (helpers).

## The IVFRN Class

The core class `IVFRN<D, B>` is templated on:

| Template Param | Meaning | Example |
|:--|:--|:--|
| `D` | Original vector dimension | 128 (SIFT), 960 (GIST) |
| `B` | Padded bit count ($= \lceil D/64 \rceil \cdot 64$) | 128, 960 → 960 |

### Key Member Variables

```cpp
template <uint32_t D, uint32_t B>
class IVFRN {
  struct Factor {
    float sqr_x;       // ||o_r - c||²
    float error;        // Error bound coefficient
    float factor_ppc;   // Per-popcount factor
    float factor_ip;    // Per-inner-product factor
  };

  Factor* fac;         // N precomputed factor structs
  uint32_t N, C;       // N vectors, C clusters
  uint32_t* start;     // Cluster start indices (sorted by cluster)
  uint32_t* len;       // Cluster sizes
  uint32_t* id;        // Original vector IDs (reordered by cluster)
  float* dist_to_c;    // ||o_r - c|| per vector
  float* x0;           // ⟨ō,o⟩ per vector
  float* centroid;     // C × B rotated centroids (P⁻¹c)
  float* data;         // N × D original vectors (for re-ranking)
  uint64_t* binary_code; // N × (B/64) packed binary codes
  uint8_t* packed_code;  // FAST_SCAN: reorganized nibble layout
  float* u;            // B random numbers for query quantization
};
```

**Math-to-code mapping:**

| Theory (US-009–014) | Code Variable | Type |
|:--|:--|:--|
| $\|\mathbf{o}_r - \mathbf{c}\|$ | `dist_to_c[i]` | float |
| $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | `x0[i]` | float |
| $\bar{\mathbf{x}}_b$ (D-bit code) | `binary_code[i * B/64 .. ]` | uint64[] |
| $P^{-1}\mathbf{c}_c$ (rotated centroid) | `centroid[c * B .. ]` | float[] |
| $\mathbf{o}_r$ (original vector) | `data[i * D .. ]` | float[] |

### The Factor Struct: Precomputed Distance Components

During `load()`, per-vector factors are precomputed to minimize work during search:

```cpp
static constexpr float fac_norm = const_sqrt(1.0 * B);  // √B = √D
static constexpr float max_x1 = 1.9 / const_sqrt(1.0 * B - 1.0);  // ε₀/√(D-1)

for (int i = 0; i < N; i++) {
    long double x_x0 = (long double)dist_to_c[i] / x0[i];  // ||o_r-c|| / ⟨ō,o⟩
    fac[i].sqr_x = dist_to_c[i] * dist_to_c[i];             // ||o_r-c||²
    fac[i].error = 2 * max_x1 * sqrt(x_x0*x_x0 - dist_to_c[i]*dist_to_c[i]);
    fac[i].factor_ppc = -2/fac_norm * x_x0 * (popcount(binary_code[i]) * 2 - B);
    fac[i].factor_ip  = -2/fac_norm * x_x0;
}
```

**Derivation of `factor_ip`:** From the distance formula (US-009):

$$
\|\mathbf{o}_r - \mathbf{q}_r\|^2 = \|\mathbf{o}_r - \mathbf{c}\|^2 + \|\mathbf{q}_r - \mathbf{c}\|^2 - 2\|\mathbf{o}_r - \mathbf{c}\| \cdot \|\mathbf{q}_r - \mathbf{c}\| \cdot \frac{\langle \bar{\mathbf{o}}, \mathbf{q}\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle}
$$

The factor $-2 \cdot \|\mathbf{o}_r - \mathbf{c}\| / (\langle \bar{\mathbf{o}}, \mathbf{o}\rangle \cdot \sqrt{D})$ becomes `factor_ip` after absorbing the $1/\sqrt{D}$ from the codebook normalization.

**The error bound:** `max_x1 = 1.9 / √(B-1)` implements $\varepsilon_0/\sqrt{D-1}$ with $\varepsilon_0 = 1.9$ (see US-014). The `error` field stores the full error coefficient:

$$
\text{error} = 2 \cdot \frac{\varepsilon_0}{\sqrt{D-1}} \cdot \sqrt{\frac{\|\mathbf{o}_r - \mathbf{c}\|^2}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2} - \|\mathbf{o}_r - \mathbf{c}\|^2}
$$

## The search() Function

```cpp
ResultHeap IVFRN::search(float* query, float* rd_query, uint32_t k,
                         uint32_t nprobe, float distK) const {
    ResultHeap KNNs;

    // 1. Find N_probe nearest clusters (using rotated centroids)
    Result centroid_dist[C];
    for (int i = 0; i < C; i++) {
        centroid_dist[i] = {sqr_dist<B>(rd_query, centroid + i*B), i};
    }
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    // 2. For each probed cluster
    for (int pb = 0; pb < nprobe; pb++) {
        uint32_t c = centroid_dist[pb].second;
        float sqr_y = centroid_dist[pb].first;   // ||q_r - c||² (approx)

        // 3. Quantize query relative to this cluster's centroid
        float vl, vr;
        space.range(rd_query, centroid + c*B, vl, vr);
        float width = (vr - vl) / ((1 << B_QUERY) - 1);
        uint32_t sum_q = 0;
        space.quantize(byte_query, rd_query, centroid + c*B, u, vl, width, sum_q);

        // 4. Scan cluster (SCAN or FAST_SCAN mode)
#if defined(SCAN)
        transpose_bin(byte_query, quant_query);
        scan(KNNs, distK, k, quant_query, binary_code + start[c]*(B/64),
             len[c], fac + start[c], sqr_y, vl, width, sum_q,
             query, data + start[c]*D, id + start[c]);
#elif defined(FAST_SCAN)
        pack_LUT<B>(byte_query, LUT);
        fast_scan(KNNs, distK, k, LUT, packed_code + packed_start[c],
                  len[c], fac + start[c], sqr_y, vl, width, sum_q,
                  query, data + start[c]*D, id + start[c]);
#endif
    }
    return KNNs;
}
```

**Key observations:**
- `rd_query` is the **rotated query** ($P^{-1}\mathbf{q}$), computed once in `search.cpp` before calling `search()`
- `centroid` stores **rotated centroids** ($P^{-1}\mathbf{c}_c$), so centroid distances use rotated space
- Query quantization is redone **per cluster** (different `vl`, `vr` per cluster centroid)
- `distK` starts at $+\infty$ and tightens as better candidates are found

## The scan() and fast_scan() Functions

Both functions implement the same algorithm with different inner product computation:

```cpp
// Core distance estimation (same in both modes):
float tmp_dist = fac->sqr_x + sqr_y
    + fac->factor_ppc * vl
    + (bitwise_result * 2 - sumq) * fac->factor_ip * width;
float error_bound = y * fac->error;    // y = ||q_r - c||
float lower_bound = tmp_dist - error_bound;

// Re-ranking decision:
if (lower_bound < distK) {
    float gt_dist = sqr_dist<D>(query, data);  // Exact distance
    if (gt_dist < distK) {
        KNNs.emplace(gt_dist, *id);
        if (KNNs.size() > k) KNNs.pop();
        if (KNNs.size() == k) distK = KNNs.top().first;
    }
}
```

| Code Expression | Math | Meaning |
|:--|:--|:--|
| `fac->sqr_x` | $\|\mathbf{o}_r - \mathbf{c}\|^2$ | Squared data-centroid distance |
| `sqr_y` | $\|\mathbf{q}_r - \mathbf{c}\|^2$ | Squared query-centroid distance |
| `bitwise_result` | $\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u\rangle$ | Binary-integer inner product |
| `fac->factor_ip * width` | $-2\|\mathbf{o}_r-\mathbf{c}\|/(\langle\bar{\mathbf{o}},\mathbf{o}\rangle\sqrt{D}) \cdot \Delta$ | Scaling for quantized IP |
| `tmp_dist` | $\hat{d}^2$ | Estimated squared distance |
| `error_bound` | Error margin from Eq. 14 | Theoretical guarantee |
| `lower_bound` | $\hat{d}^2 - \text{err}$ | Optimistic distance bound |
| `distK` | Current $k$-th best exact distance | Re-ranking threshold |

**Crucial design:** Re-ranking uses **exact float distance** (`sqr_dist<D>`), not the estimated distance. The estimated distance only decides **whether to compute** the exact distance. This is why the original data vectors are stored (`data[]`).

## SCAN vs FAST_SCAN

| Aspect | SCAN | FAST_SCAN |
|:--|:--|:--|
| **Compile flag** | `#define SCAN` | `#define FAST_SCAN` |
| **Query prep** | `transpose_bin` → bit planes | `pack_LUT` → 16-entry LUTs |
| **Inner product** | `ip_byte_bin` (AND+popcount) | `accumulate` (vpshufb+add) |
| **Data layout** | Sequential binary codes | Packed nibble batches (32) |
| **Extra index storage** | None | `packed_code`, `packed_start` |

## Query Rotation (search.cpp)

The main entry point rotates all queries **once** before searching:

```cpp
Matrix<float> RandQ(Q.n, BB, Q);  // Copy queries, pad to B dimensions
RandQ = mul(RandQ, P);            // Rotate: rd_query = query × P = P⁻¹query
```

`mul()` uses Eigen for matrix multiplication. The rotation matrix $P$ is loaded from disk (generated by `data/rabitq.py`).

## Eigen Library Usage

| Usage | Where | Purpose |
|:--|:--|:--|
| Matrix multiplication | `mul()` in `matrix.h` | Query rotation ($Q \times P$) |
| Memory layout | Column-major (Eigen default) | Aligned for SIMD |
| **Not used for** | Distance computation | Hand-optimized SIMD instead |

Eigen handles the "heavy lifting" of matrix operations during preprocessing, while the inner loop uses hand-tuned SIMD/bitwise operations for maximum throughput.

## Data Flow Summary

```
INDEX PHASE (Python: rabitq.py):
  Raw vectors → KMeans clustering → Per-cluster normalization
  → P⁻¹o → sign extraction → binary codes + x0 + dist_to_c
  → Save: P, centroids, codes, x0, dist_to_c

LOAD PHASE (C++: load()):
  Load all data → Precompute Factor structs
  → FAST_SCAN: pack_codes for nibble layout

QUERY PHASE (C++: search()):
  Rotate query (Eigen) → Find N_probe clusters (sqr_dist)
  → Per cluster: quantize query → SCAN or FAST_SCAN
  → Lower bound test → Exact re-rank if promising
  → Return top-k ResultHeap
```

## Key Implementation Insights

- **Factor precomputation** eliminates per-query division by $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ — everything is absorbed into `factor_ip` and `factor_ppc` during `load()`
- **Error bound** is computed per-vector at load time, not per-query — the only query-dependent part is `y = ||q_r - c||`
- **Randomized query quantization** (`u[]` sampled uniformly) is done at load time and reused across queries — a simplification that still works well in practice
- **32-vector batching** in scan() (even in SCAN mode) improves cache locality — the inner loop processes 32 vectors, computes lower bounds, then re-ranks
- **Data stored twice:** Binary codes for estimation AND original float vectors for re-ranking — memory doubles but exact re-ranking is essential for high recall
- **Cluster-local processing:** Everything (codes, factors, data, IDs) is stored contiguously per cluster, maximizing cache efficiency during sequential scans

## Sources

- [RaBitQ reference — ivf_rabitq.h](https://github.com/gaoj0017/RaBitQ/blob/main/src/ivf_rabitq.h) — Core IVFRN class
- [RaBitQ reference — space.h](https://github.com/gaoj0017/RaBitQ/blob/main/src/space.h) — Bitwise operations
- [RaBitQ reference — fast_scan.h](https://github.com/gaoj0017/RaBitQ/blob/main/src/fast_scan.h) — SIMD FastScan
- [RaBitQ reference — search.cpp](https://github.com/gaoj0017/RaBitQ/blob/main/src/search.cpp) — Search entry point
- [RaBitQ reference — index.cpp](https://github.com/gaoj0017/RaBitQ/blob/main/src/index.cpp) — Index construction
- [RaBitQ reference — rabitq.py](https://github.com/gaoj0017/RaBitQ/blob/main/data/rabitq.py) — Python index phase
- [RaBitQ GitHub repo](https://github.com/gaoj0017/RaBitQ)
- [Eigen 3.4.0](https://eigen.tuxfamily.org/) — Matrix operations library
