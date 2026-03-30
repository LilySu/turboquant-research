# RaBitQ SIMD FastScan Implementation

## Prerequisites Refresher

### SIMD (Single Instruction, Multiple Data)

**SIMD** processes multiple data elements in parallel using wide registers. AVX2 provides **256-bit registers** that can hold:
- 32 × 8-bit integers, or 16 × 16-bit integers, or 8 × 32-bit integers

| Term | Meaning |
|:--|:--|
| AVX2 | Advanced Vector Extensions 2 — Intel/AMD SIMD instruction set |
| `__m256i` | C intrinsic type for a 256-bit integer register |
| `_mm256_shuffle_epi8` | The `vpshufb` instruction — parallel byte lookup |
| `_mm256_add_epi16` | Parallel 16-bit integer addition across 16 lanes |

### The `vpshufb` (Shuffle Bytes) Instruction

`vpshufb` takes two 256-bit registers: a **table** and an **index** register. For each byte in the index register, it uses the lower 4 bits as a lookup index into the corresponding 16-byte half of the table register:

$$
\text{result}[i] = \text{table}[\text{index}[i] \wedge \text{0xF}]
$$

This performs **32 parallel 4-bit LUT lookups** in a single instruction — the key enabler of FastScan.

### Look-Up Tables (LUTs)

A **LUT** precomputes all possible outputs of a function, then replaces runtime computation with a table lookup. For a 4-bit input, the LUT has $2^4 = 16$ entries — small enough to fit in half of an AVX2 register (16 bytes = 128 bits).

## Main Content

### Why Batch Processing?

The bitwise approach from US-012 processes **one data vector at a time**: $B_q$ AND + popcount operations per distance. This leaves SIMD registers underutilized — a 256-bit register holds the entire 128-bit or 256-bit code with room to spare.

**FastScan** instead processes **32 data vectors simultaneously** by:
1. Precomputing LUTs from the query vector
2. Reorganizing 32 data codes for SIMD-friendly access
3. Using `vpshufb` for parallel LUT lookup across all 32 vectors

This approach was pioneered by André et al. (2015, 2017) for Product Quantization (**PQx4fs**) and adapted by RaBitQ to work with binary quantization codes.

### Step 1: LUT Construction from Query

Split the $D$-dimensional quantized query $\bar{\mathbf{q}}_u$ into $M = D/4$ **sub-segments** of 4 coordinates each. For each sub-segment $m$, build a 16-entry LUT containing all possible inner products with 4-bit binary patterns:

$$
\text{LUT}_m[j] = \sum_{k=0}^{3} j_k \cdot \bar{q}_u[4m + k], \quad j \in \{0, 1, \ldots, 15\}
$$

| Symbol | Meaning |
|:--|:--|
| $M = D/4$ | Number of sub-segments |
| $j$ | 4-bit index ($0$ to $15$), representing a binary pattern $j_3 j_2 j_1 j_0$ |
| $j_k$ | The $k$-th bit of $j$ |
| $\text{LUT}_m[j]$ | Partial inner product for sub-segment $m$ with binary pattern $j$ |

**Each LUT has 16 entries.** For 8-bit entries, one LUT = 16 bytes = 128 bits = half of an AVX2 register. Two LUTs fit in one 256-bit register.

**Efficient LUT construction** (from the [reference code](https://github.com/gaoj0017/RaBitQ/blob/main/src/fast_scan.h)):

```cpp
template <uint32_t B>
inline void pack_LUT(uint8_t* byte_query, uint8_t* LUT) {
    constexpr uint32_t M = B / 4;
    for (int i = 0; i < M; i++) {
        LUT[0] = 0;  // Pattern 0000 → inner product 0
        for (int j = 1; j < 16; j++) {
            // Build entry j from entry j-lowbit(j) plus one query value
            LUT[j] = LUT[j - lowbit(j)] + byte_query[pos[j]];
        }
        LUT += 16;
        byte_query += 4;
    }
}
```

The `lowbit(j)` trick incrementally builds each LUT entry from previously computed entries, avoiding redundant additions.

### Step 2: Packing 32 Codes into a Batch

To enable SIMD lookup, the 32 data codes must be reorganized. Each code is a $D$-bit string, split into $M = D/4$ nibbles (4-bit groups). The layout reorganization interleaves nibbles across codes so that:

- For sub-segment $m$: all 32 nibbles (one from each code) are packed contiguously
- Each nibble is a 4-bit LUT index stored in the low/high half of a byte

**Layout:** 32 nibbles pack into 16 bytes (two nibbles per byte), fitting in one half of an AVX2 register.

```
Before: Code 0: [n₀ n₁ ... n_{M-1}], Code 1: [n₀ n₁ ...], ..., Code 31: [...]
After:  Sub-seg 0: [c₀ c₁ ... c₃₁], Sub-seg 1: [c₀ c₁ ... c₃₁], ...
```

| Layout Aspect | Detail |
|:--|:--|
| Batch size | 32 codes (fits naturally in AVX2 registers) |
| Nibbles per code | $M = D/4$ |
| Bytes per sub-segment per batch | 16 (32 nibbles, 2 per byte) |
| Total packed size per batch | $16 \cdot M / 2$ bytes |

The reference code uses a permutation array `perm0[16] = {0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15}` to interleave the low and high 16-byte halves of the batch for AVX2's 128-bit lane structure.

### Step 3: SIMD Lookup with `vpshufb`

For each sub-segment $m$, a single `vpshufb` instruction performs 32 parallel LUT lookups:

```cpp
// Load LUT (two sub-segment LUTs packed into one 256-bit register)
__m256i lut = _mm256_load_si256((__m256i const*)LUT);

// Load packed data nibbles (32 codes, low and high halves)
__m256i lo = /* low nibbles of 32 codes for sub-segment m */;
__m256i hi = /* high nibbles of 32 codes for sub-segment m */;

// Parallel lookup: 32 simultaneous table lookups per instruction
__m256i res_lo = _mm256_shuffle_epi8(lut, lo);
__m256i res_hi = _mm256_shuffle_epi8(lut, hi);
```

**What happens in hardware:** Each of the 32 bytes in the `lo`/`hi` register is a 4-bit LUT index. The `vpshufb` instruction uses this index to select the corresponding byte from the `lut` register. This replaces 32 sequential table lookups with **2 SIMD instructions**.

### Step 4: Accumulation Across Sub-segments

The partial inner products from all $M$ sub-segments are accumulated:

```cpp
// Accumulate 8-bit results into 16-bit accumulators to avoid overflow
accu[0] = _mm256_add_epi16(accu[0], res_lo);
accu[1] = _mm256_add_epi16(accu[1], _mm256_srli_epi16(res_lo, 8));
accu[2] = _mm256_add_epi16(accu[2], res_hi);
accu[3] = _mm256_add_epi16(accu[3], _mm256_srli_epi16(res_hi, 8));
```

| Accumulator | Contents |
|:--|:--|
| `accu[0]` | Even-indexed results from low half (codes 0, 2, 4, ...) |
| `accu[1]` | Odd-indexed results from low half (codes 1, 3, 5, ...) |
| `accu[2]` | Even-indexed results from high half (codes 16, 18, ...) |
| `accu[3]` | Odd-indexed results from high half (codes 17, 19, ...) |

Results are promoted from 8-bit to 16-bit to prevent overflow during accumulation across the $M$ sub-segments. The final step combines even/odd results and permutes across 128-bit lanes to produce 32 distance values.

### Computational Summary

**Bitwise (SCAN) — single vector:**

$$
\text{Cost per distance} = B_q \cdot \lceil D/64 \rceil \text{ AND+popcount operations}
$$

**FastScan — batch of 32 vectors:**

$$
\text{Cost per distance} = \frac{M}{2} \text{ shuffle instructions} + \frac{M}{2} \text{ additions} \approx M \text{ SIMD ops}
$$

| Metric | Bitwise (SCAN) | FastScan (FAST\_SCAN) |
|:--|:--|:--|
| **Vectors per iteration** | 1 | 32 |
| **Key instruction** | AND + popcount | `vpshufb` + add |
| **Operations per distance** ($D=128, B_q=4$) | $4 \times 2 = 8$ uint64 ops | $\sim 32/32 = 1$ SIMD op (amortized) |
| **Data layout** | Sequential bit strings | Interleaved nibbles |
| **Throughput advantage** | Baseline | $\geq 3\times$ faster for large batches |
| **When to use** | Small candidate sets, random access | Large clusters, sequential scan |

### ASCII Diagram: SIMD Register Layout

```
AVX2 256-bit Register (hosts 2 LUTs):
┌──────────────────────────────────────────────────────────────────┐
│  LUT for sub-seg m (low 128 bits)  │  LUT for sub-seg m+1 (high)│
│ [e₀ e₁ e₂ ... e₁₅]                │ [e₀ e₁ e₂ ... e₁₅]        │
│  16 entries × 8-bit = 128 bits     │  16 entries × 8-bit         │
└──────────────────────────────────────────────────────────────────┘

Data Register (32 packed nibbles):
┌──────────────────────────────────────────────────────────────────┐
│ Code₀ Code₈ Code₁ Code₉ ... (low)  │ Code₁₆ Code₂₄ ... (high) │
│ Each byte: [hi_nibble | lo_nibble]  │ Two codes packed per byte  │
└──────────────────────────────────────────────────────────────────┘

vpshufb(LUT_reg, Data_reg):
┌──────────────────────────────────────────────────────────────────┐
│ LUT[nibble₀] LUT[nibble₈] ...      │ LUT[nibble₁₆] ...         │
│ 32 partial inner products in one instruction                     │
└──────────────────────────────────────────────────────────────────┘
```

## Worked Example

### FastScan for $D = 8$, $M = 2$ Sub-segments, Batch of 4 Codes

(Scaled down from 32 to 4 codes for clarity.)

**Query sub-segment values:** $\bar{q}_u[0..3] = (3, 1, 0, 2)$, $\bar{q}_u[4..7] = (1, 3, 2, 0)$.

**LUT for sub-segment 0** ($M_0$):

| Pattern $j$ | Binary | $\text{LUT}_0[j] = \sum j_k \cdot \bar{q}_u[k]$ |
|:--|:--|:--|
| 0 | 0000 | $0$ |
| 1 | 0001 | $3$ |
| 2 | 0010 | $1$ |
| 3 | 0011 | $4$ |
| ... | ... | ... |
| 15 | 1111 | $3 + 1 + 0 + 2 = 6$ |

**4 data codes** (first 4 bits):

| Code | Bits [0:3] | Nibble | LUT lookup |
|:--|:--|:--|:--|
| 0 | $1010$ | $10$ | $\text{LUT}_0[10] = 1 + 2 = 3$ |
| 1 | $1101$ | $13$ | $\text{LUT}_0[13] = 3 + 0 + 2 = 5$ |
| 2 | $0110$ | $6$ | $\text{LUT}_0[6] = 1 + 0 = 1$ |
| 3 | $1111$ | $15$ | $\text{LUT}_0[15] = 6$ |

With SIMD, all 4 lookups happen **simultaneously** via `vpshufb`. Repeat for sub-segment 1 and add results. Total inner products computed with 2 SIMD lookups instead of 4 × 2 = 8 sequential AND+popcount operations.

## Comparison with TurboQuant/PQ

| Aspect | RaBitQ FastScan | PQx4fs (Faiss) |
|:--|:--|:--|
| **Code type** | Binary (1 bit/dim) | 4-bit sub-codebook index |
| **LUT entries** | 16 per 4-bit sub-segment | 16 per sub-codebook |
| **LUT content** | Binary-integer partial inner products | Squared distance to centroid |
| **Batch size** | 32 codes | 32 codes |
| **SIMD instruction** | Same (`vpshufb`) | Same (`vpshufb`) |
| **Key difference** | Needs scalar correction (÷ $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$) | Direct distance estimate |
| **Theoretical guarantees** | $O(1/\sqrt{D})$ error bound | No error bound |

RaBitQ's FastScan reuses PQx4fs infrastructure almost identically — the only change is the LUT content (binary partial products vs. centroid distances) and the post-processing (unbiased estimator correction).

## Key Takeaways

- **FastScan** processes 32 data vectors in parallel using AVX2 SIMD instructions, giving $\geq 3\times$ speedup over bitwise single-vector processing
- The $D$-bit code is split into $M = D/4$ **nibbles** (4-bit sub-segments); each nibble indexes a 16-entry **LUT**
- **`vpshufb`** performs 32 parallel 4-bit LUT lookups in a single instruction — the core of FastScan
- Two LUTs fit in one 256-bit register (16 entries × 8 bits × 2 = 256 bits)
- Data codes must be **reorganized** (interleaved nibble layout) for SIMD access — a one-time cost at index time
- RaBitQ's FastScan directly adapts **PQx4fs** (André et al. 2015, 2017) infrastructure, replacing centroid distances with binary inner products
- **SCAN mode** (bitwise) is better for small candidate sets; **FAST\_SCAN mode** (SIMD) wins for large sequential scans

## Sources

- [RaBitQ paper — arXiv:2405.12497](https://arxiv.org/abs/2405.12497) — Sections 2, 3.4, 4
- [RaBitQ paper — arXiv HTML](https://arxiv.org/html/2405.12497) — PQx4fs discussion
- [RaBitQ reference implementation — fast_scan.h](https://github.com/gaoj0017/RaBitQ/blob/main/src/fast_scan.h) — AVX2 FastScan code
- [RaBitQ reference implementation — space.h](https://github.com/gaoj0017/RaBitQ/blob/main/src/space.h) — Bitwise SCAN code
- [RaBitQ GitHub repo](https://github.com/gaoj0017/RaBitQ)
- [André et al. (2015) — Cache locality is not enough: High-Performance Nearest Neighbor Search with Product Quantization Fast Scan](https://dl.acm.org/doi/10.14778/2856318.2856324) — Original PQx4fs
- [André et al. (2017) — Accelerated Nearest Neighbor Search with Quick ADC](https://dl.acm.org/doi/10.1145/3078971.3078992) — Quick ADC with vpshufb
- [Intel Intrinsics Guide — _mm256_shuffle_epi8](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) — vpshufb specification
