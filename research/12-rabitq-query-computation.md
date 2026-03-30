# RaBitQ Query Quantization and Efficient Computation

## Prerequisites Refresher

### Uniform Scalar Quantization

**Uniform quantization** maps a continuous range $[v_l, v_r]$ into $2^{B_q}$ discrete levels with equal spacing $\Delta$:

$$
\Delta = \frac{v_r - v_l}{2^{B_q}}
$$

| Symbol | Meaning |
|:--|:--|
| $v_l, v_r$ | Lower and upper bounds of the quantization range |
| $B_q$ | Bit-width (number of bits per quantized value) |
| $\Delta$ | Step size (width of each quantization bin) |

A value $z \in [v_l, v_r]$ maps to an unsigned integer $q_u = \lfloor (z - v_l)/\Delta \rfloor \in \{0, 1, \ldots, 2^{B_q} - 1\}$.

### Bitwise Operations

| Operation | Notation | Description | Example (8-bit) |
|:--|:--|:--|:--|
| **AND** | $a \wedge b$ | Bit $i$ is 1 iff both $a_i = 1$ and $b_i = 1$ | `11010010 & 10110011 = 10010010` |
| **POPCOUNT** | $\text{popcount}(x)$ | Count the number of 1-bits | `popcount(10010010) = 3` |

The inner product of two binary vectors $\langle \mathbf{a}, \mathbf{b}\rangle = \sum_i a_i b_i = \text{popcount}(\mathbf{a} \wedge \mathbf{b})$, since $a_i b_i = 1$ iff both bits are 1.

### Bit Planes

An array of $B_q$-bit integers can be decomposed into $B_q$ **bit planes**: the $j$-th bit plane $\bar{\mathbf{q}}_u^{(j)}$ contains the $j$-th bit of every element. This transforms a byte-level array into $B_q$ binary vectors.

**Example** ($B_q = 2$, values $[3, 1, 2, 0]$ in binary $[11, 01, 10, 00]$):
- Bit plane 0 (LSB): $(1, 1, 0, 0)$
- Bit plane 1 (MSB): $(1, 0, 1, 0)$

## Main Content

### Step 1: Query Transformation

At query time, the query vector $\mathbf{q}$ undergoes the same inverse rotation as data vectors (US-010):

$$
\mathbf{q}' = P^{-1}\mathbf{q} = P^T\mathbf{q}
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{q}$ | Normalized unit query vector |
| $P^{-1} = P^T$ | Inverse rotation (same matrix used at index time) |
| $\mathbf{q}'$ | Transformed query in codebook coordinate space |

This is a single $O(D^2)$ matrix-vector multiply, done **once per query** and shared across all candidate data vectors.

### Step 2: Uniform Scalar Quantization with Randomized Rounding

Each coordinate of $\mathbf{q}'$ is quantized to a $B_q$-bit unsigned integer:

$$
\bar{q}_u[i] = \left\lfloor \frac{q'[i] - v_l}{\Delta} + u_i \right\rfloor
$$

| Symbol | Meaning |
|:--|:--|
| $v_l = \min_i q'[i]$ | Minimum coordinate of $\mathbf{q}'$ |
| $v_r = \max_i q'[i]$ | Maximum coordinate of $\mathbf{q}'$ |
| $\Delta = (v_r - v_l) / 2^{B_q}$ | Quantization step size |
| $u_i \sim \text{Uniform}(0, 1)$ | Random dither per coordinate |
| $\bar{q}_u[i] \in \{0, 1, \ldots, 2^{B_q} - 1\}$ | Quantized unsigned integer |

**Why randomized rounding?** Deterministic rounding (floor or round) introduces systematic bias — coordinates always round in the same direction. Adding uniform noise $u_i$ makes the rounding **unbiased**: $\mathbb{E}[\bar{q}_u[i]] = (q'[i] - v_l)/\Delta$. This preserves the overall unbiasedness of the RaBitQ estimator from US-011.

### Theorem 3.3: How Many Bits Suffice?

**Theorem 3.3:** $B_q = \Theta(\log\log D)$ bits per query coordinate suffice to make the additional quantization error negligible compared to the $O(1/\sqrt{D})$ bound from data quantization.

| $D$ | $\log\log D$ | Practical $B_q$ |
|:--|:--|:--|
| 128 | $\log_2 7 \approx 2.8$ | 4 |
| 768 | $\log_2 9.6 \approx 3.3$ | 4 |
| 4096 | $\log_2 12 \approx 3.6$ | 4 |

**In practice, $B_q = 4$ is always sufficient.** The query quantization error is exponentially smaller than the data quantization error, so there is little reason to use more bits.

### Step 3: The Efficient Computation Formula

The goal is to compute $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle = \langle P\bar{\mathbf{x}}, \mathbf{q}\rangle = \langle \bar{\mathbf{x}}, \mathbf{q}'\rangle$. Substituting the quantization formulas:

$$
\bar{\mathbf{x}} = \frac{2\bar{\mathbf{x}}_b - \mathbf{1}_D}{\sqrt{D}}, \qquad q'[i] \approx v_l + \Delta \cdot \bar{q}_u[i]
$$

Expanding the inner product:

$$
\langle \bar{\mathbf{x}}, \bar{\mathbf{q}}\rangle = \frac{2\Delta}{\sqrt{D}} \langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u\rangle + \frac{2v_l}{\sqrt{D}} \sum_{i=1}^{D} \bar{\mathbf{x}}_b[i] - \frac{\Delta}{\sqrt{D}} \sum_{i=1}^{D} \bar{q}_u[i] - \sqrt{D} \cdot v_l
$$

| Term | Meaning | Computation |
|:--|:--|:--|
| $\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u\rangle$ | Binary-integer inner product | **Bitwise ops** (see below) |
| $\sum \bar{\mathbf{x}}_b[i]$ | Popcount of data code | Precomputed at index time |
| $\sum \bar{q}_u[i]$ | Sum of quantized query | Precomputed once per query |
| $v_l, \Delta$ | Query quantization parameters | Computed once per query |

**Key insight:** The only term that varies per data vector is $\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u\rangle$. All other terms are either per-query constants or precomputed per-vector scalars. The entire distance computation reduces to one binary-integer inner product plus a few multiplications and additions.

### Step 4: Bitwise Decomposition

The binary-integer inner product $\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u\rangle$ decomposes via bit planes:

$$
\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u\rangle = \sum_{j=0}^{B_q - 1} 2^j \cdot \langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u^{(j)}\rangle
$$

| Symbol | Meaning |
|:--|:--|
| $\bar{\mathbf{q}}_u^{(j)} \in \{0, 1\}^D$ | The $j$-th bit plane of the quantized query |
| $\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u^{(j)}\rangle$ | Inner product of two binary vectors |
| $2^j$ | Weight of the $j$-th bit plane |

Each binary inner product is computed as **AND + popcount**:

$$
\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u^{(j)}\rangle = \text{popcount}(\bar{\mathbf{x}}_b \wedge \bar{\mathbf{q}}_u^{(j)})
$$

**Total operations per distance:** $B_q$ AND operations + $B_q$ popcounts over $D$-bit strings, plus scalar arithmetic. For $B_q = 4$ and $D = 128$: just $4 \times 2 = 8$ uint64 ANDs + popcounts + a few FP operations.

### Reference Implementation

The [RaBitQ reference code](https://github.com/gaoj0017/RaBitQ/blob/main/src/space.h) implements this pipeline:

**Range computation** (once per query per cluster):
```cpp
void Space::range(float* q, float* c, float& vl, float& vr) {
    vl = +1e20; vr = -1e20;
    for (int i = 0; i < B; i++) {
        float tmp = (*q) - (*c);  // q'[i] = (P⁻¹q)[i] - centroid
        if (tmp < vl) vl = tmp;
        if (tmp > vr) vr = tmp;
        q++; c++;
    }
}
```

**Query quantization** with randomized rounding:
```cpp
void Space::quantize(uint8_t* result, float* q, float* c,
                     float* u, float vl, float width, uint32_t& sum_q) {
    float one_over_width = 1.0 / width;  // 1/Δ
    uint32_t sum = 0;
    for (int i = 0; i < B; i++) {
        (*result) = (uint8_t)(((*q) - (*c) - vl) * one_over_width + (*u));
        sum += (*result);   // Accumulate Σq̄_u[i]
        q++; c++; result++; u++;
    }
    sum_q = sum;
}
```

**Bit plane extraction** (using AVX2 for byte-to-bitplane transpose):
```cpp
void Space::transpose_bin(uint8_t* q, uint64_t* tq) {
    for (int i = 0; i < B; i += 32) {
        __m256i v = _mm256_load_si256((__m256i*)q);
        v = _mm256_slli_epi32(v, 8 - B_QUERY);
        for (int j = 0; j < B_QUERY; j++) {
            uint32_t v1 = _mm256_movemask_epi8(v);  // Extract MSBs
            // ... pack into bit plane array
            v = _mm256_slli_epi32(v, 1);  // Shift to next bit
        }
        q += 32;
    }
}
```

**Weighted AND + popcount** across bit planes:
```cpp
uint32_t Space::ip_byte_bin(uint64_t* q, uint64_t* d) {
    uint64_t ret = 0;
    for (int i = 0; i < B_QUERY; i++) {
        ret += (ip_bin_bin(q, d) << i);  // 2^i weighting
        q += (B / 64);  // Next bit plane
    }
    return ret;
}
```

## Worked Example

### Computing $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle$ for $D = 8$, $B_q = 2$

**Data code** (from US-010): $\bar{\mathbf{x}}_b = (1, 0, 1, 1, 0, 1, 0, 1)$.

**Transformed query:** $\mathbf{q}' = P^{-1}\mathbf{q} = (0.3, -0.1, 0.5, 0.2, -0.4, 0.1, -0.2, 0.4)$.

**Step 1: Range.** $v_l = -0.4$, $v_r = 0.5$, $\Delta = (0.5 - (-0.4))/2^2 = 0.225$.

**Step 2: Quantize** (using $u_i = 0.5$ for simplicity):

| $i$ | $q'[i]$ | $(q'[i] - v_l)/\Delta$ | $+ u_i$ | $\lfloor\cdot\rfloor$ = $\bar{q}_u[i]$ |
|:--|:--|:--|:--|:--|
| 1 | $0.3$ | $3.11$ | $3.61$ | $3$ |
| 2 | $-0.1$ | $1.33$ | $1.83$ | $1$ |
| 3 | $0.5$ | $4.00$ | $4.50$ | $3^*$ |
| 4 | $0.2$ | $2.67$ | $3.17$ | $3$ |
| 5 | $-0.4$ | $0.00$ | $0.50$ | $0$ |
| 6 | $0.1$ | $2.22$ | $2.72$ | $2$ |
| 7 | $-0.2$ | $0.89$ | $1.39$ | $1$ |
| 8 | $0.4$ | $3.56$ | $4.06$ | $3^*$ |

($^*$clamped to $2^{B_q} - 1 = 3$)

$\bar{\mathbf{q}}_u = (3, 1, 3, 3, 0, 2, 1, 3)$. In binary: $(11, 01, 11, 11, 00, 10, 01, 11)$.

**Step 3: Extract bit planes:**
- Bit plane 0 (LSB): $\bar{\mathbf{q}}_u^{(0)} = (1, 1, 1, 1, 0, 0, 1, 1)$
- Bit plane 1 (MSB): $\bar{\mathbf{q}}_u^{(1)} = (1, 0, 1, 1, 0, 1, 0, 1)$

**Step 4: AND + popcount:**

$$
\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u^{(0)}\rangle = \text{popcount}((10110101) \wedge (11110011)) = \text{popcount}(10110001) = 4
$$

$$
\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u^{(1)}\rangle = \text{popcount}((10110101) \wedge (10110101)) = \text{popcount}(10110101) = 5
$$

$$
\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u\rangle = 2^0 \cdot 4 + 2^1 \cdot 5 = 4 + 10 = 14
$$

**Step 5: Precomputed scalars:**
- $\sum \bar{\mathbf{x}}_b[i] = \text{popcount}(10110101) = 5$
- $\sum \bar{q}_u[i] = 3 + 1 + 3 + 3 + 0 + 2 + 1 + 3 = 16$

**Step 6: Full formula:**

$$
\langle \bar{\mathbf{x}}, \bar{\mathbf{q}}\rangle = \frac{2 \cdot 0.225}{\sqrt{8}} \cdot 14 + \frac{2(-0.4)}{\sqrt{8}} \cdot 5 - \frac{0.225}{\sqrt{8}} \cdot 16 - \sqrt{8} \cdot (-0.4)
$$

$$
= \frac{0.45}{2.828} \cdot 14 + \frac{-0.8}{2.828} \cdot 5 - \frac{0.225}{2.828} \cdot 16 + 1.131
$$

$$
= 2.228 - 1.414 - 1.273 + 1.131 = 0.672
$$

**Verification (exact floating-point):** $\langle \bar{\mathbf{x}}, \mathbf{q}'\rangle = \frac{1}{\sqrt{8}}[(+1)(0.3) + (-1)(-0.1) + (+1)(0.5) + (+1)(0.2) + (-1)(-0.4) + (+1)(0.1) + (-1)(-0.2) + (+1)(0.4)] = \frac{2.2}{\sqrt{8}} = 0.778$.

The quantization introduces a small error ($0.672$ vs $0.778$), which is absorbed into the overall $O(1/\sqrt{D})$ error bound.

## Comparison with TurboQuant/PQ

| Aspect | RaBitQ | PQ |
|:--|:--|:--|
| **Query preprocessing** | $P^{-1}\mathbf{q}$ ($O(D^2)$) + quantize to $B_q$-bit ($O(D)$) | Build LUT per sub-codebook ($O(D \cdot k)$) |
| **Per-vector computation** | $B_q$ ANDs + popcounts ($O(B_q \cdot D/64)$) | $M$ LUT lookups ($O(M)$) |
| **Bits per query coord** | $B_q = 4$ (fixed, theory-backed) | Full float (no query quantization) |
| **Batch optimization** | FastScan with SIMD (US-013) | PQ4fs with SIMD shuffle |
| **Error from query quant** | Negligible (Theorem 3.3) | N/A (no query quantization) |

## Key Takeaways

- **Query transformation** $\mathbf{q}' = P^{-1}\mathbf{q}$ is done once per query; uniform quantization to $B_q = 4$ bits adds negligible error
- **Randomized rounding** preserves unbiasedness — deterministic rounding would introduce systematic bias
- **Bitwise decomposition** converts the binary-integer inner product into $B_q$ AND + popcount operations on $D$-bit strings
- For $D = 128$, $B_q = 4$: each distance needs just **8 uint64 ANDs + 8 popcounts** plus scalar arithmetic
- The per-vector computation is dominated by $\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u\rangle$; all other terms are precomputed constants
- Theorem 3.3 guarantees $B_q = \Theta(\log\log D)$ suffices — making the approach scale-invariant in practice

## Sources

- [RaBitQ paper — arXiv:2405.12497](https://arxiv.org/abs/2405.12497) — Section 3.3, Theorem 3.3
- [RaBitQ paper — arXiv HTML](https://arxiv.org/html/2405.12497) — Equations 17–22
- [RaBitQ reference implementation — space.h](https://github.com/gaoj0017/RaBitQ/blob/main/src/space.h) — Bitwise distance computation
- [RaBitQ GitHub repo](https://github.com/gaoj0017/RaBitQ)
- [RaBitQ Library](https://vectordb-ntu.github.io/RaBitQ-Library/)
- [Popcount — Wikipedia](https://en.wikipedia.org/wiki/Hamming_weight) — Hardware popcount instruction
