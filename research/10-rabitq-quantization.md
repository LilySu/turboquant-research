# RaBitQ Quantization: Finding the Nearest Codebook Vector

## Prerequisites Refresher

### Argmax and Optimization

The **argmax** operator returns the input that maximizes a function:

$$
\bar{\mathbf{x}} = \arg\max_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

| Symbol | Meaning |
|:--|:--|
| $\bar{\mathbf{x}}$ | The vector in $\mathcal{C}$ that maximizes $f$ |
| $\mathcal{C}$ | The search space (codebook) |
| $f(\mathbf{x})$ | The objective function |

### Sign Function

The **sign function** maps real numbers to $\{-1, 0, +1\}$:

$$
\text{sign}(z) = \begin{cases} +1 & z > 0 \\ 0 & z = 0 \\ -1 & z < 0 \end{cases}
$$

**Key property:** For any real number $z$, the product $\text{sign}(z) \cdot z = |z|$ (the absolute value).

### Inner Product and Sign Alignment

Given two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^D$, their inner product is $\langle \mathbf{a}, \mathbf{b}\rangle = \sum_{i} a_i b_i$. Each term $a_i b_i$ is **maximized** when $a_i$ and $b_i$ have the **same sign**. This is the geometric principle behind RaBitQ's quantization.

## Main Content

### The Quantization Problem

From US-009, RaBitQ's codebook is $\mathcal{C} = \{+1/\sqrt{D}, -1/\sqrt{D}\}^D$ with random rotation $P$, forming $\mathcal{C}_{\text{rand}} = \{P\mathbf{x} \mid \mathbf{x} \in \mathcal{C}\}$. For each unit data vector $\mathbf{o} \in \mathbb{S}^{D-1}$, we need the **nearest codebook vector** — the one with the largest inner product (since all vectors are unit length, maximizing $\langle \mathbf{o}, P\mathbf{x}\rangle$ minimizes $\|\mathbf{o} - P\mathbf{x}\|^2$):

$$
\bar{\mathbf{x}} = \arg\max_{\mathbf{x} \in \mathcal{C}} \langle \mathbf{o}, P\mathbf{x} \rangle
$$

| Symbol | Meaning |
|:--|:--|
| $\bar{\mathbf{x}}$ | Nearest codebook vector (in original codebook coordinates) |
| $P$ | Random orthogonal matrix ($P^T P = I$) |
| $P\bar{\mathbf{x}}$ | Nearest codebook vector in data space, denoted $\bar{\mathbf{o}}$ |

**Brute-force problem:** The codebook $\mathcal{C}$ has $2^D$ vectors. For $D = 128$, that is $\sim 10^{38}$ vectors — searching them all is impossible.

### The Inverse Transform Trick

Since $P$ is orthogonal ($P^{-1} = P^T$), inner products are preserved under rotation:

$$
\langle \mathbf{o}, P\mathbf{x} \rangle = \mathbf{o}^T P\mathbf{x} = (P^T \mathbf{o})^T \mathbf{x} = \langle P^{-1}\mathbf{o}, \mathbf{x} \rangle
$$

| Symbol | Meaning |
|:--|:--|
| $P^{-1}\mathbf{o} = P^T\mathbf{o}$ | Data vector inverse-transformed into codebook space |
| $\langle P^{-1}\mathbf{o}, \mathbf{x}\rangle$ | Inner product computed in codebook space |

**Key insight:** Instead of rotating every codebook vector toward the data (rotating $2^D$ vectors), we rotate the **single data vector** into the codebook's coordinate system. This replaces an impossible search over $2^D$ vectors with a single $O(D^2)$ matrix-vector multiplication.

$$
\bar{\mathbf{x}} = \arg\max_{\mathbf{x} \in \mathcal{C}} \langle P^{-1}\mathbf{o}, \mathbf{x} \rangle
$$

### Why Signs Give the Optimal Answer

Let $\mathbf{v} = P^{-1}\mathbf{o}$ denote the inverse-transformed data vector. We need:

$$
\bar{\mathbf{x}} = \arg\max_{\mathbf{x} \in \mathcal{C}} \langle \mathbf{v}, \mathbf{x} \rangle = \arg\max_{\mathbf{x} \in \mathcal{C}} \sum_{i=1}^{D} v_i \cdot x_i
$$

Since each $x_i \in \{+1/\sqrt{D}, -1/\sqrt{D}\}$, the $i$-th term $v_i \cdot x_i$ is maximized by choosing $x_i$ to have the **same sign** as $v_i$:

$$
\bar{x}_i = \frac{\text{sign}(v_i)}{\sqrt{D}}
$$

| Symbol | Meaning |
|:--|:--|
| $v_i = (P^{-1}\mathbf{o})_i$ | $i$-th coordinate of inverse-transformed data vector |
| $\text{sign}(v_i)$ | Sign of $v_i$: $+1$ if positive, $-1$ if negative |
| $\bar{x}_i$ | $i$-th coordinate of nearest codebook vector |

**The optimization decomposes coordinate-wise.** Each coordinate independently chooses its sign, so the global optimum is found by $D$ independent sign decisions — no search required.

### The D-Bit String Representation

Since each coordinate stores exactly one bit of information (its sign), the quantization code is a binary string $\bar{\mathbf{x}}_b \in \{0, 1\}^D$:

$$
\bar{\mathbf{x}}_b[i] = \begin{cases} 1 & \text{if } (P^{-1}\mathbf{o})_i > 0 \\ 0 & \text{if } (P^{-1}\mathbf{o})_i \leq 0 \end{cases}
$$

The full codebook vector is reconstructed from the bit string:

$$
\bar{\mathbf{x}} = \frac{2 \bar{\mathbf{x}}_b - \mathbf{1}_D}{\sqrt{D}}
$$

| Symbol | Meaning | Example ($D=4$, bits $= 1011$) |
|:--|:--|:--|
| $\bar{\mathbf{x}}_b$ | Binary quantization code | $(1, 0, 1, 1)$ |
| $2\bar{\mathbf{x}}_b - \mathbf{1}_D$ | Maps $\{0,1\} \to \{-1,+1\}$ | $(+1, -1, +1, +1)$ |
| $\bar{\mathbf{x}}$ | Codebook vector | $(+0.5, -0.5, +0.5, +0.5)$ |
| $\bar{\mathbf{o}} = P\bar{\mathbf{x}}$ | Quantized vector in data space | (depends on $P$) |

**Storage cost:** Exactly $D$ bits per data vector. For $D = 128$, that is 16 bytes — dramatically smaller than the original 512 bytes ($128 \times 4$ bytes for float32).

### Computing $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$: The Quality Metric

The inner product between the quantized vector $\bar{\mathbf{o}} = P\bar{\mathbf{x}}$ and the original unit vector $\mathbf{o}$ measures quantization quality. Using the inverse transform:

$$
\langle \bar{\mathbf{o}}, \mathbf{o} \rangle = \langle P\bar{\mathbf{x}}, \mathbf{o} \rangle = \langle \bar{\mathbf{x}}, P^{-1}\mathbf{o} \rangle = \langle \bar{\mathbf{x}}, \mathbf{v} \rangle
$$

Substituting $\bar{x}_i = \text{sign}(v_i)/\sqrt{D}$:

$$
\langle \bar{\mathbf{o}}, \mathbf{o} \rangle = \sum_{i=1}^{D} \frac{\text{sign}(v_i)}{\sqrt{D}} \cdot v_i = \frac{1}{\sqrt{D}} \sum_{i=1}^{D} |v_i| = \frac{\|\mathbf{v}\|_1}{\sqrt{D}}
$$

| Symbol | Meaning |
|:--|:--|
| $\|\mathbf{v}\|_1 = \sum_i \|v_i\|$ | $L^1$ norm (sum of absolute values) of $P^{-1}\mathbf{o}$ |
| $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | Always positive (since each term is $\|v_i\|/\sqrt{D} \geq 0$) |

**Geometric meaning:** $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle = \cos\theta$ where $\theta$ is the angle between the original and quantized vectors. Values near 1 mean excellent quantization.

**Expected value:** The RaBitQ paper shows:

$$
\mathbb{E}[\langle \bar{\mathbf{o}}, \mathbf{o}\rangle] = \sqrt{\frac{D}{\pi}} \cdot \frac{2\,\Gamma(D/2)}{(D-1)\,\Gamma((D-1)/2)}
$$

This evaluates to approximately **0.798–0.800** for $D \in [100, 10^6]$, meaning the quantized vector captures ~80% of the original direction. The concentration of measure (US-002) guarantees this value barely fluctuates across different vectors or rotations.

### What Is Stored Per Data Vector

During the **index phase**, RaBitQ precomputes and stores three quantities per data vector:

| Stored Value | Type | Size | Purpose |
|:--|:--|:--|:--|
| $\bar{\mathbf{x}}_b$ | $D$-bit string | $D$ bits | The quantization code |
| $\|\mathbf{o}_r - \mathbf{c}\|$ | float | 4 bytes | Norm for distance recovery (US-009) |
| $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | float | 4 bytes | Denominator of the unbiased estimator |

**Total per vector:** $D + 64$ bits. For $D = 128$: $128 + 64 = 192$ bits = 24 bytes.

The quantization code $\bar{\mathbf{x}}_b$ goes into the **numerator** of the estimator (computing $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle$ via bitwise operations — see US-012). The scalar $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ goes into the **denominator** to remove bias (see US-011). The norm $\|\mathbf{o}_r - \mathbf{c}\|$ converts the inner product estimate back to a distance estimate.

### Reference Implementation

The [RaBitQ reference code](https://github.com/gaoj0017/RaBitQ/blob/main/data/rabitq.py) implements quantization in Python:

```python
# Generate random orthogonal matrix via QR decomposition
P = Orthogonal(D)   # P^T P = I
P = P.T             # Use transpose (P^{-1} = P^T for orthogonal)

# Inverse-transform: compute P^{-1}(o_r - c) for all vectors at once
XP = np.dot(X_pad, P) - CP[cluster_id]

# Extract signs → D-bit binary code
bin_XP = (XP > 0)

# Compute ⟨ō,o⟩ = ||v||_1 / √D, with v = P^{-1}o (unit normalized)
x0 = np.sum(XP[:, :B] * (2*bin_XP[:, :B] - 1) / B**0.5,
            axis=1, keepdims=True) / np.linalg.norm(XP, axis=1, keepdims=True)

# Pack bits into uint64 for efficient storage
uint64_XP = np.packbits(bin_XP.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
```

**Code-to-math mapping:**

| Code | Math |
|:--|:--|
| `XP = X_pad @ P - CP[cluster_id]` | $\mathbf{v}' = P^{-1}(\mathbf{o}_r - \mathbf{c})$ (unnormalized) |
| `bin_XP = (XP > 0)` | $\bar{\mathbf{x}}_b[i] = \mathbf{1}[v'_i > 0]$ |
| `(2*bin_XP - 1) / B**0.5` | $\bar{\mathbf{x}} = (2\bar{\mathbf{x}}_b - \mathbf{1})/\sqrt{B}$ |
| `np.sum(XP * ...) / np.linalg.norm(XP)` | $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle = \|\mathbf{v}\|_1 / \sqrt{D}$ |
| `np.packbits(...)` | Pack $D$ bits into $D/64$ uint64 values |

Note: The code uses `B = ceil(D/64)*64` (padded to 64-bit boundary) and divides by `np.linalg.norm(XP)` (= $\|\mathbf{o}_r - \mathbf{c}\|$) to normalize $P^{-1}(\mathbf{o}_r - \mathbf{c})$ to $P^{-1}\mathbf{o}$.

## Worked Example

### Quantizing a Unit Vector in $D = 4$

**Given:** Unit vector $\mathbf{o} = (0.6, -0.3, 0.7, -0.2)$ (verify: $\|\mathbf{o}\|^2 = 0.36 + 0.09 + 0.49 + 0.04 = 0.98 \approx 1$; close enough for illustration).

**Random orthogonal matrix** (a simple permutation-with-flips for clarity):

$$
P = \begin{pmatrix} 0 & 0 & -1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
$$

(Verify: $P^T P = I$, and each column is a unit vector.)

**Step 1: Inverse-transform.** $\mathbf{v} = P^{-1}\mathbf{o} = P^T\mathbf{o}$:

$$
\mathbf{v} = P^T \mathbf{o} = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ -1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} 0.6 \\ -0.3 \\ 0.7 \\ -0.2 \end{pmatrix} = \begin{pmatrix} -0.3 \\ -0.7 \\ -0.6 \\ -0.2 \end{pmatrix}
$$

**Step 2: Extract signs** → binary code $\bar{\mathbf{x}}_b$:

| Coordinate | $v_i$ | $\text{sign}(v_i)$ | $\bar{\mathbf{x}}_b[i]$ |
|:--|:--|:--|:--|
| 1 | $-0.3$ | $-1$ | $0$ |
| 2 | $-0.7$ | $-1$ | $0$ |
| 3 | $-0.6$ | $-1$ | $0$ |
| 4 | $-0.2$ | $-1$ | $0$ |

$\bar{\mathbf{x}}_b = (0, 0, 0, 0)$

**Step 3: Reconstruct codebook vector:**

$$
\bar{\mathbf{x}} = \frac{2(0,0,0,0) - (1,1,1,1)}{\sqrt{4}} = \frac{(-1,-1,-1,-1)}{2} = (-0.5, -0.5, -0.5, -0.5)
$$

**Step 4: Compute $\bar{\mathbf{o}} = P\bar{\mathbf{x}}$:**

$$
\bar{\mathbf{o}} = P\bar{\mathbf{x}} = \begin{pmatrix} 0 & 0 & -1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} -0.5 \\ -0.5 \\ -0.5 \\ -0.5 \end{pmatrix} = \begin{pmatrix} 0.5 \\ -0.5 \\ 0.5 \\ -0.5 \end{pmatrix}
$$

**Step 5: Compute $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$:**

$$
\langle \bar{\mathbf{o}}, \mathbf{o}\rangle = 0.5 \cdot 0.6 + (-0.5)(-0.3) + 0.5 \cdot 0.7 + (-0.5)(-0.2) = 0.3 + 0.15 + 0.35 + 0.1 = 0.9
$$

**Verify via $L^1$ formula:** $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle = \|\mathbf{v}\|_1 / \sqrt{D} = (0.3 + 0.7 + 0.6 + 0.2)/\sqrt{4} = 1.8/2 = 0.9$ ✓

**Storage summary for this vector:**

| Value | Content | Size |
|:--|:--|:--|
| $\bar{\mathbf{x}}_b$ | `0000` | 4 bits |
| $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | $0.9$ | 4 bytes |
| $\|\mathbf{o}_r - \mathbf{c}\|$ | (from index phase) | 4 bytes |

## Comparison with TurboQuant/PQ

| Aspect | RaBitQ | TurboQuant | Product Quantization |
|:--|:--|:--|:--|
| **Quantization** | Sign of $P^{-1}\mathbf{o}$ | Lloyd-Max per coordinate | Nearest sub-codebook centroid |
| **Time** | $O(D^2)$ (one matrix-vector multiply) | $O(D)$ (per-coordinate lookup) | $O(D \cdot k)$ ($k$ = sub-codebook size) |
| **Bits/vector** | $D$ (1 bit per dimension) | $b \cdot D$ (tunable $b$) | $M \cdot \lceil\log_2 k\rceil$ ($M$ sub-spaces) |
| **Bias** | Unbiased (with $\langle\bar{\mathbf{o}},\mathbf{o}\rangle$ correction) | Biased (MSE) or Unbiased (prod) | Biased |
| **Stored scalars** | 2 ($\langle\bar{\mathbf{o}},\mathbf{o}\rangle$, $\|\mathbf{o}_r - \mathbf{c}\|$) | 0 (for MSE) or 1 (residual norm) | 0 |

## Key Takeaways

- **Inverse-transform trick:** Instead of searching $2^D$ codebook vectors, compute $P^{-1}\mathbf{o}$ (one matrix-vector multiply) and take coordinate signs — $O(D^2)$ instead of $O(2^D)$
- **Coordinate-wise decomposition:** Each bit is determined independently by the sign of $(P^{-1}\mathbf{o})_i$, making the optimization trivially parallel
- **$D$-bit representation:** $\bar{\mathbf{x}}_b \in \{0,1\}^D$ encodes the nearest codebook vector; reconstruct via $\bar{\mathbf{x}} = (2\bar{\mathbf{x}}_b - \mathbf{1}_D)/\sqrt{D}$
- **Quality metric:** $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle = \|\mathbf{v}\|_1/\sqrt{D} \approx 0.8$ measures how well the quantized vector approximates the original
- **Three stored values** per vector: the $D$-bit code, $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$, and $\|\mathbf{o}_r - \mathbf{c}\|$ — together enabling unbiased distance estimation (US-011)

## Sources

- [RaBitQ paper — arXiv:2405.12497](https://arxiv.org/abs/2405.12497) — Section 3.1
- [RaBitQ paper — arXiv HTML](https://arxiv.org/html/2405.12497) — Section 3.1.3
- [RaBitQ reference implementation — rabitq.py](https://github.com/gaoj0017/RaBitQ/blob/main/data/rabitq.py) — Index phase Python code
- [RaBitQ reference implementation — ivf_rabitq.h](https://github.com/gaoj0017/RaBitQ/blob/main/src/ivf_rabitq.h) — C++ search data structures
- [RaBitQ GitHub repo](https://github.com/gaoj0017/RaBitQ)
- [RaBitQ Library](https://vectordb-ntu.github.io/RaBitQ-Library/)
