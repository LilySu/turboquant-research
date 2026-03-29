# RaBitQ Core Theory: Normalization and Codebook

## Prerequisites Refresher

### Unit Vectors and the Hypersphere

A **unit vector** has Euclidean norm 1: $\|\mathbf{v}\| = 1$. The set of all unit vectors in $\mathbb{R}^D$ forms the unit hypersphere $\mathbb{S}^{D-1}$ (see US-002 for details). RaBitQ maps all data and query vectors onto $\mathbb{S}^{D-1}$ via normalization.

### Centroid

The **centroid** $\mathbf{c}$ of a set of vectors $\{\mathbf{o}_1, \ldots, \mathbf{o}_n\}$ is their mean:

$$
\mathbf{c} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{o}_i
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{c}$ | Centroid (center of mass) of the data vectors |
| $n$ | Number of data vectors |

### Hypercube Vertices

The $D$-dimensional hypercube $\{-1, +1\}^D$ has $2^D$ vertices — every combination of $\pm 1$ in $D$ coordinates. For $D = 2$: the 4 corners $(\pm 1, \pm 1)$. For $D = 3$: the 8 corners of a cube. Each vertex, when scaled by $1/\sqrt{D}$, becomes a unit vector.

## Main Content

### Step 1: Normalization — Reducing Distance to Inner Product

RaBitQ begins by centering vectors on a centroid and normalizing to unit length. Given raw data vector $\mathbf{o}_r$ and centroid $\mathbf{c}$:

$$
\mathbf{o} = \frac{\mathbf{o}_r - \mathbf{c}}{\|\mathbf{o}_r - \mathbf{c}\|}
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{o}_r$ | Raw data vector (unnormalized) |
| $\mathbf{c}$ | Centroid of the cluster containing $\mathbf{o}_r$ |
| $\mathbf{o}_r - \mathbf{c}$ | Centered vector |
| $\|\mathbf{o}_r - \mathbf{c}\|$ | Distance from $\mathbf{o}_r$ to centroid |
| $\mathbf{o}$ | Normalized unit vector on $\mathbb{S}^{D-1}$ |

Query vectors normalize similarly: $\mathbf{q} = (\mathbf{q}_r - \mathbf{c}) / \|\mathbf{q}_r - \mathbf{c}\|$.

**Why normalize?** Because the squared distance between raw vectors decomposes cleanly:

$$
\|\mathbf{o}_r - \mathbf{q}_r\|^2 = \|\mathbf{o}_r - \mathbf{c}\|^2 + \|\mathbf{q}_r - \mathbf{c}\|^2 - 2 \|\mathbf{o}_r - \mathbf{c}\| \cdot \|\mathbf{q}_r - \mathbf{c}\| \cdot \langle \mathbf{o}, \mathbf{q}\rangle
$$

| Symbol | Meaning | Known at search time? |
|:--|:--|:--|
| $\|\mathbf{o}_r - \mathbf{c}\|^2$ | Squared distance from data to centroid | Yes (precomputed at index time) |
| $\|\mathbf{q}_r - \mathbf{c}\|^2$ | Squared distance from query to centroid | Yes (computed once per query) |
| $\langle \mathbf{o}, \mathbf{q}\rangle$ | Inner product of normalized vectors | **This is what RaBitQ estimates** |

**Derivation:** Expand $\|\mathbf{o}_r - \mathbf{q}_r\|^2 = \|(\mathbf{o}_r - \mathbf{c}) - (\mathbf{q}_r - \mathbf{c})\|^2$ and use $\langle \mathbf{o}, \mathbf{q}\rangle = \langle \frac{\mathbf{o}_r - \mathbf{c}}{\|\mathbf{o}_r - \mathbf{c}\|}, \frac{\mathbf{q}_r - \mathbf{c}}{\|\mathbf{q}_r - \mathbf{c}\|}\rangle$.

This reduces the **distance estimation** problem to an **inner product estimation** problem on unit vectors — exactly the setting where RaBitQ's theoretical guarantees apply.

### Step 2: The Bi-Valued Codebook

RaBitQ's codebook consists of all vectors whose coordinates are either $+1/\sqrt{D}$ or $-1/\sqrt{D}$:

$$
\mathcal{C} = \left\{ +\frac{1}{\sqrt{D}}, -\frac{1}{\sqrt{D}} \right\}^D
$$

| Symbol | Meaning | Example ($D = 3$) |
|:--|:--|:--|
| $\mathcal{C}$ | The codebook: set of all quantization targets | 8 vectors |
| $D$ | Dimension of the vectors | $3$ |
| $1/\sqrt{D}$ | Scale factor making each vector unit-length | $1/\sqrt{3} \approx 0.577$ |
| $2^D$ | Number of vectors in $\mathcal{C}$ | $8$ |

**Why are these unit vectors?** Each vector $\mathbf{x} \in \mathcal{C}$ has:

$$
\|\mathbf{x}\|^2 = \sum_{i=1}^{D} x_i^2 = D \cdot \frac{1}{D} = 1
$$

**Why $2^D$ vectors?** Each of $D$ coordinates independently chooses from $\{+1/\sqrt{D}, -1/\sqrt{D}\}$, giving $2^D$ combinations. These are the scaled vertices of the $D$-dimensional hypercube.

**Geometric picture:** In $D = 2$, the codebook is the 4 points $(\pm 1/\sqrt{2}, \pm 1/\sqrt{2})$ — the corners of a square inscribed in the unit circle. In $D = 3$, the 8 vertices of a cube inscribed in the unit sphere. In $D = 128$, a $2^{128}$-vertex hypercube inscribed in $\mathbb{S}^{127}$.

### Step 3: Random Rotation — Removing Codebook Bias

The deterministic codebook $\mathcal{C}$ is biased: it has preferred directions (aligned with coordinate axes). Data vectors near these directions quantize well; others don't. To fix this, RaBitQ applies a **random orthogonal rotation** $P$:

$$
\mathcal{C}_{\text{rand}} = \{P\mathbf{x} \mid \mathbf{x} \in \mathcal{C}\}
$$

| Symbol | Meaning |
|:--|:--|
| $P \in \mathbb{R}^{D \times D}$ | Random orthogonal matrix ($P^T P = I$) |
| $\mathcal{C}_{\text{rand}}$ | Rotated codebook |
| $P\mathbf{x}$ | Rotated codebook vector |

**Key properties of the rotation:**
1. **Preserves norms:** $\|P\mathbf{x}\| = \|\mathbf{x}\| = 1$, so $\mathcal{C}_{\text{rand}}$ still consists of unit vectors
2. **Preserves inner products:** $\langle P\mathbf{x}, P\mathbf{y}\rangle = \langle \mathbf{x}, \mathbf{y}\rangle$
3. **Removes directional bias:** No direction is systematically closer to or farther from the codebook
4. **Enables theoretical analysis:** The randomness connects to the Johnson-Lindenstrauss framework, providing concentration bounds

**Connection to JL transformation:** The random rotation $P$ serves the same role as the JL projection matrix — it makes the codebook "look random" from the data's perspective, enabling probabilistic guarantees. The key theorem from JL theory: inner products between a fixed vector and a randomly rotated codebook vector concentrate around their expected values with sub-Gaussian tails.

**Why rotation removes preference:** Without rotation, a data vector $\mathbf{o} = (1, 0, 0, \ldots)$ would always quantize to $(1/\sqrt{D}, -1/\sqrt{D}, \ldots)$ with only the first coordinate correct. With rotation, the codebook is isotropically distributed, and the quantization error depends only on geometric properties (the angle between $\mathbf{o}$ and its nearest codebook vector), not on the specific direction of $\mathbf{o}$.

### Contrast with TurboQuant

| Aspect | TurboQuant | RaBitQ |
|:--|:--|:--|
| **What gets rotated** | The data vector $\mathbf{x}$ | The codebook $\mathcal{C}$ |
| **Codebook** | Lloyd-Max centroids for Beta distribution | $\{+1/\sqrt{D}, -1/\sqrt{D}\}^D$ (hypercube vertices) |
| **Quantization** | Nearest centroid per coordinate (scalar) | Nearest hypercube vertex (vector) |
| **Bits per vector** | $b \cdot D$ (tunable $b$) | $D$ bits (1 bit per coordinate) |
| **Application** | KV cache compression (LLM inference) | ANN search (vector databases) |

Both methods use random orthogonal rotation to enable theoretical guarantees — this is the shared foundation explored in US-002 and US-004.

## Worked Example

### Normalizing and Quantizing in $D = 4$

**Raw vectors:** $\mathbf{o}_r = (3, 1, 4, 2)$, centroid $\mathbf{c} = (2, 2, 2, 2)$.

**Step 1: Center.** $\mathbf{o}_r - \mathbf{c} = (1, -1, 2, 0)$, $\|\mathbf{o}_r - \mathbf{c}\| = \sqrt{6} \approx 2.449$.

**Step 2: Normalize.** $\mathbf{o} = (0.408, -0.408, 0.816, 0)$.

**Step 3: Codebook.** $\mathcal{C}$ has $2^4 = 16$ vectors, each with entries $\pm 1/\sqrt{4} = \pm 0.5$.

**Without rotation** ($P = I$): The nearest codebook vector has signs matching $\mathbf{o}$:

$$
\bar{\mathbf{x}} = (+0.5, -0.5, +0.5, +0.5)
$$

(The 4th coordinate is $+0.5$ since $0 > 0$ is ambiguous; convention is $+$.)

$\langle \bar{\mathbf{x}}, \mathbf{o}\rangle = 0.5 \cdot 0.408 + (-0.5)(-0.408) + 0.5 \cdot 0.816 + 0.5 \cdot 0 = 0.816$

**With rotation:** A random $P$ would rotate both $\mathcal{C}$ and the procedure, but the quantization code is still just $D = 4$ bits: the signs of $P^{-1}\mathbf{o}$.

## Key Takeaways

- RaBitQ **normalizes** vectors to $\mathbb{S}^{D-1}$, reducing distance estimation to inner product estimation
- The **distance formula** $\|\mathbf{o}_r - \mathbf{q}_r\|^2 = \|\mathbf{o}_r - \mathbf{c}\|^2 + \|\mathbf{q}_r - \mathbf{c}\|^2 - 2\|\mathbf{o}_r - \mathbf{c}\|\|\mathbf{q}_r - \mathbf{c}\|\langle \mathbf{o}, \mathbf{q}\rangle$ makes only $\langle \mathbf{o}, \mathbf{q}\rangle$ unknown
- The **codebook** $\mathcal{C} = \{+1/\sqrt{D}, -1/\sqrt{D}\}^D$ has $2^D$ unit vectors (scaled hypercube vertices)
- **Random rotation** $P$ removes directional bias and enables JL-based theoretical guarantees
- Each data vector is stored as just **$D$ bits** (the sign pattern) plus precomputed scalars
- TurboQuant rotates data, RaBitQ rotates the codebook — both exploit the same rotational symmetry

## Sources

- [RaBitQ paper — arXiv:2405.12497](https://arxiv.org/abs/2405.12497)
- [RaBitQ paper — arXiv HTML](https://arxiv.org/html/2405.12497)
- [RaBitQ — ACM Digital Library (SIGMOD 2024)](https://dl.acm.org/doi/10.1145/3654970)
- [RaBitQ GitHub repo](https://github.com/gaoj0017/RaBitQ)
- [RaBitQ Library](https://vectordb-ntu.github.io/RaBitQ-Library/)
- [Extended RaBitQ (SIGMOD 2025)](https://github.com/VectorDB-NTU/Extended-RaBitQ)
- [Elasticsearch: RaBitQ Binary Quantization 101](https://www.elastic.co/search-labs/blog/rabitq-explainer-101)
- [EmergentMind: RaBitQ summary](https://www.emergentmind.com/papers/2405.12497)
- [alphaXiv: RaBitQ discussion](https://www.alphaxiv.org/abs/2405.12497)
- [Johnson-Lindenstrauss Lemma — Wikipedia](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
