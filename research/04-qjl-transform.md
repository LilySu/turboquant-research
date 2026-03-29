# The QJL (Quantized Johnson-Lindenstrauss) Transform

## Prerequisites Refresher

### Inner Product $\langle \mathbf{x}, \mathbf{y}\rangle$

The **inner product** (dot product) of two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ is:

$$
\langle \mathbf{x}, \mathbf{y}\rangle = \sum_{i=1}^{d} x_i \, y_i = \|\mathbf{x}\| \, \|\mathbf{y}\| \cos\theta
$$

| Symbol | Meaning |
|:--|:--|
| $\langle \mathbf{x}, \mathbf{y}\rangle$ | Inner product of vectors $\mathbf{x}$ and $\mathbf{y}$ |
| $x_i, y_i$ | The $i$-th coordinates of each vector |
| $\|\mathbf{x}\|, \|\mathbf{y}\|$ | Euclidean norms (lengths) of the vectors |
| $\theta$ | Angle between $\mathbf{x}$ and $\mathbf{y}$ |

**Geometric interpretation:** The inner product measures *how aligned* two vectors are. If $\theta = 0$ (same direction), $\langle \mathbf{x}, \mathbf{y}\rangle = \|\mathbf{x}\|\|\mathbf{y}\|$ (maximum). If $\theta = 90°$ (perpendicular), $\langle \mathbf{x}, \mathbf{y}\rangle = 0$. In attention mechanisms, the inner product $\langle \mathbf{q}, \mathbf{k}\rangle$ determines how much a query attends to a key.

### Expectation and Variance

For a random variable $Z$:
- **Expectation**: $\mathbb{E}[Z]$ is the average value of $Z$ (see US-001)
- **Variance**: $\text{Var}(Z) = \mathbb{E}[(Z - \mathbb{E}[Z])^2] = \mathbb{E}[Z^2] - (\mathbb{E}[Z])^2$

Variance measures how spread out $Z$ is around its mean. Low variance means $Z$ is consistently close to $\mathbb{E}[Z]$.

### The Johnson-Lindenstrauss (JL) Lemma

The **JL Lemma** (1984) states: any $n$ points in high-dimensional space can be projected to $k = O(\log n / \varepsilon^2)$ dimensions while preserving all pairwise distances within a factor of $(1 \pm \varepsilon)$.

**How?** Multiply by a random matrix $A \in \mathbb{R}^{k \times d}$ with i.i.d. $\mathcal{N}(0, 1)$ entries:

$$
\mathbf{x} \mapsto \frac{1}{\sqrt{k}} A \mathbf{x}
$$

| Symbol | Meaning |
|:--|:--|
| $A$ | Random Gaussian matrix ($k \times d$) |
| $k$ | Target (lower) dimension |
| $d$ | Original (high) dimension |
| $1/\sqrt{k}$ | Normalization to preserve expected norm |

**Why it works:** By concentration of measure (US-002), the projected norm $\|A\mathbf{x}/\sqrt{k}\|^2$ concentrates around $\|\mathbf{x}\|^2$. Gaussian random matrices are rotationally invariant, so only the *angle* between vectors matters — the projection treats all directions equally.

**Key insight for QJL:** Instead of storing the full projected vector, QJL keeps only the **signs** of each projected coordinate. This extreme 1-bit quantization still preserves inner product information.

## Main Content

### Definition 1: The QJL Mapping

The QJL transform (Zandieh et al., 2024) quantizes each coordinate to a single sign bit after a JL projection.

**Quantization (forward):**

$$
Q_{\text{qjl}}(\mathbf{x}) := \text{sign}(S \cdot \mathbf{x}) \in \{-1, +1\}^d
$$

| Symbol | Meaning |
|:--|:--|
| $Q_{\text{qjl}}(\mathbf{x})$ | Quantized output: a vector of $d$ sign bits |
| $S \in \mathbb{R}^{d \times d}$ | Random matrix with i.i.d. $\mathcal{N}(0, 1)$ entries |
| $S \cdot \mathbf{x}$ | Matrix-vector product: projects $\mathbf{x}$ into a new basis |
| $\text{sign}(\cdot)$ | Element-wise: $+1$ if positive, $-1$ if negative |

**Dequantization (inverse):**

$$
Q_{\text{qjl}}^{-1}(\mathbf{z}) := \frac{\sqrt{\pi/2}}{d} \, S^T \mathbf{z}, \quad \mathbf{z} \in \{-1, +1\}^d
$$

| Symbol | Meaning |
|:--|:--|
| $Q_{\text{qjl}}^{-1}(\mathbf{z})$ | Reconstructed vector from sign bits |
| $\sqrt{\pi/2}$ | Correction factor ($\approx 1.253$) — compensates for sign quantization |
| $d$ | Dimension (normalizes the sum) |
| $S^T$ | Transpose of $S$ (maps back to original space) |
| $\mathbf{z}$ | The stored sign bits $\in \{-1, +1\}^d$ |

**Storage:** Only $d$ bits (the signs) plus the random seed for $S$. No scale factors or zero points needed — a key advantage over traditional quantization.

### Why sign() Preserves Inner Product Information

**Geometric argument (Charikar's SimHash, 2002).** Consider two vectors $\mathbf{x}$ and $\mathbf{y}$ with angle $\theta$ between them. A random Gaussian vector $\mathbf{g}$ defines a random hyperplane through the origin. The probability that $\mathbf{x}$ and $\mathbf{y}$ land on *different* sides is:

$$
\Pr[\text{sign}(\langle \mathbf{g}, \mathbf{x}\rangle) \neq \text{sign}(\langle \mathbf{g}, \mathbf{y}\rangle)] = \frac{\theta}{\pi}
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{g}$ | A single random Gaussian vector (one row of $S$) |
| $\theta$ | Angle between $\mathbf{x}$ and $\mathbf{y}$ |
| $\theta/\pi$ | Fraction of random hyperplanes that separate $\mathbf{x}$ from $\mathbf{y}$ |

**Intuition:** If $\theta$ is small (vectors nearly parallel), almost no random hyperplane separates them, so their sign bits almost always agree. If $\theta = \pi/2$ (perpendicular), exactly half of random hyperplanes separate them. With $d$ independent random projections, the fraction of disagreeing signs concentrates around $\theta/\pi$, encoding the angle (and thus the inner product for unit vectors) into binary.

### Lemma 4: Unbiasedness and Variance Bound

**Statement.** For any $\mathbf{x} \in \mathbb{S}^{d-1}$ and any $\mathbf{y} \in \mathbb{R}^d$:

**1. Unbiasedness:**

$$
\mathbb{E}\!\left[\langle \mathbf{y}, \, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{x}))\rangle\right] = \langle \mathbf{y}, \mathbf{x}\rangle
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbb{E}[\cdot]$ | Expectation over randomness in $S$ |
| $Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{x}))$ | Quantize $\mathbf{x}$, then reconstruct |
| $\langle \mathbf{y}, \cdot \rangle$ | Inner product with query $\mathbf{y}$ |

This says: *on average*, the inner product with the reconstructed vector equals the true inner product. No systematic over- or under-estimation.

**2. Variance bound:**

$$
\text{Var}\!\left(\langle \mathbf{y}, \, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{x}))\rangle\right) \leq \frac{\pi}{2d} \|\mathbf{y}\|_2^2
$$

| Symbol | Meaning |
|:--|:--|
| $\text{Var}(\cdot)$ | Variance: how much the estimate fluctuates |
| $\pi/(2d)$ | Decays with dimension — higher $d$ means less noise |
| $\|\mathbf{y}\|_2^2$ | Squared norm of query: error scales with query magnitude |

### Proof Sketch of Unbiasedness

**Step 1:** Expand the reconstructed inner product. Let $\mathbf{s}_j$ denote the $j$-th row of $S$:

$$
\langle \mathbf{y}, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{x})) \rangle = \frac{\sqrt{\pi/2}}{d} \sum_{j=1}^{d} \langle \mathbf{y}, \mathbf{s}_j \rangle \cdot \text{sign}(\langle \mathbf{s}_j, \mathbf{x} \rangle)
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{s}_j$ | $j$-th row of random matrix $S$ (a random Gaussian vector) |
| $\langle \mathbf{y}, \mathbf{s}_j \rangle$ | Projection of query onto $\mathbf{s}_j$ |
| $\text{sign}(\langle \mathbf{s}_j, \mathbf{x} \rangle)$ | Sign of the projection of $\mathbf{x}$ onto $\mathbf{s}_j$ |

**Step 2:** Take expectation of a single term. For a fixed pair $(\mathbf{x}, \mathbf{y})$ and random $\mathbf{s} \sim \mathcal{N}(\mathbf{0}, I)$:

$$
\mathbb{E}[\langle \mathbf{y}, \mathbf{s} \rangle \cdot \text{sign}(\langle \mathbf{s}, \mathbf{x} \rangle)] = \sqrt{\frac{2}{\pi}} \langle \mathbf{y}, \mathbf{x} \rangle
$$

This uses the fact that for jointly Gaussian $(U, V) = (\langle \mathbf{y}, \mathbf{s} \rangle, \langle \mathbf{s}, \mathbf{x} \rangle)$, the expectation $\mathbb{E}[U \cdot \text{sign}(V)] = \sqrt{2/\pi} \cdot \text{Cov}(U, V) / \sqrt{\text{Var}(V)}$, and $\text{Cov}(U, V) = \langle \mathbf{y}, \mathbf{x} \rangle$ while $\text{Var}(V) = \|\mathbf{x}\|^2 = 1$.

**Step 3:** Sum $d$ terms and apply the $\sqrt{\pi/2}/d$ scaling:

$$
\mathbb{E}[\text{estimator}] = \frac{\sqrt{\pi/2}}{d} \cdot d \cdot \sqrt{\frac{2}{\pi}} \langle \mathbf{y}, \mathbf{x} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle
$$

The $\sqrt{\pi/2}$ in the dequantization formula is chosen precisely to cancel the $\sqrt{2/\pi}$ from the sign expectation. $\blacksquare$

## Worked Examples

### Example: QJL on a 4D vector

Let $d = 4$, $\mathbf{x} = (0.5, 0.5, 0.5, 0.5)$ (unit norm), $\mathbf{y} = (1, 0, 0, 0)$.

True inner product: $\langle \mathbf{y}, \mathbf{x}\rangle = 0.5$.

Suppose one draw of $S$ gives $S\mathbf{x} = (0.73, -0.21, 0.44, -0.88)$:
- $Q_{\text{qjl}}(\mathbf{x}) = \text{sign}(S\mathbf{x}) = (+1, -1, +1, -1)$
- $Q_{\text{qjl}}^{-1}(\mathbf{z}) = \frac{\sqrt{\pi/2}}{4} S^T \mathbf{z}$, which produces some reconstructed vector $\hat{\mathbf{x}}$
- $\langle \mathbf{y}, \hat{\mathbf{x}} \rangle$ will be a noisy estimate of $0.5$

Over many random draws of $S$, the average of $\langle \mathbf{y}, \hat{\mathbf{x}} \rangle$ converges to $0.5$ (unbiasedness), with variance $\leq \frac{\pi}{2 \cdot 4} \cdot 1^2 = 0.393$.

### Variance Scaling with Dimension

| $d$ | Variance bound $\frac{\pi}{2d}\|\mathbf{y}\|^2$ (for $\|\mathbf{y}\|=1$) | Std. dev. |
|:-:|:-:|:-:|
| 4 | 0.3927 | 0.627 |
| 64 | 0.02454 | 0.157 |
| 128 | 0.01227 | 0.111 |
| 256 | 0.006136 | 0.078 |

At $d = 128$ (typical head dimension), the standard deviation of the QJL estimate is only $\approx 0.11$ — small compared to typical inner product magnitudes.

## Connection to TurboQuant

QJL is the second stage of TurboQuant's inner-product-optimized quantizer (**Algorithm 2**, TurboQuant$_{\text{prod}}$):

1. **Stage 1**: Apply TurboQuant$_{\text{mse}}$ with $b-1$ bits → get an MSE-optimal (but biased) approximation
2. **Compute residual**: $\mathbf{r} = \mathbf{x} - Q_{\text{mse}}^{-1}(Q_{\text{mse}}(\mathbf{x}))$
3. **Stage 2**: Apply QJL to the residual: $\text{qjl} = \text{sign}(S \cdot \mathbf{r})$, store $\|\mathbf{r}\|$
4. **Reconstruct**: $\hat{\mathbf{x}} = Q_{\text{mse}}^{-1}(\cdot) + \|\mathbf{r}\| \cdot Q_{\text{qjl}}^{-1}(\text{qjl})$

The MSE quantizer alone introduces inner product bias (explored in US-005). QJL on the residual corrects this bias because it is an *unbiased* estimator of the residual's contribution to the inner product.

**Why 1 bit is enough for the residual:** The residual $\mathbf{r}$ has small norm (most of the signal is captured by the $b-1$ bit MSE stage). QJL's variance is proportional to $\|\mathbf{r}\|^2$, which is already small — so even 1-bit quantization gives acceptable noise.

## Key Takeaways

- The **inner product** $\langle \mathbf{x}, \mathbf{y}\rangle$ measures alignment and is the core operation in attention
- The **JL Lemma** shows random projection preserves distances; QJL goes further by keeping only the **signs**
- **Definition 1**: $Q_{\text{qjl}}(\mathbf{x}) = \text{sign}(S\mathbf{x})$ with dequantization $Q_{\text{qjl}}^{-1}(\mathbf{z}) = \frac{\sqrt{\pi/2}}{d} S^T \mathbf{z}$
- **Unbiased**: $\mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}} \rangle] = \langle \mathbf{y}, \mathbf{x}\rangle$ — no systematic error
- **Low variance**: $\text{Var} \leq \frac{\pi}{2d}\|\mathbf{y}\|^2$ — noise decreases with dimension
- The $\sqrt{\pi/2}$ dequantization factor exists precisely to cancel the $\sqrt{2/\pi}$ from $\mathbb{E}[\text{sign}]$
- In TurboQuant, QJL is applied to the residual (1 bit) to correct the bias of the MSE quantizer ($b-1$ bits)

## Sources

- [QJL paper — arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- [QJL — AAAI 2025 proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34773)
- [QJL official GitHub repo](https://github.com/amirzandieh/QJL)
- [QJL — NSF PAR](https://par.nsf.gov/servlets/purl/10644417)
- [QJL — ResearchGate](https://www.researchgate.net/publication/381193095_QJL_1-Bit_Quantized_JL_Transform_for_KV_Cache_Quantization_with_Zero_Overhead)
- [QJL — Semantic Scholar](https://www.semanticscholar.org/paper/QJL:-1-Bit-Quantized-JL-Transform-for-KV-Cache-with-Zandieh-Daliri/7318a804566baadc9f4b4ca8255f78744e749a32)
- [TurboQuant paper — arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [TurboQuant HTML — arXiv](https://arxiv.org/html/2504.19874v1)
- [Johnson-Lindenstrauss Lemma — Wikipedia](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
- [Dasgupta & Gupta, "An Elementary Proof of JL"](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf)
- [Freksen, "Introduction to JL Transforms" — arXiv:2103.00564](https://arxiv.org/pdf/2103.00564)
- [Random Projection — Wikipedia](https://en.wikipedia.org/wiki/Random_projection)
- [Charikar (2002), "Similarity Estimation from Rounding Algorithms"](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf)
- [Locality-Sensitive Hashing — Wikipedia](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
- [Wisconsin CS731: Random Projection lecture](https://pages.cs.wisc.edu/~jerryzhu/cs731/projection.pdf)
- [TTIC: Random Projections and JL lemma](https://home.ttic.edu/~gregory/courses/LargeScaleLearning/lectures/jl.pdf)
- [UCI: JL Transformation and Random Projection](https://www.math.uci.edu/~chenlong/MathPKU/JL.pdf)
- [Princeton CS: Dimensionality Reduction and JL](https://www.cs.princeton.edu/~smattw/Teaching/Fa19Lectures/lec9/lec9.pdf)
- [scikit-learn: Random Projection documentation](https://scikit-learn.org/stable/modules/random_projection.html)
- [turboquant-pytorch — GitHub (community implementation)](https://github.com/tonbistudio/turboquant-pytorch)
