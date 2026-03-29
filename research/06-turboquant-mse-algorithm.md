# TurboQuant_mse Algorithm (Algorithm 1)

## Prerequisites Refresher

### QR Decomposition

Any matrix $A \in \mathbb{R}^{d \times d}$ with linearly independent columns can be factored as:

$$
A = QR
$$

| Symbol | Meaning |
|:--|:--|
| $A$ | Input matrix ($d \times d$) |
| $Q$ | Orthogonal matrix: $Q^T Q = I$ (columns are orthonormal) |
| $R$ | Upper triangular matrix |

**Why does QR give an orthogonal matrix?** The Gram-Schmidt process takes the columns of $A$ and produces orthonormal vectors one by one, removing the component along each previously computed direction. The resulting orthonormal vectors form $Q$; the coefficients form $R$.

**For TurboQuant:** To generate a random orthogonal (rotation) matrix $\Pi$:
1. Fill a $d \times d$ matrix $A$ with i.i.d. $\mathcal{N}(0, 1)$ entries
2. Compute $A = QR$
3. The $Q$ matrix is a uniformly random orthogonal matrix — this is $\Pi$

This works because the Gaussian distribution is rotationally invariant, so $Q$ is uniform over the orthogonal group $O(d)$.

### Matrix-Vector Multiplication

Multiplying $\Pi \in \mathbb{R}^{d \times d}$ by $\mathbf{x} \in \mathbb{R}^d$ gives $\mathbf{y} = \Pi \mathbf{x} \in \mathbb{R}^d$, where each output coordinate is:

$$
y_j = \sum_{k=1}^{d} \Pi_{jk} \, x_k
$$

| Symbol | Meaning |
|:--|:--|
| $y_j$ | $j$-th coordinate of the rotated vector |
| $\Pi_{jk}$ | Entry in row $j$, column $k$ of the rotation matrix |
| $x_k$ | $k$-th coordinate of the input vector |

Cost: $O(d^2)$ operations. (Can be reduced to $O(d \log d)$ with structured random rotations, but the paper uses the full matrix.)

## Main Content

### Algorithm 1: TurboQuant_mse — Annotated Pseudocode

#### Global Setup (runs once)

| Line | Code | Explanation |
|:-:|:--|:--|
| 1 | **Input:** dimension $d$, bit-width $b$ | $d$ = vector size, $b$ = bits per coordinate |
| 2 | Generate $\Pi \in \mathbb{R}^{d \times d}$ via QR of $\mathcal{N}(0,1)$ matrix | Random orthogonal rotation (see refresher above) |
| 3 | Compute centroids $c_1, \ldots, c_{2^b}$ minimizing Eq. 4 | Lloyd-Max codebook for the Beta distribution (US-003) |

The centroids $c_i$ are precomputed by solving the continuous k-means problem from US-003:

$$
\{c_i\} = \arg\min \sum_{i=1}^{2^b} \int_{\frac{c_{i-1}+c_i}{2}}^{\frac{c_i+c_{i+1}}{2}} |x - c_i|^2 f_X(x) \, dx
$$

| Symbol | Meaning |
|:--|:--|
| $c_1, \ldots, c_{2^b}$ | Optimal reconstruction levels (centroids) |
| $f_X(x)$ | Beta distribution from Lemma 1 (US-002) |
| $2^b$ | Number of quantization levels ($b$ bits) |

#### Quant_mse (quantization)

| Line | Code | Explanation |
|:-:|:--|:--|
| 5 | $\mathbf{y} \leftarrow \Pi \cdot \mathbf{x}$ | Rotate input into random basis |
| 6 | $\text{idx}_j \leftarrow \arg\min_{k \in [2^b]} \|y_j - c_k\|$ for each $j$ | Find nearest centroid for each coordinate |
| 7 | **output:** idx | Return $d$ indices, each $b$ bits → total $bd$ bits |

**Line 5** rotates the input so coordinates follow the Beta distribution (Lemma 1). **Line 6** is a nearest-neighbor lookup — for each coordinate, find which of the $2^b$ centroids is closest. This is the heart of the quantizer: a scalar operation applied independently to each coordinate.

#### DeQuant_mse (dequantization)

| Line | Code | Explanation |
|:-:|:--|:--|
| 9 | $\tilde{y}_j \leftarrow c_{\text{idx}_j}$ for each $j$ | Look up centroid values from indices |
| 10 | $\tilde{\mathbf{x}} \leftarrow \Pi^T \cdot \tilde{\mathbf{y}}$ | Rotate back to original basis |
| 11 | **output:** $\tilde{\mathbf{x}}$ | Reconstructed vector |

**Line 9** replaces each index with its centroid value. **Line 10** applies the inverse rotation $\Pi^T = \Pi^{-1}$ (since $\Pi$ is orthogonal) to recover a vector in the original coordinate system.

### Theorem 1: MSE Distortion Bound

**Statement.** For any bit-width $b \geq 1$ and any $\mathbf{x} \in \mathbb{S}^{d-1}$:

$$
D_{\text{mse}} := \mathbb{E}_\Pi\!\left[\|\mathbf{x} - \tilde{\mathbf{x}}\|_2^2\right] \leq \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b}
$$

| Symbol | Meaning | Example ($b=2$) |
|:--|:--|:--|
| $D_{\text{mse}}$ | Expected MSE distortion | $\leq 0.170$ |
| $\mathbb{E}_\Pi$ | Expectation over the random rotation $\Pi$ | — |
| $\mathbf{x}$ | Original unit vector | — |
| $\tilde{\mathbf{x}}$ | Reconstructed vector (after quant + dequant) | — |
| $\sqrt{3}\pi/2 \approx 2.72$ | Gap to the information-theoretic optimum | — |
| $4^{-b}$ | Exponential improvement per bit | $4^{-2} = 0.0625$ |

**Specific values:**

| $b$ | $D_{\text{mse}}$ (upper bound) | $D_{\text{mse}}$ (empirical) |
|:-:|:-:|:-:|
| 1 | 0.680 | $\approx 0.36$ |
| 2 | 0.170 | $\approx 0.117$ |
| 3 | 0.0425 | $\approx 0.03$ |
| 4 | 0.0106 | $\approx 0.009$ |

### Theorem 1: Step-by-Step Proof

**Step 1: Rotation preserves MSE.** Since $\Pi$ is orthogonal:

$$
\|\mathbf{x} - \tilde{\mathbf{x}}\|_2^2 = \|\Pi\mathbf{x} - \Pi\tilde{\mathbf{x}}\|_2^2 = \|\mathbf{y} - \tilde{\mathbf{y}}\|_2^2
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{y} = \Pi\mathbf{x}$ | Rotated input |
| $\tilde{\mathbf{y}}$ | Quantized coordinates (centroids looked up from indices) |

The MSE in the original space equals the MSE in the rotated space.

**Step 2: Decompose into per-coordinate errors.** Since quantization is applied independently:

$$
\|\mathbf{y} - \tilde{\mathbf{y}}\|_2^2 = \sum_{j=1}^{d} (y_j - c_{\text{idx}_j})^2
$$

**Step 3: Each coordinate has the same distribution.** By Lemma 1 (US-002), each $y_j$ follows the Beta distribution $f_X$. By the codebook construction (US-003), the per-coordinate MSE is $\mathcal{C}(f_X, b)$. So:

$$
D_{\text{mse}} = \mathbb{E}\!\left[\sum_{j=1}^d (y_j - c_{\text{idx}_j})^2\right] = d \cdot \mathcal{C}(f_X, b)
$$

| Symbol | Meaning |
|:--|:--|
| $\mathcal{C}(f_X, b)$ | Per-coordinate MSE of the optimal $b$-bit quantizer for $f_X$ |
| $d \cdot \mathcal{C}(f_X, b)$ | Total MSE across all coordinates |

**Step 4: Apply the Panter-Dite bound (US-003).**

$$
\mathcal{C}(f_X, b) \leq \frac{1}{12}\left(\int f_X(x)^{1/3} dx\right)^3 \cdot 4^{-b} = \frac{\sqrt{3}\,\pi}{2d} \cdot 4^{-b}
$$

Multiplying by $d$:

$$
D_{\text{mse}} = d \cdot \mathcal{C}(f_X, b) \leq \frac{\sqrt{3}\,\pi}{2} \cdot 4^{-b} \quad \blacksquare
$$

## Worked Example

### Quantizing a Specific Vector with $b = 2$, $d = 4$

Let $\mathbf{x} = (0.5, 0.5, 0.5, 0.5)$ (unit norm). Suppose the random rotation $\Pi$ gives:

$$
\mathbf{y} = \Pi\mathbf{x} = (0.82, -0.31, 0.47, -0.12)
$$

**2-bit codebook** for $\mathcal{N}(0, 1/4)$ (scaling $\mathcal{N}(0,1)$ values by $1/\sqrt{4} = 0.5$):

| Level | Value |
|:-:|:-:|
| $c_1$ | $-0.755$ |
| $c_2$ | $-0.226$ |
| $c_3$ | $+0.226$ |
| $c_4$ | $+0.755$ |

Boundaries: $-0.491, 0, +0.491$.

**Quantization** (Line 6 — nearest centroid):

| $j$ | $y_j$ | Nearest $c_k$ | idx |
|:-:|:-:|:-:|:-:|
| 1 | $0.82$ | $c_4 = 0.755$ | 4 |
| 2 | $-0.31$ | $c_2 = -0.226$ | 2 |
| 3 | $0.47$ | $c_3 = 0.226$ | 3 |
| 4 | $-0.12$ | $c_2 = -0.226$ | 2 |

**Dequantization** (Lines 9-10):

$\tilde{\mathbf{y}} = (0.755, -0.226, 0.226, -0.226)$

$\tilde{\mathbf{x}} = \Pi^T \tilde{\mathbf{y}}$ (rotate back — a $4 \times 4$ matrix multiply)

**Per-coordinate errors:** $(0.82 - 0.755)^2 + (-0.31 + 0.226)^2 + (0.47 - 0.226)^2 + (-0.12 + 0.226)^2 = 0.004 + 0.007 + 0.060 + 0.011 = 0.082$

This matches the expected MSE for $b = 2$: $d \cdot \mathcal{C}(f_X, 2) \approx 4 \times 0.029 = 0.117$.

## Connection to TurboQuant

1. **Algorithm 1 is the MSE workhorse.** It provides the low-distortion reconstruction that captures most of the signal. In TurboQuant$_{\text{prod}}$ (US-007), it uses $b-1$ bits as the first stage.

2. **The rotation is the key insight.** Without rotation, worst-case inputs (e.g., $\mathbf{x} = (1, 0, \ldots, 0)$) would have all information in one coordinate, making scalar quantization ineffective. The random rotation spreads information uniformly across coordinates.

3. **Scalar quantization is embarrassingly parallel.** Each coordinate is quantized independently — ideal for GPU implementation. The only non-trivial operation is the matrix multiply ($O(d^2)$).

4. **Codebook is data-independent.** Unlike product quantization or learned methods, TurboQuant's codebook depends only on $d$ and $b$ — computed once and reused for all vectors.

## Key Takeaways

- **QR decomposition** of a Gaussian matrix gives a uniformly random orthogonal matrix $\Pi$
- **Quant_mse**: rotate → nearest-centroid lookup → store indices ($bd$ bits total)
- **DeQuant_mse**: look up centroids → rotate back via $\Pi^T$
- **Theorem 1**: $D_{\text{mse}} \leq \frac{\sqrt{3}\pi}{2} \cdot 4^{-b}$, proved by decomposing into per-coordinate errors and applying Panter-Dite
- The rotation is what makes scalar quantization near-optimal: it turns any input into a "nice" distribution
- Empirical MSE is even better than the bound (0.117 vs 0.170 at $b = 2$)

## Sources

- [TurboQuant paper — arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [TurboQuant HTML — arXiv](https://arxiv.org/html/2504.19874v1)
- [TurboQuant — OpenReview (ICLR 2026)](https://openreview.net/forum?id=tO3ASKZlok)
- [QR Decomposition — Wikipedia](https://en.wikipedia.org/wiki/QR_decomposition)
- [Gram-Schmidt Process — Wikipedia](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
- [UCLA: QR Decomposition with Gram-Schmidt](https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf)
- [UBC MATH 307: QR Decomposition](https://ubcmath.github.io/MATH307/orthogonality/qr.html)
- [QuantEcon: QR Decomposition in Python](https://python.quantecon.org/qr_decomp.html)
- [Kwok (Medium): QR Decomposition by Gram-Schmidt](https://kwokanthony.medium.com/important-decomposition-in-linear-algebra-detailed-explanation-on-qr-decomposition-by-classical-3f8f5425915f)
- [StatLect: QR Decomposition](https://www.statlect.com/matrix-algebra/QR-decomposition)
- [LibreTexts: QR Decomposition](https://math.libretexts.org/Bookshelves/Linear_Algebra/Map:_Linear_Algebra_(Waldron_Cherney_and_Denton)/14:_Orthonormal_Bases_and_Complements/14.05:_QR_Decomposition)
