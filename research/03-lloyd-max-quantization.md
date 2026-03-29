# Lloyd-Max Scalar Quantization

## Prerequisites Refresher

### K-Means Clustering

**K-means** partitions data into $k$ groups by minimizing the total squared distance from each point to its assigned cluster center (centroid). The objective:

$$
\min_{c_1, \ldots, c_k} \sum_{i=1}^{k} \sum_{x \in S_i} |x - c_i|^2
$$

| Symbol | Meaning |
|:--|:--|
| $c_1, \ldots, c_k$ | Cluster centers (centroids) |
| $S_i$ | Set of data points assigned to cluster $i$ |
| $\|x - c_i\|^2$ | Squared distance from point $x$ to its centroid |

The algorithm alternates two steps:
1. **Assign** each point to the nearest centroid
2. **Update** each centroid to the mean of its assigned points

These are the same two conditions used in Lloyd-Max quantization.

### Quantization as Partitioning

A **scalar quantizer** maps a continuous value $x \in \mathbb{R}$ to one of $M = 2^b$ discrete **reconstruction levels** (centroids) $c_1, \ldots, c_M$. The real line is partitioned into $M$ intervals by **decision boundaries** $d_0 < d_1 < \cdots < d_M$:

- If $x \in [d_{i-1}, d_i)$, output $c_i$

The quantizer stores only the index $i$ (requiring $b$ bits), and the decoder reconstructs by looking up $c_i$.

## Main Content

### The Lloyd-Max Algorithm

The **Lloyd-Max quantizer** (Lloyd 1957/1982, Max 1960) finds the $M$-level scalar quantizer that minimizes MSE distortion. It relies on two optimality conditions:

#### Condition 1: Centroid Rule

Each reconstruction level $c_i$ must be the **conditional expectation** (centroid) of $X$ within its partition:

$$
c_i = \frac{\int_{d_{i-1}}^{d_i} x \, f_X(x) \, dx}{\int_{d_{i-1}}^{d_i} f_X(x) \, dx} = \mathbb{E}[X \mid d_{i-1} \leq X < d_i]
$$

| Symbol | Meaning |
|:--|:--|
| $c_i$ | Reconstruction level (centroid) for partition $i$ |
| $d_{i-1}, d_i$ | Left and right boundaries of partition $i$ |
| $f_X(x)$ | PDF of the source (e.g., Gaussian or Beta) |
| $\mathbb{E}[X \mid \cdot]$ | Conditional expectation: average of $X$ within the interval |

**Intuition:** The best representative for a region is the center of mass of the probability in that region.

#### Condition 2: Midpoint Rule (Nearest Neighbor)

Each decision boundary $d_i$ must be the **midpoint** between adjacent reconstruction levels:

$$
d_i = \frac{c_i + c_{i+1}}{2}, \quad i = 1, \ldots, M-1
$$

| Symbol | Meaning |
|:--|:--|
| $d_i$ | Decision boundary between partitions $i$ and $i+1$ |
| $c_i, c_{i+1}$ | Adjacent reconstruction levels |

**Intuition:** A point exactly between two centroids should be assigned to whichever is closer — the midpoint is the natural boundary.

#### The Iterative Algorithm

1. **Initialize** $M$ reconstruction levels $c_1, \ldots, c_M$ (e.g., uniformly spaced)
2. **Compute boundaries**: $d_i = (c_i + c_{i+1})/2$
3. **Update centroids**: $c_i = \mathbb{E}[X \mid d_{i-1} \leq X < d_i]$
4. **Repeat** steps 2-3 until convergence (distortion change $< \epsilon$)

This is exactly k-means applied to the continuous distribution $f_X(x)$.

### Equation 4: TurboQuant's Codebook Optimization

The TurboQuant paper formulates the optimal codebook as a continuous 1D k-means problem over the Beta distribution from Lemma 1:

$$
\mathcal{C}(f_X, b) := \min_{-1 \leq c_1 \leq \cdots \leq c_{2^b} \leq 1} \sum_{i=1}^{2^b} \int_{\frac{c_{i-1}+c_i}{2}}^{\frac{c_i+c_{i+1}}{2}} |x - c_i|^2 \cdot f_X(x) \, dx
$$

| Symbol | Meaning |
|:--|:--|
| $\mathcal{C}(f_X, b)$ | Minimum achievable distortion for $b$-bit quantization of source $f_X$ |
| $b$ | Bits per coordinate (yields $2^b$ reconstruction levels) |
| $c_1, \ldots, c_{2^b}$ | Reconstruction levels (centroids), sorted in $[-1, 1]$ |
| $\frac{c_{i-1}+c_i}{2}$ | Left boundary of partition $i$ (midpoint rule) |
| $\frac{c_i+c_{i+1}}{2}$ | Right boundary of partition $i$ |
| $\|x - c_i\|^2$ | Squared error: cost of representing $x$ by $c_i$ |
| $f_X(x)$ | Source PDF (the Beta distribution from Lemma 1 of US-002) |

The boundaries are set to $c_0 = -1$ and $c_{2^b+1} = 1$ (the support of $f_X$ on the unit sphere).

### Voronoi Tessellation

The midpoint boundaries $d_i = (c_i + c_{i+1})/2$ define a **Voronoi tessellation** of $[-1, 1]$: each point is assigned to its nearest centroid. In 1D, Voronoi cells are simply intervals.

**Why midpoints?** For any MSE-minimizing quantizer, if $x$ is closer to $c_i$ than to $c_{i+1}$, then $|x - c_i|^2 < |x - c_{i+1}|^2$. The crossover point where both distances are equal is exactly the midpoint $(c_i + c_{i+1})/2$.

### Optimal Codebook Values

Solving Equation 4 numerically via the Lloyd-Max algorithm gives the following codebooks.

#### For $\mathcal{N}(0, 1)$ (standard Gaussian, from Max 1960)

| $b$ | Levels ($M$) | Positive Reconstruction Levels | Positive Thresholds | MSE |
|:-:|:-:|:--|:--|:-:|
| 1 | 2 | $\pm 0.7979$ | $0$ | 0.3634 |
| 2 | 4 | $0.4528, \; 1.5104$ | $0, \; 0.9816$ | 0.1175 |
| 3 | 8 | $0.2451, \; 0.7560, \; 1.3440, \; 2.1520$ | $0, \; 0.5006, \; 1.0500, \; 1.7479$ | 0.03454 |
| 4 | 16 | $0.1284, \; 0.3881, \; 0.6568, \; 0.9424,$  $1.2562, \; 1.6180, \; 2.0690, \; 2.7326$ | $0, \; 0.2582, \; 0.5224, \; 0.7996,$  $1.0993, \; 1.4371, \; 1.8447, \; 2.4010$ | 0.009497 |

(Table is symmetric: negative levels are $-c_i$, negative thresholds are $-d_i$.)

#### For TurboQuant (coordinates of $\mathbb{S}^{d-1}$, approximately $\mathcal{N}(0, 1/d)$)

Since TurboQuant's coordinates have variance $1/d$ instead of $1$, the codebook values scale by $1/\sqrt{d}$:

| $b$ | Centroids (scaled) | Per-coordinate MSE |
|:-:|:--|:-:|
| 1 | $\pm 0.7979/\sqrt{d}$ | $\approx 0.36/d$ |
| 2 | $\pm 0.4528/\sqrt{d}, \; \pm 1.5104/\sqrt{d}$ | $\approx 0.117/d$ |
| 3 | (8 levels, each divided by $\sqrt{d}$) | $\approx 0.03/d$ |
| 4 | (16 levels, each divided by $\sqrt{d}$) | $\approx 0.009/d$ |

The total MSE across all $d$ coordinates is $d \times (\text{per-coordinate MSE})$, giving the Theorem 1 bound $D_{\text{mse}} \leq \frac{\sqrt{3}\pi}{2} \cdot 4^{-b}$.

### Panter-Dite High-Resolution Formula

For larger bit-widths ($b > 4$), computing exact Lloyd-Max codebooks becomes expensive. The **Panter-Dite formula** (1951) gives an asymptotic approximation:

$$
\mathcal{C}(f_X, b) \leq \frac{1}{12} \left(\int f_X(x)^{1/3} \, dx\right)^3 \cdot \frac{1}{4^b}
$$

| Symbol | Meaning |
|:--|:--|
| $\mathcal{C}(f_X, b)$ | MSE distortion of the optimal $b$-bit quantizer |
| $f_X(x)^{1/3}$ | Cube root of the PDF — gives more weight to high-density regions |
| $\int f_X(x)^{1/3} dx$ | A "spread" measure of the distribution |
| $1/12$ | Constant from uniform quantizer distortion ($\Delta^2/12$ for interval width $\Delta$) |
| $1/4^b$ | Exponential decay: same $4\times$ per bit as the Shannon lower bound |

**Derivation sketch.** At high bit rates, the optimal quantizer approaches a **locally uniform** quantizer whose step size varies with the PDF. The optimal step size at $x$ is proportional to $f_X(x)^{-1/3}$. Integrating the local distortion $\Delta^2/12$ weighted by $f_X(x)$ and optimizing gives the Panter-Dite formula.

For TurboQuant's $\mathcal{N}(0, 1/d)$ distribution, evaluating the integral gives:

$$
\mathcal{C}(f_X, b) \leq \frac{\sqrt{3}\,\pi}{2d} \cdot \frac{1}{4^b}
$$

| Symbol | Meaning |
|:--|:--|
| $\sqrt{3}\pi / 2$ | $\approx 2.72$ — the constant factor above the Shannon lower bound |
| $d$ | Dimension (per-coordinate distortion scales as $1/d$) |
| $4^{-b}$ | Exponential improvement per additional bit |

This matches the Theorem 1 bound and confirms the $\approx 2.72\times$ gap to optimality.

## Worked Examples

### Example: 1-bit Quantizer for $\mathcal{N}(0, 1)$

With $b = 1$ ($M = 2$ levels), the quantizer has one boundary at $d_1 = 0$ and two levels $\pm c$.

By the centroid condition:

$$
c = \mathbb{E}[X \mid X > 0] = \frac{\int_0^\infty x \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \, dx}{\int_0^\infty \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \, dx} = \frac{1/\sqrt{2\pi}}{1/2} = \sqrt{\frac{2}{\pi}} \approx 0.7979
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbb{E}[X \mid X > 0]$ | Average of $X$ given it's positive |
| $\frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ | Standard Gaussian PDF |

The MSE is:

$$
D = 2 \int_0^\infty (x - 0.7979)^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \, dx = 1 - \frac{2}{\pi} \approx 0.3634
$$

### Example: Codebook for TurboQuant with $d = 128$, $b = 2$

The $\mathcal{N}(0, 1)$ codebook has levels $\pm 0.4528$ and $\pm 1.5104$. Scaling by $1/\sqrt{128} \approx 0.0884$:

| Level (Gaussian) | Level (TurboQuant, $d=128$) |
|:-:|:-:|
| $-1.5104$ | $-0.1335$ |
| $-0.4528$ | $-0.0400$ |
| $+0.4528$ | $+0.0400$ |
| $+1.5104$ | $+0.1335$ |

Per-coordinate MSE: $0.1175 / 128 \approx 0.000918$. Total MSE across 128 dimensions: $0.1175$.

## Connection to TurboQuant

1. **Codebook design** is the core of TurboQuant's MSE quantizer. After random rotation, each coordinate is independently quantized using the Lloyd-Max codebook optimized for the Beta/Gaussian distribution.

2. **Precomputation**: The codebooks depend only on $d$ and $b$, not on the input data. They are computed once (offline) and stored as a lookup table, making quantization a simple nearest-centroid lookup at runtime.

3. **Panter-Dite** justifies the $\sqrt{3}\pi/2$ constant in Theorem 1 — it connects the practical quantizer distortion to the information-theoretic lower bound from Shannon (US-001).

4. **Independence assumption**: The Lloyd-Max quantizer is applied *independently* to each coordinate. This is justified by the near-independence of coordinates on $\mathbb{S}^{d-1}$ (US-002), and the error from this approximation vanishes for large $d$.

## Key Takeaways

- The **Lloyd-Max algorithm** is k-means for continuous distributions: alternate between midpoint boundaries and centroid updates
- **Equation 4** in TurboQuant is exactly this problem applied to the Beta distribution from Lemma 1
- **Voronoi boundaries** are midpoints because each point should map to its nearest centroid
- Standard Gaussian codebooks are well-known (Max 1960); TurboQuant scales them by $1/\sqrt{d}$
- The **Panter-Dite formula** gives asymptotic distortion for high bit-widths: $\frac{1}{12}(\int f_X^{1/3} dx)^3 \cdot 4^{-b}$
- For TurboQuant's distribution, this evaluates to $\frac{\sqrt{3}\pi}{2d} \cdot 4^{-b}$ — confirming the $2.72\times$ gap

## Sources

- [TurboQuant paper — arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [TurboQuant paper — arXiv HTML version](https://arxiv.org/html/2504.19874v1)
- [Lloyd's Algorithm — Wikipedia](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
- [Max, J. (1960), "Quantizing for Minimum Distortion" — NYU hosted PDF](https://cs.nyu.edu/~roweis/csc2515-2006/readings/max60.pdf)
- [Max (1960) — Semantic Scholar](https://www.semanticscholar.org/paper/Quantizing-for-minimum-distortion-Max/466d6ecaca726323d0ba2e9a41e9c255ba328e89)
- [Stanford EE398A: Quantization lecture](https://web.stanford.edu/class/ee398a/handouts/lectures/05-Quantization.pdf)
- [Stanford EE368B: Quantization handout](https://web.stanford.edu/class/ee368b/Handouts/06-Quantization.pdf)
- [MIT 6.450: Quantization chapter](https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/926689aaa62a0315473fa9b982de1b07_book_3.pdf)
- [Colorado State: Lloyd-Max Quantization lecture](https://www.engr.colostate.edu/ECE513/SP09/lectures/lectures7_8.pdf)
- [Northeastern CSG142: Lloyd-Max Quantization](https://www.khoury.northeastern.edu/home/gsharp/csg142-fall-2006/Lloyd-Max-Quant.pdf)
- [KTH: Scalar Quantizer lecture](https://people.kth.se/~mflierl/EQ2845/08-Quantization.pdf)
- [HHI: Optimal Scalar Quantization](https://iphome.hhi.de/schwarz/assets/dc/10-OptScalarQuant.pdf)
- [NPTEL Lecture 36: Lloyd-Max Quantizer](http://elearn.psgcas.ac.in/nptel/courses/video/117101053/lec36.pdf)
- [Gray & Neuhoff (1998), "Quantization" — IEEE Trans. IT survey](https://www.math.ucdavis.edu/~saito/data/quantization/44it06-gray.pdf)
- [Quantization (signal processing) — Wikipedia](https://en.wikipedia.org/wiki/Quantization_(signal_processing))
- [Scalar Quantization — ScienceDirect overview](https://www.sciencedirect.com/topics/computer-science/scalar-quantization)
- [Lloyd Max Quantizer Python implementation — GitHub Gist](https://gist.github.com/robodhruv/43c96c05f6dd51b5664c595184942cc5)
- [Stanford EE269: Nonuniform Quantization lecture](http://web.stanford.edu/class/ee269/Lecture_nonuniform_quantization.pdf)
- [MATLAB lloyds function documentation](https://www.mathworks.com/help/comm/ref/lloyds.html)
