# RaBitQ Unbiased Estimator and Error Bound

## Prerequisites Refresher

### Unbiased Estimator

An estimator $\hat{\theta}$ of a parameter $\theta$ is **unbiased** if $\mathbb{E}[\hat{\theta}] = \theta$. Unbiasedness means the estimator is correct **on average** — individual estimates may err in either direction, but the errors cancel out. This is critical for search: a biased estimator systematically over- or under-estimates distances, causing missed nearest neighbors.

### Orthogonal Projection and Decomposition

Any vector $\mathbf{q}$ can be decomposed into a component **along** $\mathbf{o}$ and a component **perpendicular** to $\mathbf{o}$:

$$
\mathbf{q} = \langle \mathbf{q}, \mathbf{o} \rangle \mathbf{o} + (\mathbf{q} - \langle \mathbf{q}, \mathbf{o}\rangle \mathbf{o})
$$

| Symbol | Meaning |
|:--|:--|
| $\langle \mathbf{q}, \mathbf{o}\rangle \mathbf{o}$ | Projection of $\mathbf{q}$ onto $\mathbf{o}$ (parallel component) |
| $\mathbf{q} - \langle \mathbf{q}, \mathbf{o}\rangle \mathbf{o}$ | Perpendicular component (residual) |
| $\|\mathbf{q} - \langle \mathbf{q}, \mathbf{o}\rangle \mathbf{o}\|$ | Length of the perpendicular component |

For unit vectors $\mathbf{o}$ and $\mathbf{q}$, the perpendicular component has length $\sqrt{1 - \langle \mathbf{o}, \mathbf{q}\rangle^2}$ (Pythagorean theorem on the unit sphere).

### Concentration Inequalities

A random variable $X$ is **sub-Gaussian** if its tails decay at least as fast as a Gaussian: $\Pr[|X - \mathbb{E}[X]| > t] \leq 2e^{-ct^2}$ for some constant $c$. Sub-Gaussian concentration is the mathematical tool behind RaBitQ's error bounds — it guarantees that the quantization error is small with overwhelming probability.

## Main Content

### The Geometric Decomposition (Lemma 3.1)

Consider three unit vectors: $\mathbf{o}$ (original data), $\mathbf{q}$ (query), and $\bar{\mathbf{o}} = P\bar{\mathbf{x}}$ (quantized data from US-010). We want to relate $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle$ (computable) to $\langle \mathbf{o}, \mathbf{q}\rangle$ (the target).

Define the **orthonormal direction** perpendicular to $\mathbf{o}$ in the plane spanned by $\mathbf{o}$ and $\mathbf{q}$:

$$
\mathbf{e}_1 = \frac{\mathbf{q} - \langle \mathbf{q}, \mathbf{o}\rangle \mathbf{o}}{\|\mathbf{q} - \langle \mathbf{q}, \mathbf{o}\rangle \mathbf{o}\|}
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{e}_1$ | Unit vector orthogonal to $\mathbf{o}$, in the $(\mathbf{o}, \mathbf{q})$ plane |
| $\langle \mathbf{o}, \mathbf{e}_1\rangle$ | $= 0$ by construction |
| $\|\mathbf{e}_1\|$ | $= 1$ by normalization |

**Lemma 3.1 (Geometric Relationship):** For unit vectors $\mathbf{o}$, $\mathbf{q}$, $\bar{\mathbf{o}}$:

**Collinear case** ($\mathbf{o} = \pm\mathbf{q}$):

$$
\langle \bar{\mathbf{o}}, \mathbf{q}\rangle = \langle \bar{\mathbf{o}}, \mathbf{o}\rangle \cdot \langle \mathbf{o}, \mathbf{q}\rangle \tag{Eq. 9}
$$

**Non-collinear case:**

$$
\langle \bar{\mathbf{o}}, \mathbf{q}\rangle = \langle \bar{\mathbf{o}}, \mathbf{o}\rangle \cdot \langle \mathbf{o}, \mathbf{q}\rangle + \langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle \cdot \sqrt{1 - \langle \mathbf{o}, \mathbf{q}\rangle^2} \tag{Eq. 10}
$$

| Symbol | Meaning | Role |
|:--|:--|:--|
| $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | Projection of $\bar{\mathbf{o}}$ onto $\mathbf{o}$ | **Signal** — carries the target $\langle \mathbf{o}, \mathbf{q}\rangle$ |
| $\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle$ | Projection of $\bar{\mathbf{o}}$ onto $\mathbf{e}_1$ | **Noise** — the quantization error |
| $\sqrt{1 - \langle \mathbf{o}, \mathbf{q}\rangle^2}$ | Length of $\mathbf{q}$'s perpendicular component | Scales the noise contribution |

**Proof sketch:** Write $\mathbf{q}$ in the $(\mathbf{o}, \mathbf{e}_1)$ basis:

$$
\mathbf{q} = \langle \mathbf{q}, \mathbf{o}\rangle \mathbf{o} + \langle \mathbf{q}, \mathbf{e}_1\rangle \mathbf{e}_1 + \text{(components orthogonal to both)}
$$

Since $\|\mathbf{q}\| = 1$ and the remaining components are orthogonal to the plane: $\langle \mathbf{q}, \mathbf{e}_1\rangle = \sqrt{1 - \langle \mathbf{o}, \mathbf{q}\rangle^2}$. Taking the inner product with $\bar{\mathbf{o}}$:

$$
\langle \bar{\mathbf{o}}, \mathbf{q}\rangle = \langle \mathbf{q}, \mathbf{o}\rangle \langle \bar{\mathbf{o}}, \mathbf{o}\rangle + \sqrt{1 - \langle \mathbf{o}, \mathbf{q}\rangle^2} \cdot \langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle + 0
$$

The third term vanishes because $\bar{\mathbf{o}}$'s projection onto directions outside the $(\mathbf{o}, \mathbf{q})$ plane contributes nothing to $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle$. $\blacksquare$

### From Geometry to Estimator (Equations 11–12)

Rearranging Eq. 10, divide both sides by $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ (positive with overwhelming probability since $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle \approx 0.8$):

$$
\frac{\langle \bar{\mathbf{o}}, \mathbf{q}\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle} = \langle \mathbf{o}, \mathbf{q}\rangle + \underbrace{\sqrt{1 - \langle \mathbf{o}, \mathbf{q}\rangle^2} \cdot \frac{\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle}}_{\text{error term}} \tag{Eq. 12}
$$

| Symbol | Meaning |
|:--|:--|
| $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle / \langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | RaBitQ's estimator for $\langle \mathbf{o}, \mathbf{q}\rangle$ |
| $\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle / \langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | The error ratio — must have expectation 0 for unbiasedness |

### Why the Estimator Is Unbiased (Theorem 3.2)

**Theorem 3.2 (Unbiasedness):**

$$
\mathbb{E}\left[\frac{\langle \bar{\mathbf{o}}, \mathbf{q}\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle}\right] = \langle \mathbf{o}, \mathbf{q}\rangle \tag{Eq. 13}
$$

**Why $\mathbb{E}[\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle] = 0$:** The key symmetry argument:

1. The random rotation $P$ is drawn uniformly from all orthogonal matrices
2. $\mathbf{e}_1$ is a fixed unit vector orthogonal to $\mathbf{o}$
3. For any direction orthogonal to $\mathbf{o}$, the random rotation treats it symmetrically — there is no preferred "side" of $\mathbf{o}$
4. Therefore $\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle$ is **symmetrically distributed around 0**

Since $\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle$ is symmetric around 0 and $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle > 0$ (concentrated at $\approx 0.8$), the error term in Eq. 12 has expectation 0:

$$
\mathbb{E}\left[\sqrt{1 - \langle \mathbf{o}, \mathbf{q}\rangle^2} \cdot \frac{\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle}\right] = 0
$$

Therefore $\mathbb{E}[\text{estimator}] = \langle \mathbf{o}, \mathbf{q}\rangle + 0 = \langle \mathbf{o}, \mathbf{q}\rangle$. $\blacksquare$

### The Error Bound

**Theorem 3.2 (Error Bound):** With high probability:

$$
\left|\frac{\langle \bar{\mathbf{o}}, \mathbf{q}\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle} - \langle \mathbf{o}, \mathbf{q}\rangle\right| \leq \sqrt{\frac{1 - \langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2}} \cdot \frac{\varepsilon_0}{\sqrt{D - 1}} \tag{Eq. 14}
$$

with failure probability at most $2e^{-c_0 \varepsilon_0^2}$, where $c_0$ is a universal constant.

| Symbol | Meaning | Typical Value |
|:--|:--|:--|
| $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | Quantization quality (from US-010) | $\approx 0.8$ |
| $\sqrt{(1 - \langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2)/\langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2}$ | Error scaling factor | $\sqrt{(1 - 0.64)/0.64} = 0.75$ |
| $\varepsilon_0$ | Confidence parameter | Chosen by user |
| $c_0$ | Universal constant | From sub-Gaussian theory |
| $D - 1$ | Effective dimension for concentration | $127$ for $D = 128$ |

**Asymptotic form:** Since $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ concentrates at a constant ($\approx 0.8$), the error scales as:

$$
\left|\frac{\langle \bar{\mathbf{o}}, \mathbf{q}\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle} - \langle \mathbf{o}, \mathbf{q}\rangle\right| = O\left(\frac{1}{\sqrt{D}}\right) \text{ with high probability} \tag{Eq. 15}
$$

**Why this is optimal:** Alon and Klartag (2017) proved that any method encoding a unit vector with $D$ bits cannot achieve an inner product estimation error better than $O(1/\sqrt{D})$. RaBitQ matches this lower bound, making it **asymptotically optimal**.

### Confidence Interval for Distance Estimation

The error bound gives a computable confidence interval (using stored $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$):

$$
\langle \mathbf{o}, \mathbf{q}\rangle \in \left[\frac{\langle \bar{\mathbf{o}}, \mathbf{q}\rangle}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle} \pm \sqrt{\frac{1 - \langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2}{\langle \bar{\mathbf{o}}, \mathbf{o}\rangle^2}} \cdot \frac{\varepsilon_0}{\sqrt{D-1}}\right] \tag{Eq. 16}
$$

This is **unique to RaBitQ** — no other binary quantization method provides per-vector, theoretically-guaranteed confidence intervals. This enables intelligent re-ranking: only vectors whose confidence interval overlaps with the current top-$k$ threshold need exact distance computation (see US-014).

## Worked Example: Numerical Verification

### Setup

Let $D = 128$, $\mathbf{o}$ and $\mathbf{q}$ be fixed unit vectors with $\langle \mathbf{o}, \mathbf{q}\rangle = 0.6$.

Simulate $10^5$ random orthogonal matrices $P$. For each $P$:
1. Compute $\bar{\mathbf{o}} = P\bar{\mathbf{x}}$ where $\bar{\mathbf{x}}$ has signs matching $P^{-1}\mathbf{o}$
2. Record $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ and $\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle$
3. Compute the estimator $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle / \langle \bar{\mathbf{o}}, \mathbf{o}\rangle$

### Expected Results (from the paper, $D = 128$)

| Quantity | Theoretical | Empirical ($10^5$ samples) |
|:--|:--|:--|
| $\mathbb{E}[\langle \bar{\mathbf{o}}, \mathbf{o}\rangle]$ | $\approx 0.798$ | $\approx 0.798$ |
| Std$[\langle \bar{\mathbf{o}}, \mathbf{o}\rangle]$ | $O(1/\sqrt{D}) \approx 0.044$ | $\approx 0.044$ |
| $\mathbb{E}[\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle]$ | $= 0$ (exact) | $\approx 0.0001$ |
| Std$[\langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle]$ | $O(1/\sqrt{D}) \approx 0.053$ | $\approx 0.053$ |
| $\mathbb{E}[\text{estimator}]$ | $= 0.6$ (exact) | $\approx 0.600$ |

### Concrete Error Calculation

For $D = 128$ and $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle = 0.8$, the error bound with $\varepsilon_0 = 3$ (failure probability $\leq 2e^{-9c_0}$, very small):

$$
\text{error} \leq \sqrt{\frac{1 - 0.64}{0.64}} \cdot \frac{3}{\sqrt{127}} = 0.75 \cdot 0.266 = 0.200
$$

For $D = 768$ (typical LLM embedding dimension):

$$
\text{error} \leq 0.75 \cdot \frac{3}{\sqrt{767}} = 0.75 \cdot 0.108 = 0.081
$$

The error shrinks as $1/\sqrt{D}$ — higher-dimensional vectors are quantized more accurately.

## Comparison with PQ

**Product Quantization (PQ)** uses $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle$ directly as an estimate of $\langle \mathbf{o}, \mathbf{q}\rangle$:

$$
\langle \bar{\mathbf{o}}, \mathbf{q}\rangle = \langle \bar{\mathbf{o}}, \mathbf{o}\rangle \cdot \langle \mathbf{o}, \mathbf{q}\rangle + \langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle \cdot \sqrt{1 - \langle \mathbf{o}, \mathbf{q}\rangle^2}
$$

Taking expectations:

$$
\mathbb{E}[\langle \bar{\mathbf{o}}, \mathbf{q}\rangle] = \underbrace{\mathbb{E}[\langle \bar{\mathbf{o}}, \mathbf{o}\rangle]}_{\approx 0.8} \cdot \langle \mathbf{o}, \mathbf{q}\rangle + 0 = 0.8 \cdot \langle \mathbf{o}, \mathbf{q}\rangle \neq \langle \mathbf{o}, \mathbf{q}\rangle
$$

| Property | RaBitQ: $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle / \langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ | PQ: $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle$ |
|:--|:--|:--|
| **Bias** | Unbiased ($\mathbb{E} = \langle \mathbf{o}, \mathbf{q}\rangle$) | Biased ($\mathbb{E} \approx 0.8 \cdot \langle \mathbf{o}, \mathbf{q}\rangle$) |
| **Error bound** | $O(1/\sqrt{D})$ proven | No theoretical bound |
| **Confidence interval** | Yes (Eq. 16) | No |
| **Extra storage** | $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ per vector (4 bytes) | None |
| **Re-ranking** | Theory-guided (error-bound-based) | Empirical threshold |

The **key insight**: dividing by $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ exactly cancels the systematic shrinkage factor, at the cost of storing one extra scalar per vector.

## Key Takeaways

- **Lemma 3.1** decomposes $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle$ into a signal term ($\propto \langle \mathbf{o}, \mathbf{q}\rangle$) and a noise term ($\propto \langle \bar{\mathbf{o}}, \mathbf{e}_1\rangle$)
- **Unbiasedness** follows from the noise term having symmetric distribution around 0 (due to rotational symmetry of $P$)
- The **error bound** $O(1/\sqrt{D})$ comes from sub-Gaussian concentration of projections on the unit hypersphere
- This bound is **asymptotically optimal** — no $D$-bit code can do better (Alon & Klartag 2017)
- PQ's estimator $\langle \bar{\mathbf{o}}, \mathbf{q}\rangle \approx \langle \mathbf{o}, \mathbf{q}\rangle$ is **biased** by a factor $\approx 0.8$; RaBitQ's division corrects this
- The stored scalar $\langle \bar{\mathbf{o}}, \mathbf{o}\rangle$ serves double duty: unbiasing the estimator AND providing per-vector confidence intervals for intelligent re-ranking

## Sources

- [RaBitQ paper — arXiv:2405.12497](https://arxiv.org/abs/2405.12497) — Section 3.2, Lemma 3.1, Theorem 3.2
- [RaBitQ paper — arXiv HTML](https://arxiv.org/html/2405.12497) — Equations 9–16
- [RaBitQ GitHub repo](https://github.com/gaoj0017/RaBitQ)
- [RaBitQ Library](https://vectordb-ntu.github.io/RaBitQ-Library/)
- [Alon & Klartag (2017) — Optimal lower bound](https://arxiv.org/abs/1611.00310) — Proves $O(1/\sqrt{D})$ is tight for $D$-bit codes
- [Gao & Long (2024) — RaBitQ Technical Report](https://github.com/gaoj0017/RaBitQ/blob/main/technical_report.pdf) — Full proofs
