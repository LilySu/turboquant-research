# MSE vs Inner Product Bias Problem

## Prerequisites Refresher

### Unbiased Estimators

An **estimator** $\hat{\theta}$ is a formula that approximates some true quantity $\theta$ using random data. It is **unbiased** if, on average, it equals the true value:

$$
\mathbb{E}[\hat{\theta}] = \theta
$$

| Symbol | Meaning |
|:--|:--|
| $\hat{\theta}$ | The estimate (a random variable — depends on the data) |
| $\theta$ | The true quantity we want to estimate |
| $\mathbb{E}[\hat{\theta}]$ | Average of the estimate over many trials |

**Why unbiasedness matters:** A biased estimator systematically over- or under-estimates. If $\mathbb{E}[\hat{\theta}] = 0.64 \cdot \theta$, every estimate is shrunk by 36% on average. In attention, this means queries consistently under-attend to relevant keys — degrading generation quality.

### Residuals

The **residual** is the error left after an approximation:

$$
\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbf{r}$ | Residual vector (the leftover error) |
| $\mathbf{x}$ | Original vector |
| $\hat{\mathbf{x}}$ | Approximation (e.g., from MSE quantization) |

By definition, $\mathbf{x} = \hat{\mathbf{x}} + \mathbf{r}$, so any inner product decomposes as $\langle \mathbf{y}, \mathbf{x}\rangle = \langle \mathbf{y}, \hat{\mathbf{x}}\rangle + \langle \mathbf{y}, \mathbf{r}\rangle$.

## Main Content

### The Bias Problem at 1-Bit

Consider the simplest case: 1-bit quantization ($b = 1$) of a unit vector $\mathbf{x} \in \mathbb{S}^{d-1}$ after random rotation.

From US-003, the optimal 1-bit MSE codebook for $\mathcal{N}(0, 1/d)$ coordinates is $\{-\sqrt{2/(\pi d)}, +\sqrt{2/(\pi d)}\}$. The quantizer maps each rotated coordinate $y_j$ to:

$$
Q_{\text{mse}}(y_j) = \sqrt{\frac{2}{\pi d}} \cdot \text{sign}(y_j)
$$

| Symbol | Meaning |
|:--|:--|
| $y_j$ | $j$-th coordinate of the rotated vector $\Pi\mathbf{x}$ |
| $\text{sign}(y_j)$ | $+1$ if $y_j > 0$, $-1$ if $y_j < 0$ |
| $\sqrt{2/(\pi d)}$ | Optimal 1-bit centroid for $\mathcal{N}(0, 1/d)$ |

Now compute the expected inner product with any query $\mathbf{y}$:

$$
\mathbb{E}\!\left[\langle \mathbf{y}, Q_{\text{mse}}^{-1}(Q_{\text{mse}}(\mathbf{x}))\rangle\right] = \frac{2}{\pi} \langle \mathbf{y}, \mathbf{x}\rangle
$$

| Symbol | Meaning |
|:--|:--|
| $Q_{\text{mse}}^{-1}(Q_{\text{mse}}(\mathbf{x}))$ | Quantize then dequantize (reconstruct) |
| $2/\pi \approx 0.637$ | The **multiplicative bias** — inner products are systematically shrunk |
| $\langle \mathbf{y}, \mathbf{x}\rangle$ | True inner product |

### Where Does $2/\pi$ Come From?

The bias arises from the same identity used in the QJL proof (US-004). For a Gaussian random variable $V \sim \mathcal{N}(0, \sigma^2)$:

$$
\mathbb{E}[V \cdot \text{sign}(V)] = \sigma \sqrt{\frac{2}{\pi}}
$$

| Symbol | Meaning |
|:--|:--|
| $V$ | A Gaussian coordinate (e.g., $\langle \mathbf{s}, \mathbf{x}\rangle$) |
| $\text{sign}(V)$ | The quantized version (just the sign) |
| $\sigma\sqrt{2/\pi}$ | The expected product — always less than $\mathbb{E}[V^2] = \sigma^2$ |

**Geometric intuition:** When you replace a Gaussian value with its sign ($\pm 1$, rescaled), you lose information about magnitude. Large values get clamped to the same centroid as small ones. This "regression to the mean" within each partition creates a systematic shrinkage in the inner product. The factor $2/\pi$ is the ratio of $\mathbb{E}[|V|]$ to $\sigma$ for a Gaussian — a fundamental property of the half-normal distribution.

**Why not just rescale by $\pi/2$?** You could multiply the dequantized vector by $\pi/2$ to remove the bias. But this amplifies the variance by $(\pi/2)^2 \approx 2.47$, making the estimator much noisier. TurboQuant's two-stage approach achieves unbiasedness without this variance penalty.

### Bias at Higher Bit-Widths

As $b$ increases, the quantizer captures more magnitude information, and the bias shrinks. The bias factor approaches 1 exponentially:

| $b$ (bits) | Bias factor $\alpha_b$ | Inner product estimate | $D_{\text{prod}}$ (per dimension) |
|:-:|:-:|:--|:-:|
| 1 | $2/\pi \approx 0.637$ | $0.637 \langle \mathbf{y}, \mathbf{x}\rangle$ | $\sim 1.57/d$ |
| 2 | $\approx 0.93$ | $0.93 \langle \mathbf{y}, \mathbf{x}\rangle$ | $\sim 0.56/d$ |
| 3 | $\approx 0.98$ | $0.98 \langle \mathbf{y}, \mathbf{x}\rangle$ | $\sim 0.18/d$ |
| 4 | $\approx 0.997$ | Nearly unbiased | $\sim 0.047/d$ |

At $b \geq 4$, the bias is negligible. But at the low bit-widths critical for KV cache compression ($b = 2$ or $3$), the bias is significant enough to degrade attention quality.

### The Two-Stage Solution: MSE($b-1$) + QJL(1)

TurboQuant$_{\text{prod}}$ (Algorithm 2) eliminates the bias by splitting the $b$-bit budget:

- **$(b-1)$ bits**: MSE-optimal quantization (biased but low-distortion)
- **1 bit**: QJL on the residual (unbiased, corrects the bias)

#### Algorithm 2 Walkthrough

**Quantization** (Quant$_{\text{prod}}$):

$$
\begin{aligned}
\text{idx} &\leftarrow Q_{\text{mse}}(\mathbf{x}) & \text{(quantize with } b-1 \text{ bits)} \\
\mathbf{r} &\leftarrow \mathbf{x} - Q_{\text{mse}}^{-1}(\text{idx}) & \text{(compute residual)} \\
\text{qjl} &\leftarrow \text{sign}(S \cdot \mathbf{r}) & \text{(QJL on residual)} \\
\gamma &\leftarrow \|\mathbf{r}\|_2 & \text{(store residual norm)}
\end{aligned}
$$

| Symbol | Meaning |
|:--|:--|
| idx | Quantization indices ($b-1$ bits per coordinate) |
| $\mathbf{r}$ | Residual: what the MSE quantizer missed |
| qjl | Sign bits of the projected residual (1 bit per coordinate) |
| $\gamma = \|\mathbf{r}\|_2$ | Residual norm (stored in full precision) |

**Dequantization** (DeQuant$_{\text{prod}}$):

$$
\hat{\mathbf{x}} = \underbrace{Q_{\text{mse}}^{-1}(\text{idx})}_{\hat{\mathbf{x}}_{\text{mse}}} + \underbrace{\gamma \cdot \frac{\sqrt{\pi/2}}{d} S^T \cdot \text{qjl}}_{\hat{\mathbf{x}}_{\text{qjl}}}
$$

| Symbol | Meaning |
|:--|:--|
| $\hat{\mathbf{x}}_{\text{mse}}$ | MSE reconstruction (from $b-1$ bits) |
| $\hat{\mathbf{x}}_{\text{qjl}}$ | QJL reconstruction of the residual (from 1 bit) |
| $\gamma$ | Residual norm — scales the QJL reconstruction |
| $\sqrt{\pi/2}/d$ | QJL dequantization constant (see US-004) |

### Proof: Why the Combination Is Unbiased

**Goal:** Show $\mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}\rangle] = \langle \mathbf{y}, \mathbf{x}\rangle$.

**Step 1: Decompose.** Since $\hat{\mathbf{x}} = \hat{\mathbf{x}}_{\text{mse}} + \hat{\mathbf{x}}_{\text{qjl}}$:

$$
\langle \mathbf{y}, \hat{\mathbf{x}}\rangle = \langle \mathbf{y}, \hat{\mathbf{x}}_{\text{mse}}\rangle + \langle \mathbf{y}, \hat{\mathbf{x}}_{\text{qjl}}\rangle
$$

**Step 2: Apply QJL unbiasedness (Lemma 4 from US-004).** Conditioned on $\hat{\mathbf{x}}_{\text{mse}}$ (which determines $\mathbf{r}$), the QJL stage gives:

$$
\mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}_{\text{qjl}}\rangle \mid \hat{\mathbf{x}}_{\text{mse}}] = \langle \mathbf{y}, \mathbf{r}\rangle
$$

| Symbol | Meaning |
|:--|:--|
| $\mathbb{E}[\cdot \mid \hat{\mathbf{x}}_{\text{mse}}]$ | Expectation over QJL randomness, given the MSE stage is fixed |
| $\langle \mathbf{y}, \mathbf{r}\rangle$ | True inner product of query with residual |

**Step 3: Combine.** Taking the total expectation:

$$
\mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}\rangle] = \mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}_{\text{mse}}\rangle] + \mathbb{E}[\langle \mathbf{y}, \mathbf{r}\rangle] = \mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}_{\text{mse}} + \mathbf{r}\rangle] = \langle \mathbf{y}, \mathbf{x}\rangle
$$

The last equality uses $\hat{\mathbf{x}}_{\text{mse}} + \mathbf{r} = \mathbf{x}$ by definition. $\blacksquare$

**Key insight:** The proof doesn't require the MSE stage to be unbiased! It only needs the QJL stage to be unbiased *for the residual*. The MSE stage can be as biased as it wants — the QJL "mops up" whatever error remains, and it does so without bias.

## Worked Examples

### Example: 2-bit TurboQuant$_{\text{prod}}$ vs Pure MSE

At $b = 2$ total bits, TurboQuant$_{\text{prod}}$ uses 1 bit for MSE + 1 bit for QJL.

| Method | Bias factor | Inner product distortion (per dim) |
|:--|:-:|:-:|
| Pure MSE (2-bit) | $\alpha_2 \approx 0.93$ | $\sim 0.56/d$ |
| TurboQuant$_{\text{prod}}$ (1+1 bit) | **1.0 (unbiased)** | $\sim 1.57/d$ |
| Rescaled MSE ($\times \pi/2$, 2-bit) | 1.0 | $\sim 1.38/d$ (higher variance) |

TurboQuant$_{\text{prod}}$ trades slightly higher distortion for **zero bias** — critical for attention accuracy.

### Example: Why Bias Matters in Attention

With $d = 128$, $b = 2$, and a true attention score $\langle \mathbf{q}, \mathbf{k}\rangle = 3.0$:

- **Pure MSE**: Expected score = $0.93 \times 3.0 = 2.79$ (7% underestimate on every token)
- **TurboQuant$_{\text{prod}}$**: Expected score = $3.0$ (correct on average)

Over thousands of tokens in a long context, the systematic 7% shrinkage causes the model to under-attend to relevant keys, degrading needle-in-haystack retrieval and coherence.

## Connection to TurboQuant

1. **The core design choice**: TurboQuant splits its bit budget into $(b-1)$ MSE bits + 1 QJL bit, sacrificing a small amount of MSE optimality to achieve **unbiased** inner product estimation.

2. **Why not just more MSE bits?** At $b = 4$, pure MSE is nearly unbiased. But at the critical $b = 2$ and $b = 3$ bit-widths needed for aggressive KV cache compression, the bias is severe enough to matter.

3. **The residual is small**: After $(b-1)$ bits of MSE quantization, $\|\mathbf{r}\|^2 \approx \mathcal{C}(f_X, b-1)$ which is already small. QJL's variance is proportional to $\|\mathbf{r}\|^2$, so the noise added by the 1-bit QJL stage is modest.

4. **Theorem 2 bound**: The combined distortion is $D_{\text{prod}} \leq \frac{\sqrt{3}\pi^2 \|\mathbf{y}\|^2}{d} \cdot 4^{-b}$, near-optimal by the information-theoretic lower bound.

## Key Takeaways

- MSE-optimal quantizers are **biased** for inner products: they systematically shrink $\langle \mathbf{y}, \mathbf{x}\rangle$ by a factor $\alpha_b < 1$
- At 1-bit, the bias is $2/\pi \approx 0.637$ — from the half-normal distribution's $\mathbb{E}[|V|]/\sigma$ ratio
- Simple rescaling fixes the bias but amplifies variance by $\sim 2.5\times$
- TurboQuant's two-stage solution: MSE($b-1$ bits) + QJL(1 bit on residual) is **unbiased** without variance penalty
- The proof is elegant: QJL is unbiased for the residual, and $\hat{\mathbf{x}}_{\text{mse}} + \mathbf{r} = \mathbf{x}$, so everything cancels
- Bias matters most at low bit-widths ($b = 2, 3$) — exactly where KV cache compression operates

## Sources

- [TurboQuant paper — arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [TurboQuant HTML — arXiv](https://arxiv.org/html/2504.19874v1)
- [TurboQuant — OpenReview (ICLR 2026)](https://openreview.net/forum?id=tO3ASKZlok)
- [TurboQuant — OpenReview PDF](https://openreview.net/pdf?id=tO3ASKZlok)
- [TurboQuant — HuggingFace papers](https://huggingface.co/papers/2504.19874)
- [TurboQuant explained simply — Minakeep](https://minakeep.mina.asia/notes/turboquant-explained-simply-the-easiest-way-to-understand-the-paper-after-learning-inner-product-and-quantization)
- [TurboQuant: What 3-Bit KV Caches Actually Mean — The ML Surgeon](https://themlsurgeon.substack.com/p/turboquant-what-3-bit-kv-caches-actually)
- [QJL paper — arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- [High-Rate Quantized Matrix Multiplication — arXiv:2601.17187](https://arxiv.org/pdf/2601.17187)
- [Ordentlich & Polyanskiy (2025), HTML version](https://arxiv.org/html/2601.17187)
- [Norm-Explicit Quantization — arXiv:1911.04654](https://arxiv.org/abs/1911.04654)
- [MIT 6.450: Quantization chapter](https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/926689aaa62a0315473fa9b982de1b07_book_3.pdf)
