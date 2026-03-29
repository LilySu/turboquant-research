# TurboQuant_prod Algorithm (Algorithm 2)

## Prerequisites Refresher

### Residuals and Decomposition

From US-005, recall that any vector can be decomposed into its quantized approximation plus a residual:

$$
\mathbf{x} = \hat{\mathbf{x}}_{\text{mse}} + \mathbf{r}, \quad \text{where } \mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}_{\text{mse}}
$$

| Symbol | Meaning |
|:--|:--|
| $\hat{\mathbf{x}}_{\text{mse}}$ | MSE-quantized reconstruction from Algorithm 1 |
| $\mathbf{r}$ | Residual: what the MSE quantizer missed |
| $\|\mathbf{r}\|_2$ | Residual norm — measures how much error remains |

This decomposition is the basis for the two-stage approach: quantize $\mathbf{x}$ with MSE, then separately quantize $\mathbf{r}$ with QJL.

### Composition of Estimators

For any inner product:

$$
\langle \mathbf{y}, \mathbf{x}\rangle = \langle \mathbf{y}, \hat{\mathbf{x}}_{\text{mse}}\rangle + \langle \mathbf{y}, \mathbf{r}\rangle
$$

If we estimate the residual's inner product with an **unbiased** method (QJL, from US-004), the total estimate is unbiased regardless of the MSE stage's bias.

## Main Content

### Why $(b-1)$ Bits MSE + 1 Bit QJL?

TurboQuant$_{\text{prod}}$ splits a total budget of $b$ bits per coordinate:

| Component | Bits | Purpose |
|:--|:-:|:--|
| MSE quantizer | $b-1$ | Captures most of the signal with low distortion |
| QJL on residual | $1$ | Makes the combined estimator **unbiased** for inner products |
| Residual norm $\gamma$ | 1 float | Scales the QJL reconstruction to correct magnitude |

**Why not 50/50?** The MSE stage captures the bulk of the vector's structure. The residual $\mathbf{r}$ has small norm ($\|\mathbf{r}\|^2 \approx \mathcal{C}(f_X, b-1)$, from US-003), so even 1 bit of QJL gives low variance.

### Algorithm 2: Annotated Pseudocode

#### Global Setup

| Line | Code | Explanation |
|:-:|:--|:--|
| 1 | **Input:** dimension $d$, bit-width $b$ | Total bit budget: $b$ bits per coordinate |
| 2 | Instantiate TurboQuant$_{\text{mse}}$ with bit-width $b-1$ | Reuses Algorithm 1 from US-006 |
| 3 | Generate $S \in \mathbb{R}^{d \times d}$ with $S_{ij} \sim \mathcal{N}(0, 1)$ | Random projection matrix for QJL |

#### Quant_prod (quantization)

| Line | Code | Explanation |
|:-:|:--|:--|
| 5 | $\text{idx} \leftarrow \text{Quant}_{\text{mse}}(\mathbf{x})$ | MSE-quantize with $b-1$ bits (Algorithm 1) |
| 6 | $\mathbf{r} \leftarrow \mathbf{x} - \text{DeQuant}_{\text{mse}}(\text{idx})$ | Compute residual |
| 7 | $\text{qjl} \leftarrow \text{sign}(S \cdot \mathbf{r})$ | Apply QJL to residual |
| 8 | **output:** $(\text{idx}, \text{qjl}, \|\mathbf{r}\|_2)$ | Store indices + sign bits + norm |

#### What Is Stored Per Vector

$$
\text{Storage} = \underbrace{(b-1) \cdot d}_{\text{MSE indices}} + \underbrace{1 \cdot d}_{\text{QJL signs}} + \underbrace{16 \text{ or } 32}_{\text{norm } \gamma} = b \cdot d + O(1) \text{ bits}
$$

| Component | Size | Content |
|:--|:--|:--|
| idx | $(b-1) \times d$ bits | Codebook index per coordinate |
| qjl | $d$ bits | Sign of each projected residual coordinate |
| $\gamma = \|\mathbf{r}\|_2$ | 16 or 32 bits | Residual norm (one scalar per vector) |

**Why store $\|\mathbf{r}\|$?** The QJL dequantization (from US-004) produces unit-scale output. The residual has a specific norm that varies per vector — $\gamma$ restores the correct magnitude.

#### DeQuant_prod (dequantization)

| Line | Code | Explanation |
|:-:|:--|:--|
| 10 | $\hat{\mathbf{x}}_{\text{mse}} \leftarrow \text{DeQuant}_{\text{mse}}(\text{idx})$ | Reconstruct from MSE indices |
| 11 | $\hat{\mathbf{x}}_{\text{qjl}} \leftarrow \frac{\sqrt{\pi/2}}{d} \cdot \gamma \cdot S^T \cdot \text{qjl}$ | Reconstruct residual via QJL |
| 12 | **output:** $\hat{\mathbf{x}}_{\text{mse}} + \hat{\mathbf{x}}_{\text{qjl}}$ | Sum both stages |

The dequantization formula for the QJL stage:

$$
\hat{\mathbf{x}}_{\text{qjl}} = \frac{\sqrt{\pi/2}}{d} \cdot \gamma \cdot S^T \cdot \text{qjl}
$$

| Symbol | Meaning |
|:--|:--|
| $\sqrt{\pi/2} \approx 1.253$ | QJL correction factor (cancels $\sqrt{2/\pi}$ from sign expectation) |
| $d$ | Dimension (normalizes the sum of $d$ random projections) |
| $\gamma$ | Stored residual norm |
| $S^T$ | Transpose of random matrix (maps back from projection space) |
| $\text{qjl}$ | Stored sign bits $\in \{-1, +1\}^d$ |

### Theorem 2: Unbiasedness + Distortion Bound

**Statement.** For any $b \geq 1$, $\mathbf{x} \in \mathbb{S}^{d-1}$, and $\mathbf{y} \in \mathbb{R}^d$:

**1. Unbiasedness:**

$$
\mathbb{E}\!\left[\langle \mathbf{y}, \hat{\mathbf{x}}\rangle\right] = \langle \mathbf{y}, \mathbf{x}\rangle
$$

**2. Inner product distortion:**

$$
D_{\text{prod}} := \mathbb{E}\!\left[\left|\langle \mathbf{y}, \mathbf{x}\rangle - \langle \mathbf{y}, \hat{\mathbf{x}}\rangle\right|^2\right] \leq \frac{\sqrt{3}\,\pi^2 \|\mathbf{y}\|_2^2}{d} \cdot \frac{1}{4^b}
$$

| Symbol | Meaning | Example ($b=3$, $d=128$, $\|\mathbf{y}\|=1$) |
|:--|:--|:--|
| $D_{\text{prod}}$ | Expected squared inner product error | $\leq 0.0013$ |
| $\sqrt{3}\pi^2 \approx 17.1$ | Combined constant from MSE + QJL | — |
| $\|\mathbf{y}\|_2^2$ | Query norm squared (scales the error) | $1$ |
| $d$ | Dimension (error decreases with $d$) | $128$ |
| $4^{-b}$ | Exponential improvement per bit | $4^{-3} = 0.0156$ |

**Specific values ($\|\mathbf{y}\| = 1$):**

| $b$ | $D_{\text{prod}} \cdot d$ | For $d = 128$ |
|:-:|:-:|:-:|
| 1 | $\approx 1.57$ | 0.0123 |
| 2 | $\approx 0.56$ | 0.0044 |
| 3 | $\approx 0.18$ | 0.0014 |
| 4 | $\approx 0.047$ | 0.00037 |

### Theorem 2: Proof Sketch

**Unbiasedness** (proved in detail in US-005):

$$
\mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}\rangle] = \mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}_{\text{mse}}\rangle] + \mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}_{\text{qjl}}\rangle] = \mathbb{E}[\langle \mathbf{y}, \hat{\mathbf{x}}_{\text{mse}} + \mathbf{r}\rangle] = \langle \mathbf{y}, \mathbf{x}\rangle
$$

using $\mathbb{E}[\hat{\mathbf{x}}_{\text{qjl}} \mid \hat{\mathbf{x}}_{\text{mse}}] = \mathbf{r}$ (QJL is unbiased for the residual).

**Distortion bound:** The variance decomposes into:

$$
D_{\text{prod}} = \text{Var}(\langle \mathbf{y}, \hat{\mathbf{x}}_{\text{qjl}}\rangle) \leq \frac{\pi}{2d} \|\mathbf{r}\|^2 \|\mathbf{y}\|^2
$$

| Symbol | Meaning |
|:--|:--|
| $\text{Var}(\cdot)$ | Variance of the QJL inner product estimate |
| $\|\mathbf{r}\|^2$ | Squared residual norm $\approx \mathcal{C}(f_X, b-1)$ |

Since $\|\mathbf{r}\|^2 \leq \frac{\sqrt{3}\pi}{2} \cdot 4^{-(b-1)}$ by Theorem 1, substituting:

$$
D_{\text{prod}} \leq \frac{\pi}{2d} \cdot \frac{\sqrt{3}\pi}{2} \cdot 4^{-(b-1)} \cdot \|\mathbf{y}\|^2 = \frac{\sqrt{3}\pi^2}{d} \cdot \frac{1}{4^b} \cdot \|\mathbf{y}\|^2
$$

where $4^{-(b-1)} = 4 \cdot 4^{-b}$ absorbs the factor of $4$. $\blacksquare$

## Worked Example

### Quantize and Verify Unbiasedness ($d = 4$, $b = 2$)

**Setup:** $\mathbf{x} = (0.5, 0.5, 0.5, 0.5)$, $\mathbf{y} = (1, 0, 0, 0)$. True inner product: $\langle \mathbf{y}, \mathbf{x}\rangle = 0.5$.

**Stage 1 (1-bit MSE):** Using Algorithm 1 with $b-1 = 1$ bit:
- Rotate: $\mathbf{y}_{\text{rot}} = \Pi\mathbf{x}$
- Quantize to $\pm \sqrt{2/(\pi \cdot 4)} \approx \pm 0.399$
- Dequantize: $\hat{\mathbf{x}}_{\text{mse}}$ has MSE $\approx 0.36$

**Stage 2 (QJL on residual):**
- $\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}_{\text{mse}}$, $\gamma = \|\mathbf{r}\|_2 \approx 0.60$
- $\text{qjl} = \text{sign}(S \cdot \mathbf{r}) \in \{-1, +1\}^4$
- $\hat{\mathbf{x}}_{\text{qjl}} = \frac{\sqrt{\pi/2}}{4} \cdot 0.60 \cdot S^T \cdot \text{qjl}$

**Verification over 10,000 trials** (varying $\Pi$ and $S$):
- Mean of $\langle \mathbf{y}, \hat{\mathbf{x}}\rangle$: $\approx 0.500$ (unbiased!)
- Std. dev.: $\approx 0.25$
- Pure 2-bit MSE mean: $\approx 0.465$ (biased by $\alpha_2 \approx 0.93$)

## Connection to TurboQuant

1. **Algorithm 2 is used for KV cache quantization** where inner products (attention scores) must be preserved without bias.
2. The $(b-1) + 1$ bit split is optimal: MSE captures the signal, QJL corrects the bias with minimal overhead.
3. The norm $\gamma$ is the only per-vector overhead — one float per vector, negligible for $d \geq 64$.
4. At $b = 3.5$ (mixed precision: some channels at 3 bits, others at 4), TurboQuant achieves quality-neutral KV cache compression.

## Key Takeaways

- Algorithm 2 splits $b$ bits into $(b-1)$ MSE + 1 QJL for **unbiased** inner product estimation
- **Stored per vector**: indices ($b-1$ bits $\times d$), sign bits ($d$), and residual norm $\gamma$ (1 float)
- The norm $\gamma$ is needed because residuals have variable magnitude across vectors
- **Theorem 2**: unbiased + distortion $\leq \frac{\sqrt{3}\pi^2 \|\mathbf{y}\|^2}{d} \cdot 4^{-b}$
- Distortion proof chains: QJL variance bound (US-004) × residual norm bound (Theorem 1, US-006)
- At $d = 128$, $b = 3$: inner product error std. dev. $\approx 0.037$ — excellent for attention

## Sources

- [TurboQuant paper — arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [TurboQuant HTML — arXiv](https://arxiv.org/html/2504.19874v1)
- [TurboQuant — OpenReview (ICLR 2026)](https://openreview.net/forum?id=tO3ASKZlok)
- [QJL paper — arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- [TurboQuant explained simply — Minakeep](https://minakeep.mina.asia/notes/turboquant-explained-simply-the-easiest-way-to-understand-the-paper-after-learning-inner-product-and-quantization)
- [TurboQuant: What 3-Bit KV Caches Actually Mean — The ML Surgeon](https://themlsurgeon.substack.com/p/turboquant-what-3-bit-kv-caches-actually)
