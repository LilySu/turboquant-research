# Shannon's Source Coding Theory

## Prerequisites Refresher

### Probability Distributions and Expectation

A **random variable** $X$ takes values according to a probability distribution. For a discrete $X$ with possible values $\{x_1, x_2, \ldots\}$ and probabilities $p(x_i)$, the **expectation** (or mean) is:

$$
\mathbb{E}[X] = \sum_i x_i \, p(x_i)
$$

For a continuous $X$ with probability density function (pdf) $f_X(x)$, the expectation is:

$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x \, f_X(x) \, dx
$$

Expectation is linear: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$.

### Information Content

The **information content** (or surprisal) of an event with probability $p$ is:

$$
I(x) = -\log_2 p(x) \quad \text{(in bits)}
$$

**Intuition**: A coin landing heads ($p = 0.5$) carries $-\log_2(0.5) = 1$ bit of information. An event with $p = 0.01$ carries $\approx 6.64$ bits — rarer events are more "surprising" and carry more information.

### Entropy (Discrete)

**Entropy** is the *expected* information content — the average surprise per observation:

$$
H(X) = -\sum_x p(x) \log_2 p(x) = \mathbb{E}[-\log_2 p(X)]
$$

**Key examples:**
- Fair coin ($p = 0.5$): $H = -2 \cdot 0.5 \log_2 0.5 = 1$ bit (maximum uncertainty)
- Biased coin ($p = 0.9$): $H \approx 0.47$ bits (less uncertain, less info per flip)
- Certain outcome ($p = 1$): $H = 0$ bits (no uncertainty at all)

Entropy is bounded: $0 \leq H(X) \leq \log_2 |\mathcal{X}|$, where $|\mathcal{X}|$ is the number of possible outcomes. The upper bound is achieved when all outcomes are equally likely (uniform distribution).

## Main Content

### Differential Entropy $h(X)$

When $X$ is continuous, we extend entropy to **differential entropy**:

$$
h(X) = -\int_{-\infty}^{\infty} f_X(x) \log_2 f_X(x) \, dx
$$

Unlike discrete entropy, differential entropy can be **negative**. For example, a uniform distribution on $[0, 1/2]$ has $h(X) = \log_2(1/2) = -1$ bit.

**Important properties:**
- Translation invariant: $h(X + c) = h(X)$
- Scaling: $h(aX) = h(X) + \log_2 |a|$
- **Gaussian maximizes differential entropy** for a given variance $\sigma^2$:

$$
h(X) \leq \frac{1}{2} \log_2(2\pi e \sigma^2)
$$

with equality if and only if $X \sim \mathcal{N}(\mu, \sigma^2)$.

**Example:** For $X \sim \mathcal{N}(0, 1)$:

$$
h(X) = \frac{1}{2} \log_2(2\pi e) \approx 2.047 \text{ bits}
$$

### Mutual Information $I(X; Y)$

**Mutual information** measures how much knowing $Y$ reduces your uncertainty about $X$:

$$
I(X; Y) = h(X) - h(X|Y) = h(Y) - h(Y|X)
$$

Equivalently, for continuous $(X, Y)$ with joint density $p(x, y)$:

$$
I(X; Y) = \int\!\!\int p(x, y) \log_2 \frac{p(x, y)}{f_X(x) \, f_Y(y)} \, dx \, dy
$$

**Key properties:**
- $I(X; Y) \geq 0$, with equality iff $X \perp\!\!\!\perp Y$ (independent)
- $I(X; Y) = I(Y; X)$ (symmetric)
- $I(X; Y) = h(X) + h(Y) - h(X, Y)$

**Example:** Gaussian channel $Y = X + Z$ where $X \sim \mathcal{N}(0, S)$, $Z \sim \mathcal{N}(0, 1)$, independent:

$$
I(X; Y) = \frac{1}{2} \log_2(1 + S)
$$

This is Shannon's famous channel capacity formula — the maximum rate at which information can be reliably transmitted.

### The Distortion-Rate Function $D(R)$

#### Building Intuition

Suppose you have a source emitting real-valued signals and you want to compress them. **Lossless** compression requires at least $H(X)$ bits per symbol (Shannon's source coding theorem). But what if you're willing to tolerate some error?

This is **lossy compression**. You accept a distortion $d(x, \hat{x})$ between the original $x$ and its reconstruction $\hat{x}$. The natural question is:

> *What is the minimum distortion achievable at rate $R$ bits per symbol?*

The **rate-distortion function** $R(D)$ answers the dual: the minimum rate needed to achieve distortion $\leq D$. Its inverse, the **distortion-rate function** $D(R)$, directly answers our question.

#### Formal Definition

For a source $\mathbf{x} \in \mathbb{R}^d$ with distribution $p_X$ and MSE distortion:

$$
D(p_X, B) := \inf \left\{ \mathbb{E}\!\left[\|\mathbf{x} - \mathbf{y}\|_2^2\right] : I(\mathbf{x}; \mathbf{y}) \leq B \right\}
$$

The infimum is over all joint distributions of $(\mathbf{x}, \mathbf{y})$ satisfying the mutual information constraint. Here $B$ is the total bit budget across all $d$ coordinates.

**Shannon's Lossy Source Coding Theorem** says this bound is tight:
- **Achievability**: For any rate $R > R(D)$, there exists a code achieving distortion $\leq D$.
- **Converse**: No code with rate $R < R(D)$ can achieve distortion $\leq D$.

#### Gaussian Source Closed Form

For $X \sim \mathcal{N}(0, \sigma^2)$ with MSE distortion:

$$
R(D) = \frac{1}{2} \log_2 \frac{\sigma^2}{D}, \quad D(R) = \sigma^2 \cdot 2^{-2R}
$$

This gives the **"6 dB per bit" rule**: each additional bit of rate halves the distortion (a 6 dB improvement in signal-to-noise ratio).

### The Shannon Lower Bound — Lemma 2

The **Shannon Lower Bound (SLB)** extends rate-distortion theory to *arbitrary* source distributions. It is the key tool used in TurboQuant's near-optimality proof.

#### Statement (Lemma 2 from the TurboQuant paper)

For a random vector $\mathbf{x} \in \mathbb{R}^d$ with distribution $p_X$ and finite differential entropy $h(\mathbf{x})$, the MSE distortion-rate function satisfies:

$$
D(p_X, B) \geq \frac{d}{2\pi e} \cdot 2^{\frac{2}{d}(h(\mathbf{x}) - B)}
$$

for bit budget $B \geq 0$.

#### Step-by-Step Derivation

**Step 1: Relate distortion to conditional entropy.** For any joint distribution of $(\mathbf{x}, \mathbf{y})$ achieving distortion $D$, the reconstruction error $\mathbf{z} = \mathbf{x} - \mathbf{y}$ satisfies $\mathbb{E}[\|\mathbf{z}\|^2] = D$. The covariance of $\mathbf{z}$ has trace equal to $D$.

**Step 2: Entropy power inequality.** The Gaussian maximizes entropy for a given covariance, so:

$$
h(\mathbf{x} | \mathbf{y}) \leq h(\mathbf{z}) \leq \frac{d}{2} \log_2\!\left(\frac{2\pi e \cdot D}{d}\right)
$$

The right side is the entropy of a $d$-dimensional Gaussian with per-coordinate variance $D/d$.

**Step 3: Apply mutual information constraint.** Since $I(\mathbf{x}; \mathbf{y}) = h(\mathbf{x}) - h(\mathbf{x}|\mathbf{y}) \leq B$, we get:

$$
h(\mathbf{x}|\mathbf{y}) \geq h(\mathbf{x}) - B
$$

**Step 4: Combine.** Substituting Step 3 into Step 2:

$$
h(\mathbf{x}) - B \leq \frac{d}{2} \log_2\!\left(\frac{2\pi e \cdot D}{d}\right)
$$

Solving for $D$:

$$
D \geq \frac{d}{2\pi e} \cdot 2^{\frac{2}{d}(h(\mathbf{x}) - B)}
$$

This is exactly the SLB. The bound is **tight** for Gaussian sources, and **asymptotically tight** (at high bit rates) for many non-Gaussian sources.

## Worked Examples

### Example 1: Gaussian Source at 2 bits

Let $X \sim \mathcal{N}(0, 1)$ with $R = 2$ bits per sample.

$$
D(R) = \sigma^2 \cdot 2^{-2R} = 1 \cdot 2^{-4} = 0.0625
$$

With 2 bits per sample, the best possible MSE is $0.0625$. A uniform 4-level quantizer achieves $D \approx 0.09$ — not far from optimal but not at the bound either.

### Example 2: SLB Applied to TurboQuant

TurboQuant operates on vectors $\mathbf{x}$ uniformly distributed on the unit sphere $\mathbb{S}^{d-1}$, using $b$ bits per coordinate ($B = bd$ total bits).

By Theorem 3 of the paper (which uses Yao's minimax principle to reduce to this case), the lower bound simplifies to:

$$
D_{\text{mse}} \geq \frac{1}{4^b}
$$

TurboQuant (Theorem 1) achieves:

$$
D_{\text{mse}} \leq \frac{\sqrt{3}\,\pi}{2} \cdot \frac{1}{4^b}
$$

The approximation ratio is $\frac{\sqrt{3}\,\pi}{2} \approx 2.72$, constant across all bit-widths:

| $b$ (bits) | Lower Bound $\frac{1}{4^b}$ | TurboQuant $\frac{\sqrt{3}\pi}{2 \cdot 4^b}$ | Ratio |
|:-:|:-:|:-:|:-:|
| 1 | 0.2500 | 0.6802 | 2.72 |
| 2 | 0.0625 | 0.1700 | 2.72 |
| 3 | 0.01563 | 0.04251 | 2.72 |
| 4 | 0.003906 | 0.01063 | 2.72 |

## Connection to TurboQuant

Shannon's theory provides the **impossibility results** that make TurboQuant's guarantees meaningful:

1. **Differential entropy** $h(\mathbf{x})$ quantifies how "compressible" the rotated KV cache vectors are. After random rotation, coordinates follow a distribution converging to $\mathcal{N}(0, 1/d)$, whose entropy is analytically known.

2. **Mutual information** $I(\mathbf{x}; \mathbf{y}) \leq B$ is the fundamental constraint: a $b$-bit-per-coordinate quantizer can carry at most $bd$ bits of information about the input.

3. The **distortion-rate function** $D(R)$ sets the floor. No quantizer — no matter how cleverly designed — can beat it.

4. The **Shannon Lower Bound** (Lemma 2) is applied via **Yao's minimax principle** to prove Theorem 3: any quantizer on $\mathbb{S}^{d-1}$ must incur $D_{\text{mse}} \geq 1/4^b$.

5. TurboQuant's scalar quantization after random rotation achieves $D_{\text{mse}} \leq \frac{\sqrt{3}\pi}{2} \cdot \frac{1}{4^b}$ — within a factor of $\approx 2.72$ of optimal. This near-optimality is remarkable for an algorithm with $O(d \log d)$ runtime.

## Key Takeaways

- **Entropy** measures average surprise; **differential entropy** extends this to continuous variables but can be negative
- **Mutual information** $I(X;Y)$ measures shared information and constrains what any quantizer can preserve
- The **distortion-rate function** $D(R)$ is the information-theoretic floor on compression error at rate $R$
- The **Shannon Lower Bound** (Lemma 2) generalizes this floor to arbitrary distributions via a clean formula involving $h(\mathbf{x})$
- TurboQuant exploits the SLB to prove it is within $\approx 2.72\times$ of the best *any* algorithm can achieve
- The Gaussian is the "hardest" source to compress at a given variance — this is why TurboQuant's rotation (which induces near-Gaussian coordinates) is a natural design choice

## Sources

- [Shannon (1948), "A Mathematical Theory of Communication" — Harvard hosted PDF](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
- [Rate-Distortion Theory — Wikipedia](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory)
- [Stanford EE398A: Rate Distortion Theory handout](https://web.stanford.edu/class/ee398a/handouts/lectures/04-RateDistortionTheory.pdf)
- [Yale Lecture 13: Shannon Lower Bound, Fano's method](http://www.stat.yale.edu/~yw562/teaching/598/lec13.pdf)
- [Mutual Information — Wikipedia](https://en.wikipedia.org/wiki/Mutual_information)
- [TurboQuant paper — arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Wiley. Chapters 2, 8, 10.
