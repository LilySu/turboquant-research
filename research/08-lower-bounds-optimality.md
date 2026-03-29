# Lower Bounds and Near-Optimality

## Prerequisites Refresher

### Yao's Minimax Principle

**Yao's principle** (Yao, 1977) connects randomized algorithms to deterministic ones. In plain terms:

> The worst-case expected cost of the best **randomized** algorithm equals the best expected cost of a **deterministic** algorithm on the hardest input distribution.

**Why this matters for TurboQuant:** TurboQuant is a *randomized* quantizer (it uses a random rotation $\Pi$). To prove a lower bound against all randomized quantizers, Yao's principle lets us instead:
1. Choose a **hard input distribution** (uniform on $\mathbb{S}^{d-1}$)
2. Show that every **deterministic** quantizer has high distortion on this distribution
3. Conclude that every **randomized** quantizer has the same worst-case distortion

This reduction simplifies the proof enormously: instead of reasoning about all possible randomized strategies, we only need to analyze one fixed distribution.

### Lower Bounds in Algorithms

A **lower bound** proves that no algorithm (however clever) can do better than a certain threshold. Upper bounds show what *is* achievable; lower bounds show what *isn't*. The gap between them measures how close the algorithm is to optimal.

## Main Content

### Theorem 3: Lower Bounds on Quantization Distortion

**Statement.** For any randomized quantizer $Q: \mathbb{S}^{d-1} \to \{0, 1\}^{bd}$ with bit-width $b$ and reconstruction map $Q^{-1}$, there exist hard inputs $\mathbf{x} \in \mathbb{S}^{d-1}$ such that:

**MSE lower bound:**

$$
D_{\text{mse}}(Q) = \mathbb{E}\!\left[\|\mathbf{x} - Q^{-1}(Q(\mathbf{x}))\|_2^2\right] \geq \frac{1}{4^b}
$$

| Symbol | Meaning |
|:--|:--|
| $D_{\text{mse}}(Q)$ | Worst-case MSE distortion of quantizer $Q$ |
| $Q: \mathbb{S}^{d-1} \to \{0,1\}^{bd}$ | Any quantizer: maps unit vectors to $bd$ bits |
| $Q^{-1}$ | Reconstruction (dequantization) map |
| $4^{-b}$ | The floor: cannot be beaten by any method |

**Inner product lower bound:**

$$
D_{\text{prod}}(Q) = \mathbb{E}\!\left[\left|\langle \mathbf{y}, \mathbf{x}\rangle - \langle \mathbf{y}, Q^{-1}(Q(\mathbf{x}))\rangle\right|^2\right] \geq \frac{1}{d} \cdot \frac{1}{4^b}
$$

| Symbol | Meaning |
|:--|:--|
| $D_{\text{prod}}(Q)$ | Worst-case inner product distortion |
| $1/d$ | Dimension factor: inner product error is inherently $1/d$ scale |
| $4^{-b}$ | Same exponential decay as MSE |

### Proof of Theorem 3

#### Step 1: Apply Yao's Minimax Principle

By Yao's minimax, the worst-case expected MSE of any randomized quantizer $Q$ is at least:

$$
\max_{\mathbf{x} \in \mathbb{S}^{d-1}} \mathbb{E}_Q\!\left[\|\mathbf{x} - Q^{-1}(Q(\mathbf{x}))\|^2\right] \geq \min_{\text{det. } Q} \mathbb{E}_{\mathbf{x} \sim \text{Unif}(\mathbb{S}^{d-1})}\!\left[\|\mathbf{x} - Q^{-1}(Q(\mathbf{x}))\|^2\right]
$$

| Left side | Right side |
|:--|:--|
| Best randomized $Q$, worst-case input $\mathbf{x}$ | Best deterministic $Q$, uniform input on $\mathbb{S}^{d-1}$ |

The uniform distribution on $\mathbb{S}^{d-1}$ is the "hardest" distribution because it has maximum entropy over the sphere — no direction is easier to predict.

#### Step 2: Apply the Shannon Lower Bound (Lemma 3)

For $\mathbf{x}$ uniform on $\mathbb{S}^{d-1}$, Lemma 3 (a specialization of Lemma 2 from US-001) gives:

$$
D(B) \geq 2^{-2B/d}
$$

| Symbol | Meaning |
|:--|:--|
| $D(B)$ | Minimum achievable MSE for any coding scheme with $B$ bits total |
| $B = bd$ | Total bit budget ($b$ bits per coordinate) |
| $2^{-2B/d} = 2^{-2b} = 4^{-b}$ | Substituting $B = bd$ |

**Derivation of Lemma 3.** The differential entropy of the uniform distribution on $\mathbb{S}^{d-1}$ is $h(\mathbf{x}) = \log_2 A_d$, where $A_d = 2\pi^{d/2}/\Gamma(d/2)$ is the surface area. Applying Lemma 2 (SLB):

$$
D(B) \geq \frac{d}{2\pi e} \cdot A_d^{2/d} \cdot 2^{-2B/d}
$$

Using Stirling's approximation: $A_d^{2/d} \geq (2\pi e / d) \cdot (1 - O(1/d))$. Substituting:

$$
D(B) \geq \frac{d}{2\pi e} \cdot \frac{2\pi e}{d} \cdot 2^{-2b} = 4^{-b}
$$

The sphere's surface area and Gaussian entropy constants cancel perfectly, leaving the clean bound $4^{-b}$.

#### Step 3: Combine

By Steps 1 and 2: any randomized quantizer $Q$ must have worst-case $D_{\text{mse}} \geq 4^{-b}$. $\blacksquare$

The inner product bound $D_{\text{prod}} \geq (1/d) \cdot 4^{-b}$ follows similarly by considering the Cauchy-Schwarz relationship between MSE and inner product distortion.

### The Approximation Factor

Comparing TurboQuant's upper bounds (Theorems 1 and 2) to the lower bounds (Theorem 3):

**MSE:**

$$
\text{Ratio} = \frac{D_{\text{mse}}^{\text{upper}}}{D_{\text{mse}}^{\text{lower}}} = \frac{\frac{\sqrt{3}\pi}{2} \cdot 4^{-b}}{4^{-b}} = \frac{\sqrt{3}\,\pi}{2} \approx 2.72
$$

**Inner product:**

$$
\text{Ratio} = \frac{D_{\text{prod}}^{\text{upper}}}{D_{\text{prod}}^{\text{lower}}} = \frac{\frac{\sqrt{3}\pi^2}{d} \cdot 4^{-b}}{\frac{1}{d} \cdot 4^{-b}} = \sqrt{3}\,\pi^2 \approx 17.1
$$

The inner product gap is larger because it combines the MSE gap ($\sqrt{3}\pi/2$) with the QJL variance factor ($\pi$).

### Comparison Table

| $b$ | Lower bound $4^{-b}$ | TurboQuant MSE (Thm 1) | Empirical MSE | Ratio (Thm 1) | Ratio (empirical) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.2500 | 0.6802 | $\approx 0.36$ | 2.72 | $\approx 1.44$ |
| 2 | 0.0625 | 0.1700 | $\approx 0.117$ | 2.72 | $\approx 1.87$ |
| 3 | 0.01563 | 0.04251 | $\approx 0.03$ | 2.72 | $\approx 1.92$ |
| 4 | 0.003906 | 0.01063 | $\approx 0.009$ | 2.72 | $\approx 2.30$ |
| 5 | 0.000977 | 0.002658 | — | 2.72 | — |

### Why the Gap Is Tighter at Low Bit-Widths

The Theorem 1 bound of $\sqrt{3}\pi/2 \approx 2.72$ comes from the **Panter-Dite high-resolution formula**, which is an *asymptotic* approximation valid for large $2^b$. At low bit-widths ($b = 1, 2$):

1. The actual Lloyd-Max codebook achieves **better MSE** than the Panter-Dite prediction
2. The empirical ratio (column 6) is much tighter: $\approx 1.44\times$ at $b = 1$
3. As $b$ increases, the Panter-Dite approximation becomes more accurate and the empirical ratio approaches $2.72$

**This means TurboQuant is especially good at the low bit-widths most relevant for KV cache compression** ($b = 2, 3, 4$).

## Connection to TurboQuant

1. **Near-optimality is the paper's central claim.** Theorem 3 proves no algorithm can beat $4^{-b}$; Theorems 1-2 show TurboQuant achieves $\approx 2.72 \times 4^{-b}$. The gap is a *constant* — it doesn't grow with $d$ or $b$.

2. **The proof strategy** elegantly chains: Yao's minimax → uniform sphere distribution → Shannon Lower Bound → Stirling's approximation → clean $4^{-b}$ bound.

3. **Practical implications:** At 3.5 bits/coordinate, TurboQuant achieves "quality-neutral" KV cache compression with $4\text{-}5\times$ memory savings. The lower bound confirms this is close to the information-theoretic limit.

4. **What the gap means:** The $2.72\times$ factor means TurboQuant uses roughly $\log_4(2.72) \approx 0.72$ bits more per coordinate than the information-theoretic minimum. This is a modest overhead for a practical, $O(d \log d)$ algorithm.

## Key Takeaways

- **Yao's minimax principle** reduces worst-case randomized bounds to average-case deterministic bounds
- **Theorem 3**: any quantizer on $\mathbb{S}^{d-1}$ must incur $D_{\text{mse}} \geq 4^{-b}$ (from Shannon + Stirling)
- TurboQuant achieves $D_{\text{mse}} \leq 2.72 \times 4^{-b}$ — a **constant-factor** gap to optimal
- The gap is even tighter empirically: $\approx 1.4\times$ at $b = 1$, $\approx 1.9\times$ at $b = 2$
- The $2.72$ factor comes from Panter-Dite and is loose at low bit-widths
- In practice: $\approx 0.72$ extra bits per coordinate above the information-theoretic minimum

## Sources

- [TurboQuant paper — arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [TurboQuant HTML — arXiv](https://arxiv.org/html/2504.19874v1)
- [TurboQuant — OpenReview (ICLR 2026)](https://openreview.net/forum?id=tO3ASKZlok)
- [Yao's Principle — Wikipedia](https://en.wikipedia.org/wiki/Yao%27s_principle)
- [Georgia Tech: Yao's Minimax Lemma lecture](https://faculty.cc.gatech.edu/~ssingla7/courses/Spring22/lec8.pdf)
- [MIT OCW: Yao's Minimax Principle (6.856J)](https://ocw.mit.edu/courses/6-856j-randomized-algorithms-fall-2002/e6cd56bb54a44a8a138490daa2ff48ba_n3.pdf)
- [University of Bonn: Yao's Principle lecture notes](https://nerva.cs.uni-bonn.de/lib/exe/fetch.php/teaching/ws1819/vl-aau/lecturenotes05.pdf)
- [Waterloo: Minimax Principle](https://cs.uwaterloo.ca/~eblais/cs860/w25/minimax)
- [Shannon Lower Bound — see US-001 sources](./01-shannon-source-coding.md)
- [Rate-Distortion Theory — Wikipedia](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory)
