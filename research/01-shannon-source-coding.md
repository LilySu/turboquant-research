# Shannon's Source Coding Theory and Rate-Distortion Foundations

## Overview

Shannon's source coding theory provides the information-theoretic bedrock for TurboQuant. It establishes fundamental limits on how well any compression algorithm can perform — limits that TurboQuant approaches within a factor of ~2.7x. This document traces the key concepts from Shannon's 1948 paper through to their application in TurboQuant's distortion bounds.

## 1. Differential Entropy h(x)

**What is it?** Differential entropy extends Shannon's discrete entropy to continuous random variables. It measures the "inherent information content" or uncertainty of a continuous distribution.

**Definition.** For a continuous random variable X with probability density function f_X(x):

```
h(X) = -∫ f_X(x) log f_X(x) dx
```

where the integral is over the support of X and log is base-2 (for bits) or natural (for nats).

**Key properties:**
- Unlike discrete entropy, differential entropy can be **negative** (e.g., a uniform distribution on [0, 1/2] has h(X) = -1 bit).
- It is **translation invariant**: h(X + c) = h(X).
- It scales with amplitude: h(aX) = h(X) + log|a|.
- Among all distributions with variance σ², the **Gaussian maximizes** differential entropy: h(X) ≤ ½ log(2πeσ²), with equality iff X ~ N(μ, σ²).

**Example: Gaussian.** For X ~ N(0, σ²):

```
h(X) = ½ log₂(2πeσ²)
```

For σ² = 1: h(X) = ½ log₂(2πe) ≈ 2.047 bits.

**Connection to TurboQuant:** After random rotation, the coordinates of a unit-sphere vector follow a distribution that converges to N(0, 1/d). The differential entropy of these coordinates determines the Shannon Lower Bound on achievable distortion.

## 2. Mutual Information I(x; y)

**What is it?** Mutual information quantifies how much knowing one random variable tells you about another. It is the reduction in uncertainty about X when Y is observed (or vice versa).

**Definition.** For jointly continuous random variables (X, Y) with joint density p(x, y) and marginals p_X(x), p_Y(y):

```
I(X; Y) = ∫∫ p(x,y) log[ p(x,y) / (p_X(x) · p_Y(y)) ] dx dy
```

**Equivalent formulations via entropy:**

```
I(X; Y) = h(X) - h(X|Y) = h(Y) - h(Y|X) = h(X) + h(Y) - h(X, Y)
```

**Key properties:**
- **Non-negative**: I(X; Y) ≥ 0, with equality iff X and Y are independent.
- **Symmetric**: I(X; Y) = I(Y; X).
- **Concave** in p_X for fixed channel p_{Y|X}.
- **Convex** in p_{Y|X} for fixed source p_X.

**Example: Gaussian channel.** If Y = X + Z where X ~ N(0, S) and Z ~ N(0, 1) are independent:

```
I(X; Y) = ½ log₂(1 + S)
```

This is exactly Shannon's capacity formula for the Gaussian channel.

**Connection to TurboQuant:** The distortion-rate function is defined as an optimization over mutual information. A quantizer that uses B bits has at most B bits of mutual information between input and output, constraining the achievable distortion.

## 3. The Distortion-Rate Function D(R)

**Plain English.** The distortion-rate function D(R) answers: *Given a budget of R bits per symbol, what is the minimum possible distortion?* No matter how clever the encoder/decoder pair, they cannot beat D(R). It is the inverse of the rate-distortion function R(D).

**Formal definition (vector form, as in TurboQuant).** For a random vector **x** ∈ ℝ^d with distribution p_X:

```
D(p_X, B) := inf { E[||x - y||²₂] : I(x; y) ≤ B }
```

where the infimum is over all joint distributions of (x, y) satisfying the mutual information constraint, and B is the total bit budget.

**Shannon's Lossy Source Coding Theorem:**
- **Achievability**: For any rate R > R(D), there exists a coding scheme achieving distortion ≤ D.
- **Converse**: No coding scheme with rate R < R(D) can achieve distortion ≤ D.

**Gaussian source closed form.** For X ~ N(0, σ²) with MSE distortion:

```
R(D) = ½ log₂(σ²/D)    for D ≤ σ²
D(R) = σ² · 2^(-2R)
```

This yields the "6 dB per bit" rule: each additional bit halves the distortion (reduces it by 6 dB).

## 4. The Shannon Lower Bound (Lemma 2 in TurboQuant)

The Shannon Lower Bound (SLB) is the key tool TurboQuant uses to prove its near-optimality. It provides a **universal lower bound** on D(R) that applies to any source distribution — not just Gaussian.

**Lemma 2 (from TurboQuant paper).** For a random vector **x** ∈ ℝ^d with distribution p_X and finite differential entropy h(**x**), the MSE distortion-rate function D(B) for bit complexity B ≥ 0 satisfies:

```
D(p_X, B) ≥ (d / 2πe) · 2^((2/d)(h(x) - B))
```

**Derivation intuition.** The SLB arises from three facts:
1. The mutual information I(x; y) ≤ B constrains how much information the quantized version y can carry about x.
2. The distortion E[||x - y||²] relates to the conditional entropy h(x|y).
3. The Gaussian maximizes entropy for a given variance, so the Gaussian source is the "hardest" to compress — giving a universal lower bound.

More precisely, the SLB decomposes as:

```
R_SLB(D) = h(X) - ½ log₂(2πeD)
```

Inverting: D ≥ (1/2πe) · 2^(2(h(X) - R)). The d-dimensional vector version scales by d.

**Why this matters for TurboQuant.** The paper proves lower bounds (Theorem 3) by:
1. Using **Yao's minimax principle** to reduce worst-case randomized quantizers to average-case deterministic quantizers on the uniform hypersphere distribution.
2. Applying the **SLB** (Lemma 2) to the uniform distribution on S^{d-1}, whose differential entropy is known.
3. Obtaining: D_mse(Q) ≥ 1/4^b for any quantizer Q using b bits per coordinate.

TurboQuant achieves D_mse ≤ (√3π/2) · (1/4^b), matching the lower bound within a factor of √3π/2 ≈ 2.72.

## 5. Worked Examples

### Example 1: Rate-Distortion for a Gaussian Source

Let X ~ N(0, 1) (σ² = 1). Using b = 2 bits per sample:

```
D(R) = σ² · 2^(-2R) = 1 · 2^(-4) = 0.0625
```

So with 2 bits per sample, the minimum achievable MSE distortion for a Gaussian source is 0.0625.

### Example 2: Shannon Lower Bound on the Unit Sphere

For **x** uniform on S^{d-1} with d = 128, using b = 2 bits per coordinate (B = 256 total bits):

The differential entropy of the uniform distribution on S^{d-1} is:

```
h(x) = log(A_d)    where A_d = 2π^(d/2) / Γ(d/2)
```

For large d, the SLB gives approximately:

```
D ≥ 1/4^b = 1/16 = 0.0625
```

TurboQuant achieves D_mse ≤ (√3π/2)/4^b ≈ 2.72 × 0.0625 ≈ 0.170.

### Example 3: Numerical Comparison Across Bit-Widths

```python
import math

def slb_lower_bound(b):
    """Information-theoretic lower bound on MSE for b bits/coordinate."""
    return 1.0 / (4 ** b)

def turboquant_mse(b):
    """TurboQuant's achieved MSE distortion (Theorem 1)."""
    return (math.sqrt(3) * math.pi / 2) / (4 ** b)

print(f"{'Bits':>4} | {'Lower Bound':>12} | {'TurboQuant':>12} | {'Ratio':>6}")
print("-" * 42)
for b in [1, 2, 3, 4]:
    lb = slb_lower_bound(b)
    tq = turboquant_mse(b)
    print(f"{b:>4} | {lb:>12.6f} | {tq:>12.6f} | {tq/lb:>6.2f}")
```

Output:

```
Bits |  Lower Bound |   TurboQuant |  Ratio
------------------------------------------
   1 |     0.250000 |     0.680175 |   2.72
   2 |     0.062500 |     0.170044 |   2.72
   3 |     0.015625 |     0.042511 |   2.72
   4 |     0.003906 |     0.010628 |   2.72
```

The constant factor of ≈2.72 (= √3π/2) is maintained across all bit-widths, confirming TurboQuant's near-optimality.

## 6. Summary: How Shannon Connects to TurboQuant

| Concept | Role in TurboQuant |
|---|---|
| Differential entropy h(x) | Determines the fundamental compressibility of rotated KV vectors |
| Mutual information I(x;y) | Constrains how much info a b-bit quantizer can preserve |
| Distortion-rate D(R) | Establishes the floor — no quantizer can beat this |
| Shannon Lower Bound | Used in Theorem 3 to prove TurboQuant is within 2.72x of optimal |
| Gaussian maximality | Justifies why the Gaussian/Beta-distributed coordinates after rotation are a natural fit |

## References

1. Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379-423. [Harvard hosted PDF](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
2. Shannon, C.E. (1959). "Coding Theorems for a Discrete Source with a Fidelity Criterion." *IRE National Convention Record*, Part 4, 142-163.
3. Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
4. Dufy et al. (2025). "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
5. [Rate-Distortion Theory — Wikipedia](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory)
6. [Stanford EE398A: Rate Distortion Theory](https://web.stanford.edu/class/ee398a/handouts/lectures/04-RateDistortionTheory.pdf)
7. [Yale Lecture 13: Shannon Lower Bound](http://www.stat.yale.edu/~yw562/teaching/598/lec13.pdf)
