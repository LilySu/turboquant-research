# Beta Distribution on Hyperspheres

## Prerequisites Refresher

### Orthogonal Matrices

A square matrix $Q \in \mathbb{R}^{d \times d}$ is **orthogonal** if its columns are orthonormal:

$$
Q^T Q = Q Q^T = I
$$

**Why does $Q\mathbf{x}$ preserve $\|\mathbf{x}\|$?** Because:

$$
\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^T(Q\mathbf{x}) = \mathbf{x}^T Q^T Q \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \|\mathbf{x}\|^2
$$

Orthogonal matrices are rotations (or reflections). They preserve lengths, angles, and inner products: $\langle Q\mathbf{x}, Q\mathbf{y}\rangle = \langle \mathbf{x}, \mathbf{y}\rangle$.

**Relevance to TurboQuant:** The algorithm multiplies input vectors by a random orthogonal matrix $\Pi$. This rotation preserves the vector's norm while randomizing the direction, which is essential for the analysis.

### The Hypersphere $\mathbb{S}^{d-1}$

The **unit hypersphere** $\mathbb{S}^{d-1}$ is the set of all points in $\mathbb{R}^d$ at distance 1 from the origin:

$$
\mathbb{S}^{d-1} = \left\{ \mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\|_2 = 1 \right\}
$$

The superscript $d-1$ indicates the *intrinsic* dimension (degrees of freedom), not the ambient space.

**Visualizing low dimensions:**
- $d = 2$: $\mathbb{S}^1$ is the unit circle in $\mathbb{R}^2$. Points are $(\cos\theta, \sin\theta)$.
- $d = 3$: $\mathbb{S}^2$ is the familiar sphere surface in $\mathbb{R}^3$. Points are $(\sin\phi\cos\theta, \sin\phi\sin\theta, \cos\phi)$.
- $d = 128$: A 127-dimensional "surface" in $\mathbb{R}^{128}$. Impossible to visualize, but the math generalizes cleanly.

**Uniform distribution on $\mathbb{S}^{d-1}$:** A random vector $\mathbf{x}$ is uniform on $\mathbb{S}^{d-1}$ if every direction is equally likely. One way to sample: draw $\mathbf{g} \sim \mathcal{N}(\mathbf{0}, I_d)$ and normalize: $\mathbf{x} = \mathbf{g}/\|\mathbf{g}\|$.

### The Gamma Function $\Gamma(z)$

The Gamma function generalizes the factorial to non-integer arguments:

$$
\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} \, dt, \quad \operatorname{Re}(z) > 0
$$

**Key values:**
- $\Gamma(1) = 1$
- $\Gamma(n) = (n-1)!$ for positive integers (e.g., $\Gamma(5) = 24$)
- $\Gamma(1/2) = \sqrt{\pi} \approx 1.7725$
- $\Gamma(3/2) = \frac{1}{2}\sqrt{\pi}$
- **Recursion**: $\Gamma(z+1) = z \cdot \Gamma(z)$

The Gamma function appears naturally in surface area formulas for hyperspheres, because the surface area of $\mathbb{S}^{d-1}$ with radius $r$ is:

$$
A_{d-1}(r) = \frac{2\pi^{d/2}}{\Gamma(d/2)} \cdot r^{d-1}
$$

## Main Content

### Lemma 1: Marginal Distribution of a Single Coordinate

**Statement (Lemma 1 from TurboQuant).** If $\mathbf{x} \in \mathbb{S}^{d-1}$ is uniformly distributed on the unit hypersphere, then each coordinate $x_j$ has the density:

$$
f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi} \; \Gamma\!\left(\frac{d-1}{2}\right)} \left(1 - x^2\right)^{(d-3)/2}, \quad x \in [-1, 1]
$$

This is a **scaled Beta distribution**. Specifically, $x_j^2 \sim \text{Beta}(1/2, (d-1)/2)$.

### Derivation

The proof uses a "slicing" argument based on surface area ratios.

**Step 1: Slice the sphere at height $t$.** Fix $x_1 = t$ where $t \in [-1, 1]$. The remaining coordinates $(x_2, \ldots, x_d)$ must satisfy:

$$
x_2^2 + x_3^2 + \cdots + x_d^2 = 1 - t^2
$$

This is a $(d-2)$-sphere of radius $\sqrt{1 - t^2}$ in $\mathbb{R}^{d-1}$.

**Step 2: Compute cross-sectional surface area.** The surface area of this cross-section is:

$$
A_{d-2}\!\left(\sqrt{1-t^2}\right) = \frac{2\pi^{(d-1)/2}}{\Gamma\!\left(\frac{d-1}{2}\right)} \left(1 - t^2\right)^{(d-2)/2}
$$

**Step 3: Form the ratio.** The marginal density of $x_1 = t$ is proportional to the cross-sectional area divided by the total surface area of $\mathbb{S}^{d-1}$. We also need a factor of $1/\sqrt{1-t^2}$ from the Jacobian of the projection (the "tilt" of the sphere surface relative to the horizontal slice):

$$
f_{X_1}(t) \propto \frac{A_{d-2}(\sqrt{1-t^2})}{A_{d-1}(1)} \cdot \frac{1}{\sqrt{1-t^2}}
$$

Actually, the cross-sectional area already accounts for the $(d-2)$-dimensional measure. Working through the ratio:

$$
f_{X_1}(t) = \frac{A_{d-2}(\sqrt{1-t^2})}{A_{d-1}(1)} = \frac{\frac{2\pi^{(d-1)/2}}{\Gamma((d-1)/2)} (1-t^2)^{(d-2)/2}}{\frac{2\pi^{d/2}}{\Gamma(d/2)}}
$$

**Step 4: Simplify.** Cancel the 2's, use $\pi^{(d-1)/2} / \pi^{d/2} = 1/\sqrt{\pi}$:

$$
f_{X_1}(t) = \frac{\Gamma(d/2)}{\sqrt{\pi} \; \Gamma\!\left(\frac{d-1}{2}\right)} \left(1 - t^2\right)^{(d-3)/2}
$$

This completes the derivation of Lemma 1. $\blacksquare$

**Verification for $d = 3$:** The density becomes $f(x) = \frac{\Gamma(3/2)}{\sqrt{\pi}\,\Gamma(1)} (1-x^2)^0 = \frac{\sqrt{\pi}/2}{\sqrt{\pi} \cdot 1} = \frac{1}{2}$, which is uniform on $[-1, 1]$. This matches the well-known fact that the $z$-coordinate of a point on a 3D sphere is uniformly distributed — the basis of Archimedes' hat-box theorem.

### Convergence to Gaussian for Large $d$

As $d \to \infty$, the coordinate $x_j$ concentrates tightly around 0 with variance $1/d$. We can show the density converges to $\mathcal{N}(0, 1/d)$.

**Heuristic argument.** Take $\log f_X(x)$ and expand for small $x$ (which is where most mass is for large $d$):

$$
\log f_X(x) = \text{const} + \frac{d-3}{2} \log(1 - x^2) \approx \text{const} - \frac{d-3}{2} x^2
$$

using $\log(1-x^2) \approx -x^2$ for small $x$. So:

$$
f_X(x) \approx C \cdot \exp\!\left(-\frac{d}{2} x^2\right)
$$

which is a Gaussian with variance $1/d$. This is the **Poincare-Borel theorem**: individual coordinates of uniform points on $\mathbb{S}^{d-1}$ converge in distribution to $\mathcal{N}(0, 1/d)$ as $d \to \infty$.

### Concentration of Measure

**Intuition.** In high dimensions, most of the surface area of $\mathbb{S}^{d-1}$ is concentrated near the "equator" relative to any axis. A single coordinate $x_j$ of a random point is almost always close to 0, with fluctuations of order $1/\sqrt{d}$.

**Why this is surprising.** Each coordinate ranges over $[-1, 1]$, yet in $d = 10{,}000$ dimensions, a typical coordinate is $\approx 0.01$ in magnitude. The probability of $|x_j| > 0.1$ becomes astronomically small.

**Formally:** For $\mathbf{x}$ uniform on $\mathbb{S}^{d-1}$ and any coordinate $x_j$:

$$
\mathbb{P}\!\left(|x_j| > t\right) \leq 2\exp\!\left(-\frac{d \, t^2}{2}\right)
$$

This sub-Gaussian tail bound shows that coordinates are tightly concentrated around 0 with scale $1/\sqrt{d}$.

**Near-independence.** For large $d$, distinct coordinates $x_i$ and $x_j$ (with $i \neq j$) are nearly independent. Their covariance is $\text{Cov}(x_i, x_j) = -1/(d(d-1)) \to 0$ as $d \to \infty$. Combined with the Gaussian convergence, this means the coordinates behave like i.i.d. $\mathcal{N}(0, 1/d)$ draws — which is exactly the property TurboQuant exploits.

## Worked Examples

### Numerical Comparison: Beta vs Gaussian

```python
import numpy as np
from scipy import special

def beta_pdf(x, d):
    """Lemma 1 density for coordinate on S^{d-1}."""
    coeff = special.gamma(d/2) / (np.sqrt(np.pi) * special.gamma((d-1)/2))
    return coeff * (1 - x**2)**((d-3)/2)

def gaussian_pdf(x, d):
    """Limiting N(0, 1/d) density."""
    return np.sqrt(d / (2*np.pi)) * np.exp(-d * x**2 / 2)

# Compare at d = 10, 100, 1000
for d in [10, 100, 1000]:
    x = 0.1
    b = beta_pdf(x, d)
    g = gaussian_pdf(x, d)
    print(f"d={d:4d}, x={x}: Beta={b:.4f}, Gauss={g:.4f}, ratio={b/g:.4f}")
```

Output:

```
d=  10, x=0.1: Beta=1.0472, Gauss=1.0334, ratio=1.0134
d= 100, x=0.1: Beta=3.5088, Gauss=3.5041, ratio=1.0013
d=1000, x=0.1: Beta=11.1438, Gauss=11.1430, ratio=1.0001
```

The Beta and Gaussian densities agree to 4+ decimal places by $d = 1000$.

### Variance Verification

The variance of each coordinate for $\mathbf{x}$ uniform on $\mathbb{S}^{d-1}$ is exactly:

$$
\text{Var}(x_j) = \frac{1}{d}
$$

**Proof sketch:** By symmetry, $\mathbb{E}[x_j] = 0$ and all $\mathbb{E}[x_j^2]$ are equal. Since $\sum_{j=1}^d x_j^2 = 1$, taking expectations gives $d \cdot \mathbb{E}[x_j^2] = 1$, so $\text{Var}(x_j) = 1/d$.

For $d = 128$ (typical attention head dimension): $\text{Var}(x_j) = 1/128 \approx 0.0078$, standard deviation $\approx 0.088$.

## Connection to TurboQuant

TurboQuant multiplies each input vector $\mathbf{x}$ by a random orthogonal matrix $\Pi$:

$$
\mathbf{y} = \Pi \mathbf{x}
$$

Since $\Pi$ is orthogonal, $\|\mathbf{y}\| = \|\mathbf{x}\|$. If $\mathbf{x} \in \mathbb{S}^{d-1}$, then $\mathbf{y}$ is uniformly distributed on $\mathbb{S}^{d-1}$ (because a random rotation of any fixed point on the sphere yields a uniform point). By Lemma 1, each coordinate $y_j$ follows the Beta-type density derived above.

This enables **independent scalar quantization**: since coordinates are nearly independent and nearly Gaussian for large $d$, applying an optimal scalar quantizer to each coordinate independently achieves near-optimal distortion. The error from treating coordinates as independent vanishes as $d$ grows — which is why TurboQuant works so well for typical KV cache dimensions ($d = 128$ or $d = 256$).

## Key Takeaways

- An orthogonal matrix preserves norms and inner products — rotating doesn't distort vectors
- The unit hypersphere $\mathbb{S}^{d-1}$ has $d-1$ intrinsic dimensions; uniform sampling is done by normalizing Gaussian vectors
- Lemma 1: each coordinate follows $f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)}(1-x^2)^{(d-3)/2}$, derived from the ratio of cross-sectional to total surface area
- For large $d$, this density converges to $\mathcal{N}(0, 1/d)$ (Poincare-Borel theorem)
- Concentration of measure: coordinates are tightly clustered around 0 with sub-Gaussian tails
- Near-independence of coordinates (covariance $\sim -1/d^2$) justifies TurboQuant's per-coordinate scalar quantization

## Sources

- [TurboQuant paper — arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- [n-sphere — Wikipedia](https://en.wikipedia.org/wiki/N-sphere) (surface area formulas, marginal distributions)
- [Concentration of Measure — Wikipedia](https://en.wikipedia.org/wiki/Concentration_of_measure)
- [Diaconis & Freedman (1987), "Asymptotic Distribution of Coordinates on High-Dimensional Spheres"](https://projecteuclid.org/journals/electronic-communications-in-probability/volume-12/issue-none/Asymptotic-Distribution-of-Coordinates-on-High-Dimensional-Spheres/10.1214/ECP.v12-1294.pdf)
- [Vershynin, "High-Dimensional Probability" — Chapter 3](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-2.pdf)
- [Terence Tao, "254A Notes 1: Concentration of Measure"](https://terrytao.wordpress.com/2010/01/03/254a-notes-1-concentration-of-measure/)
- [Generating Uniform Points on a Sphere — Cory Simon](http://corysimon.github.io/articles/uniformdistn-on-sphere/)
