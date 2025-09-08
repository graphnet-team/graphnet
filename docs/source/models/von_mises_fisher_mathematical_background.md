

# Mathematical Background: von Mises-Fisher Loss Implementation

## Calculation for 3D
To show that for $m=3$,

$$- \frac{I_{m/2}(\kappa)}{I_{m/2-1}(\kappa)} = \frac{1}{\kappa}-\frac{1}{\tanh(\kappa)}$$

we first substitute $m=3$ into the equation. This gives us:

$$- \frac{I_{3/2}(\kappa)}{I_{1/2}(\kappa)} = \frac{1}{\kappa}-\frac{1}{\tanh(\kappa)}$$

We need to evaluate the left side of this equation, which involves **modified Bessel functions of the first kind** of half-integer order, specifically $I_{3/2}(\kappa)$ and $I_{1/2}(\kappa)$. 

### Expressing Bessel Functions

The modified Bessel functions of the first kind of half-integer order can be expressed in terms of elementary hyperbolic functions.

For the order $1/2$, the function is:

$$I_{1/2}(\kappa) = \sqrt{\frac{2}{\pi\kappa}} \sinh(\kappa)$$

For the order $3/2$, the function is:

$$I_{3/2}(\kappa) = \sqrt{\frac{2}{\pi\kappa}} \left( \cosh(\kappa) - \frac{\sinh(\kappa)}{\kappa} \right)$$

### Calculating the Ratio

Now, we can compute the ratio on the left side of the equation:

$$- \frac{I_{3/2}(\kappa)}{I_{1/2}(\kappa)} = - \frac{\sqrt{\frac{2}{\pi\kappa}} \left( \cosh(\kappa) - \frac{\sinh(\kappa)}{\kappa} \right)}{\sqrt{\frac{2}{\pi\kappa}} \sinh(\kappa)}$$

The term $\sqrt{\frac{2}{\pi\kappa}}$ cancels out, leaving:

$$- \frac{\cosh(\kappa) - \frac{\sinh(\kappa)}{\kappa}}{\sinh(\kappa)} = - \left( \frac{\cosh(\kappa)}{\sinh(\kappa)} - \frac{1}{\kappa} \right)$$

Using the definition of the **hyperbolic cotangent**, $\coth(\kappa) = \frac{\cosh(\kappa)}{\sinh(\kappa)}$, we have:

$$- \left(\coth(\kappa) - \frac{1}{\kappa}\right) = \frac{1}{\kappa} - \coth(\kappa)$$

### Final Result

Finally, since $\coth(\kappa) = \frac{1}{\tanh(\kappa)}$, we can write:

$$\frac{1}{\kappa} - \frac{1}{\tanh(\kappa)}$$

This is exactly the right side of the initial equation. Thus, we have shown that for $m=3$:

$$- \frac{I_{3/2}(\kappa)}{I_{1/2}(\kappa)} = \frac{1}{\kappa}-\frac{1}{\tanh(\kappa)} \quad \blacksquare$$

---

## Taylor Series Approximation

To prove that the Taylor series for $f(\kappa) = \frac{1}{\kappa}-\frac{1}{\tanh(\kappa)}$ is well-defined and alternating at $\kappa = 0$, we first need to analyze the behavior of the function at the origin and then examine the structure of its series expansion.

### Well-Defined at $\kappa = 0$
The function $f(\kappa)$ appears to be singular at $\kappa=0$ because both $\frac{1}{\kappa}$ and $\frac{1}{\tanh(\kappa)}$ diverge. To determine if the singularity is removable, we can evaluate the limit as $\kappa \to 0$.

First, rewrite the function as a single fraction:

$$f(\kappa) = \frac{1}{\kappa} - \frac{\cosh(\kappa)}{\sinh(\kappa)} = \frac{\sinh(\kappa) - \kappa\cosh(\kappa)}{\kappa\sinh(\kappa)}$$

We can find the limit by expanding the hyperbolic functions into their Taylor series around $\kappa = 0$:

$$\sinh(\kappa) = \kappa + \frac{\kappa^3}{3!} + \frac{\kappa^5}{5!} + O(\kappa^7)$$

$$\cosh(\kappa) = 1 + \frac{\kappa^2}{2!} + \frac{\kappa^4}{4!} + O(\kappa^6)$$

Substituting these into the numerator and denominator:

**Numerator:**

$$\sinh(\kappa) - \kappa\cosh(\kappa) = \left(\kappa + \frac{\kappa^3}{6} + \dots\right) - \kappa\left(1 + \frac{\kappa^2}{2} + \dots\right)$$

$$= \left(\kappa + \frac{\kappa^3}{6}\right) - \left(\kappa + \frac{\kappa^3}{2}\right) + \dots = \left(\frac{1}{6} - \frac{1}{2}\right)\kappa^3 + \dots = -\frac{1}{3}\kappa^3 + O(\kappa^5)$$

**Denominator:**

$$\kappa\sinh(\kappa) = \kappa\left(\kappa + \frac{\kappa^3}{6} + \dots\right) = \kappa^2 + \frac{\kappa^4}{6} + O(\kappa^6)$$

Now, the limit of the function is:

$$\lim_{\kappa \to 0} f(\kappa) = \lim_{\kappa \to 0} \frac{-\frac{1}{3}\kappa^3 + O(\kappa^5)}{\kappa^2 + O(\kappa^4)} = \lim_{\kappa \to 0} \frac{\kappa^3(-\frac{1}{3} + O(\kappa^2))}{\kappa^2(1 + O(\kappa^2))} = \lim_{\kappa \to 0} \kappa \frac{-\frac{1}{3} + O(\kappa^2)}{1 + O(\kappa^2)} = 0$$

Since the limit exists and is finite, the singularity at $\kappa=0$ is removable. We can define $f(0) = 0$, and thus the Taylor series for $f(\kappa)$ around $\kappa=0$ is **well-defined**.

### Alternating Series

To show the series is alternating, we derive its form using the known series expansion for $\kappa \coth(\kappa)$ which involves the **Bernoulli numbers**, $B_{2n}$. The expansion is:

$$\kappa \coth(\kappa) = \sum_{n=0}^{\infty} \frac{B_{2n} (2\kappa)^{2n}}{(2n)!} = 1 + \frac{1}{3}\kappa^2 - \frac{1}{45}\kappa^4 + \frac{2}{945}\kappa^6 - \dots$$

Our function can be written as $f(\kappa) = \frac{1}{\kappa} - \coth(\kappa) = \frac{1 - \kappa \coth(\kappa)}{\kappa}$. Substituting the series:

$$f(\kappa) = \frac{1}{\kappa} \left( 1 - \sum_{n=0}^{\infty} \frac{B_{2n} (2\kappa)^{2n}}{(2n)!} \right)$$

The first term of the sum (for $n=0$) is $\frac{B_0 (2\kappa)^0}{0!} = 1$, since $B_0 = 1$.

$$f(\kappa) = \frac{1}{\kappa} \left( 1 - \left(1 + \sum_{n=1}^{\infty} \frac{B_{2n} (2\kappa)^{2n}}{(2n)!} \right) \right) = \frac{1}{\kappa} \left( - \sum_{n=1}^{\infty} \frac{B_{2n} 2^{2n} \kappa^{2n}}{(2n)!} \right)$$

$$f(\kappa) = - \sum_{n=1}^{\infty} \frac{B_{2n} 2^{2n} \kappa^{2n-1}}{(2n)!}$$

The sign of the Bernoulli numbers $B_{2n}$ for $n \ge 1$ alternates according to the formula $\text{sgn}(B_{2n}) = (-1)^{n-1}$. We can write $B_{2n} = (-1)^{n-1} |B_{2n}|$. Substituting this into the series for $f(\kappa)$:

$$f(\kappa) = - \sum_{n=1}^{\infty} \frac{(-1)^{n-1} |B_{2n}| 2^{2n} \kappa^{2n-1}}{(2n)!} = \sum_{n=1}^{\infty} (-1)^n \frac{|B_{2n}| 2^{2n} \kappa^{2n-1}}{(2n)!}$$

Let's write out the first few terms of the series:

$$f(\kappa) = -\frac{|B_2| 2^2}{(2)!}\kappa^1 + \frac{|B_4| 2^4}{(4)!}\kappa^3 - \frac{|B_6| 2^6}{(6)!}\kappa^5 + \dots$$

With $B_2=1/6$, $B_4=-1/30$, $B_6=1/42$:

$$f(\kappa) = -\frac{1/6 \cdot 4}{2}\kappa + \frac{1/30 \cdot 16}{24}\kappa^3 - \frac{1/42 \cdot 64}{720}\kappa^5 + \dots = -\frac{1}{3}\kappa + \frac{1}{45}\kappa^3 - \frac{2}{945}\kappa^5 + \dots$$

The series for $f(\kappa)$ contains only odd powers of $\kappa$. The coefficient of the term $\kappa^{2n-1}$ is:

$$c_{2n-1} = (-1)^n \frac{|B_{2n}| 2^{2n}}{(2n)!}$$

The coefficient of the next non-zero term, $\kappa^{2(n+1)-1} = \kappa^{2n+1}$, is:

$$c_{2n+1} = (-1)^{n+1} \frac{|B_{2(n+1)}| 2^{2(n+1)}}{(2(n+1))!}$$

Since $|B_{2k}|$ is positive for all $k \ge 1$, the sign of the coefficient $c_{2n-1}$ is determined by $(-1)^n$, and the sign of $c_{2n+1}$ is determined by $(-1)^{n+1}$. Clearly, these are opposite. Therefore, the coefficients of successive non-zero terms have alternating signs. This proves that the Taylor series for $f(\kappa)$ at $\kappa=0$ is an **alternating series**.

### Error Bound

The Taylor series derived for the function $f(\kappa)$ is an alternating series, meaning the signs of successive terms alternate. For such series, the error from truncating the series can be estimated using the **Alternating Series Estimation Theorem**. This theorem states that if you approximate the sum of a convergent alternating series by its $N$-th partial sum, the absolute value of the error (the remainder) is less than or equal to the absolute value of the first neglected term. This holds true under the conditions that the absolute values of the terms are monotonically decreasing and approach zero.

These conditions are met by the series for $f(\kappa)$ within its radius of convergence ($|\kappa| < \pi$). The magnitude of the general term, $\frac{|B_{2n}| 2^{2n} |\kappa|^{2n-1}}{(2n)!}$, tends to zero as $n \to \infty$, ensuring convergence. Moreover, for any given $\kappa$ in this interval, the magnitudes of the terms will eventually be monotonically decreasing. For sufficiently small values of $\kappa$, this decreasing trend holds from the very first term. Therefore, the error in approximating the function with the sum of its first $N$ terms is bounded by the magnitude of the $(N+1)$-th term. For example, the error in the approximation $f(\kappa) \approx -\frac{1}{3}\kappa$ is less than or equal to the next term's magnitude, $\frac{1}{45}|\kappa|^3$. Thus:

$$\left\lvert \kappa \right\rvert < 10^{-6} \implies \varepsilon \lesssim \mathcal{O}(10^{-21})$$

---

## Implementation in GraphNeT

### Problem Statement
The mathematical derivation above provides exact formulas for computing gradients in the von Mises-Fisher loss function. However, a naive implementation would encounter **division by zero errors** when $\kappa = 0$, even though the mathematical limit is well-defined. This creates RuntimeWarnings and potential numerical instability.

### Numerical Challenge
The core issue arises in the backward pass when computing:

$$\frac{\partial}{\partial \kappa} \log C_3(\kappa) = \frac{1}{\kappa} - \frac{1}{\tanh(\kappa)}$$

For $\kappa = 0$:
- Both $\frac{1}{\kappa}$ and $\frac{1}{\tanh(\kappa)}$ diverge individually
- Their difference converges to the finite limit $\lim_{\kappa \to 0} f(\kappa) = 0$- Standard floating-point evaluation triggers division by zero warnings

### Solution Strategy
Our implementation uses **boolean masking** to conditionally apply different computational approaches:

```python
# Initialize gradient array
grads = np.zeros_like(kappa)

# Handle small kappa values (including zero)
small_mask = np.abs(kappa) < 1e-6
grads[small_mask] = -kappa[small_mask] / 3

# Handle large kappa values  
large_mask = ~small_mask
if np.any(large_mask):
    kappa_large = kappa[large_mask]
    grads[large_mask] = 1/kappa_large - 1/np.tanh(kappa_large)
```

### Key Implementation Features

1. **Threshold Selection**: $|\kappa| < 10^{-6}$   - Based on error analysis: truncation error $\leq \frac{|\kappa|^3}{45} \approx 10^{-21}$   - Well below machine precision for typical floating-point arithmetic
   - Provides excellent numerical accuracy

2. **Boolean Masking vs. np.where()**
   - **Avoided**: `np.where(condition, small_branch, large_branch)` 
     - Evaluates both branches, still triggers warnings
   - **Used**: Boolean indexing with separate computations
     - Only evaluates necessary expressions
     - Eliminates all RuntimeWarnings

3. **Mathematical Consistency**
   - Small $\kappa$: Uses first-order Taylor approximation $-\kappa/3$   - Large $\kappa$: Uses exact formula $\frac{1}{\kappa} - \frac{1}{\tanh(\kappa)}$   - Seamless transition preserves continuity and differentiability

4. **Edge Case Handling**
   - $\kappa = 0$: Returns exactly $0$ (no approximation needed)
   - Multiple zeros in batch: Each handled independently
   - Mixed arrays: Efficient vectorized computation

### Verification
The implementation is validated through comprehensive unit tests that verify:
- ✅ No RuntimeWarnings generated during backward pass
- ✅ Gradients remain finite for all input values
- ✅ Mathematical accuracy: $f(0) = 0$ exactly
- ✅ Correct Taylor approximation for small $|\kappa|$
- ✅ Proper handling of arrays containing multiple zeros

This approach ensures both mathematical correctness and numerical stability, making the von Mises-Fisher loss function robust for practical deep learning applications where $\kappa$ values may include zeros or near-zero elements.
