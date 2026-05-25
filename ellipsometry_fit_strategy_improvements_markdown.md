# Ellipsometry Fit Strategy Improvements

## Overview

Current architecture is already close to a research-grade ellipsometry fitting framework. The main limitations are no longer the transfer matrix calculation itself, but rather the optimization and fitting strategy.

Major weaknesses:

- Single local optimizer
- Multi-modal parameter space
- Strong parameter correlations
- Residual definition not fully physics-aware
- Limited uncertainty analysis

Current workflow:

```python
config -> lmfit.minimize(LM) -> residual(Psi, Delta, T)
```

Typical implementation:

```python
lmfit.minimize(..., method=self.method)
```

The current approach is essentially a WVASE-like local optimization pipeline.

---

# 1. Use Global → Local Optimization

## Problem

The current fitting pipeline relies mainly on a local optimizer:

- Levenberg–Marquardt (LM)
- leastsq

This causes:

- strong dependence on initial guess
- trapping in local minima
- instability for correlated parameters
- false convergence to physically incorrect solutions

Especially severe for:

- metals
- multilayer films
- roughness models
- generalized oscillator models

Ellipsometry fitting landscapes are highly non-convex.

---

## Recommended Strategy

### Stage 1 — Global Search

Use a global optimizer to locate the correct basin:

Recommended methods:

- differential evolution
- basinhopping
- dual annealing

Example:

```python
lmfit.minimize(..., method='differential_evolution')
```

---

### Stage 2 — Local Refinement

After global search, refine using LM:

```python
lmfit.minimize(..., method='leastsq')
```

---

## Recommended Workflow

```text
Global optimizer
    ↓
Find approximate basin
    ↓
Local LM refinement
    ↓
Final covariance estimation
```

---

## Why This Matters

Single LM fitting is often insufficient because:

- thickness ↔ refractive index correlation
- oscillator amplitude ↔ broadening degeneracy
- phase ambiguity
- strong parameter coupling

Global → local optimization usually gives the largest robustness improvement.

---

# 2. Parameter Scaling

## Problem

Parameters can differ by orders of magnitude.

Examples:

```text
thickness = 300
A = 0.2
gamma = 4
epsilon_inf = 1
```

This causes:

- poor Hessian conditioning
- unstable Jacobian estimation
- slow convergence
- optimizer bias toward large-scale parameters

---

## Recommended Improvements

### Normalize Parameters

Example:

```python
x_scaled = (x - center) / range
```

---

### Use lmfit Scaling Controls

Examples:

```python
par.brute_step
par.scale
```

---

## Most Important Parameters to Scale

- thickness
- plasma frequency
- oscillator amplitude
- broadening parameters

---

# 3. Oscillator Correlation Control

## Problem

Oscillator models are highly degenerate.

For example:

```text
Lorentz A ↑
gamma ↑
```

can produce nearly identical dielectric functions.

This leads to:

- overfitting
- non-physical solutions
- unstable covariance matrices
- parameter explosion

---

## Recommended Solutions

### Regularization

Introduce smoothness penalties:

```text
λ * Σ(d²ε/dE²)²
```

---

### Sparsity Constraints

Discourage excessive oscillators.

Example problem:

```text
10 oscillators fitting measurement noise
```

Possible approaches:

- L2 regularization
- L1 sparsity penalty
- minimum oscillator count constraints

---

# 4. Robust Loss Functions

## Current Situation

Current residuals are standard least squares:

```text
χ² minimization
```

---

## Problem

Experimental ellipsometry data often contains:

- backside reflections
- CCD glitches
- depolarization artifacts
- bad Δ jumps
- isolated outliers

Pure χ² fitting allows single bad points to dominate the optimization.

---

## Recommended Robust Losses

### soft_l1

### Huber loss

Example:

```python
least_squares(..., loss='soft_l1')
```

---

## Benefits

- improved robustness
- reduced sensitivity to artifacts
- smoother convergence
- more stable multilayer fitting

---

# 5. Fit sinΔ / cosΔ Instead of Δ Directly

## Problem

Phase wrapping:

```text
Δ = 0° ≈ 360°
```

Direct Δ residuals are topologically incorrect.

---

## Current Residual

```python
r_delta = delta_fit - delta_meas
```

---

## Recommended Residual

```python
r1 = cos(delta_fit) - cos(delta_meas)
r2 = sin(delta_fit) - sin(delta_meas)
```

---

## Benefits

- avoids phase discontinuities
- smoother optimization landscape
- improved convergence near wrapping boundaries
- physically consistent phase treatment

---

# 6. Physics-Aware Weighting

## Current Situation

Residual weighting mainly uses:

```text
sigma_psi
sigma_delta
```

---

## Problem

Ellipsometry sensitivity strongly depends on:

- angle of incidence
- wavelength
- Brewster regions
- material resonances

Some spectral regions carry much more information than others.

---

## Recommended Improvements

### Sensitivity Weighting

Use Jacobian-based weights:

```text
w ~ |∂Ψ/∂p|
```

---

### Fisher Information Weighting

Weight data according to parameter sensitivity.

---

## Benefits

Automatically emphasizes:

- thickness-sensitive regions
- interband transitions
- resonance structures
- Brewster-angle information

---

# 7. Improve Point-by-Point Inversion

## Current Situation

Pointwise inversion methods:

- e1_only
- e2_only

already resemble WVASE-like approaches.

---

## Problem

Point-by-point inversion is often:

- noisy
- unstable near singular regions
- physically inconsistent

---

## Recommended Improvements

### KK-Constrained Inversion

Represent ε₂ using:

- splines
- B-splines
- smooth basis functions

Then recover ε₁ using:

- Kramers–Kronig transform

---

## Benefits

- smoother dielectric functions
- physically causal optical constants
- reduced noise amplification
- improved stability

---

# 8. Uncertainty and Covariance Analysis

## Current Situation

Only standard errors are estimated:

```text
stderr
```

---

## Problem

Highly correlated fits invalidate simple covariance estimates.

Common issues:

- underestimated uncertainties
- misleading confidence intervals
- hidden parameter degeneracies

---

## Recommended Improvements

### MCMC Sampling

Example:

```python
lmfit.Minimizer.emcee()
```

---

## Benefits

Visualize:

- thickness–n correlation
- oscillator degeneracy
- parameter uncertainty distributions
- multimodal posterior structure

---

# 9. Fit Scheduling (Very Important)

## Problem

Simultaneously varying all parameters is unstable.

Real ellipsometry fitting usually proceeds incrementally.

---

## Recommended Strategy

### Step 1

Fit only:

```text
thickness
```

with optical constants fixed.

---

### Step 2

Enable:

```text
Cauchy B, C
```

---

### Step 3

Enable oscillator parameters.

---

### Step 4

Final global refinement.

---

## Suggested YAML Interface

```yaml
fit_schedule:
  - vary: [thickness]
  - vary: [cauchy.B, cauchy.C]
  - vary: [lorentz[*].A]
```

---

## Benefits

- greatly improved convergence
- reduced parameter degeneracy
- easier debugging
- more physically guided fitting

---

# 10. Bayesian / Probabilistic Fitting

## Key Insight

Ellipsometry is fundamentally an inverse problem.

The true goal is:

```text
P(parameters | data)
```

rather than:

```text
single best-fit parameters
```

---

## Recommended Methods

- MCMC
- nested sampling
- Bayesian inference
- probabilistic modeling

---

## Benefits

- uncertainty quantification
- model comparison
- multimodal parameter distributions
- physically interpretable confidence intervals

---

# Highest Priority Improvements

## Priority 1 — Global → Local Optimization

```text
Differential Evolution
→ LM refinement
```

Largest robustness improvement.

---

## Priority 2 — Fit Scheduling

Incrementally enable parameters.

---

## Priority 3 — Robust Residuals

Use:

```text
sinΔ/cosΔ residuals
+ Huber or soft_l1 loss
```

---

# Additional Engineering Improvements

## Cache Optical Constants

Avoid repeated:

```python
epsilon()
```

Use memoization:

```python
(material parameter hash, wavelength)
```

---

## Parallel Jacobian Evaluation

Different angles are independent.

Good candidates:

- multiprocessing
- vectorization
- numba
- parallel finite differences

---

## Parameter Tying

Examples:

- shared roughness
- shared oscillator widths
- constrained amplitudes

Use lmfit expressions:

```python
expr="..."
```

---

# Overall Assessment

Current status:

```text
Research-grade prototype
```

Main limitation:

```text
Optimization strategy still relies heavily on a single local optimizer
```

Most impactful improvement:

```text
Global optimization + staged fitting
```

