# QOC-specific algorithms

Quantum optimal control offers a family of algorithms that exploit the
structure of unitary / Liouvillian dynamics rather than treating fidelity as a
generic objective. PULSAR ships several.

## Krotov

Iterative method with a guaranteed monotonic increase of fidelity in the
continuous limit. Source: [`src/Optimization/Gradient/QOC/Krotov.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Gradient/QOC/Krotov.jl).

| Function | Notes |
|---|---|
| `krotov_optimize` | First-order Krotov |
| `krotov_second_order_optimize` | Second-order Krotov with curvature term |

Krotov is excellent at *refining* an already-good guess but slow to escape a
poor basin.

## GOAT and GROUP

Basis-function (analytic ansatz) approaches that parameterize each control by
a small number of coefficients in a chosen basis (Fourier, Gaussian, Slepian,
…). Source: [`src/Optimization/Gradient/QOC/BasisMethods.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Gradient/QOC/BasisMethods.jl).

| Function | Notes |
|---|---|
| `goat_optimize` | GOAT — gradient optimization of analytic terms |
| `group_optimize` | GROUP — generalized basis-method search |

These dramatically reduce the parameter count (10–50 instead of `n_steps × n_c`)
and produce smoother pulses.

## CRAB

Chopped Random Basis. A randomized Fourier-basis ansatz combined with a
direct-search outer loop (typically Nelder–Mead). Source:
[`src/Optimization/Gradient/QOC/CRAB.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Gradient/QOC/CRAB.jl).

CRAB is gradient-free in its outer loop, which is useful when the fidelity
landscape is too rugged for analytic gradients.

## T-GRAPE

Time-resolution-adaptive GRAPE: jointly optimizes amplitudes and time-step
durations. Source: [`src/Optimization/Gradient/QOC/TGRAPE.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Gradient/QOC/TGRAPE.jl).

T-GRAPE adds a "dt-block" gradient on top of standard GRAPE; the
time-evolution invariant is `Q[k]† · U_k · P[k] = U_total`.

## GRAPE-family entry points

For convenience, [`src/Optimization/Gradient/QOC/GRAPEFamily.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Gradient/QOC/GRAPEFamily.jl)
bundles dispatchers for the most common combinations:

| Function | Underlying method |
|---|---|
| `grape_optimize` | GRAPE-GA |
| `grape_cg_optimize` | GRAPE-CG |
| `grape_lbfgsb_optimize` | GRAPE-L-BFGS-B |

## Choosing among QOC-specific methods

- **Default smooth problem**: GRAPE family
- **Polish a near-optimum**: Krotov (esp. second-order)
- **Few smooth control parameters**: GOAT or GROUP
- **Rugged landscape, gradient unreliable**: CRAB
- **Joint amplitude+time optimization**: T-GRAPE
