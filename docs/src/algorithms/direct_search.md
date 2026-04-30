# Direct search

Derivative-free methods that explore the parameter space without gradients.
Useful when the objective is non-smooth, noisy, or only available through a
black-box.

## Simplex methods

| Function | Source | Notes |
|---|---|---|
| `nelder_mead_optimize` | [`src/Optimization/Direct/SimplexSearch.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Optimization/Direct/SimplexSearch.jl) | Generic Nelder–Mead |
| `nelder_mead_optimize` (QOC dispatch) | [`src/Optimization/DirectSearchMethods.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Optimization/DirectSearchMethods.jl) | QOC-tailored signature |

Robust on small problems (≲ 50 parameters). Configurable via `NelderMeadConfig`.

## Pattern-search methods

| Function | Notes |
|---|---|
| `hooke_jeeves_optimize` | Direction-of-descent + step-shrink |
| `compass_search_optimize` | Coordinate-aligned probing |
| `powell_dirset_optimize` | Powell's direction-set method |

Source: [`src/Optimization/Direct/PatternSearch.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Optimization/Direct/PatternSearch.jl).

## Quadratic-model trust-region

Derivative-free methods that build local quadratic models. Ported from
Powell's NEWUOA / BOBYQA / UOBYQA family.

| Function | Bounds | Constraints |
|---|---|---|
| `uobyqa_optimize` | None | None |
| `newuoa_optimize` | None | None |
| `bobyqa_optimize` | Box | None |
| `cobyla_optimize` | None | Linear+nonlinear |
| `lincoa_optimize` | None | Linear |

Source: [`src/Optimization/Direct/QuadraticModels.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Optimization/Direct/QuadraticModels.jl)
and [`src/Optimization/Direct/ConstrainedDirect.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Optimization/Direct/ConstrainedDirect.jl).

## When to use

- **Non-smooth or noisy objective** (e.g., experimental feedback in the loop)
- **Few parameters** (≲ 100). Direct-search scales poorly with dimension.
- **Black-box analytic pulses** (CRAB-style basis fits with few coefficients)

For high-dimensional smooth problems prefer GRAPE / L-BFGS / GRAPE-CG.

## Example — Nelder–Mead on a 1-qubit problem

```julia
config = NelderMeadConfig(max_iter = 2000, ftol = 1e-7)
result = nelder_mead_optimize(sys, target, ctrl; config=config)
```

## Composing with global search

A common pattern: run a metaheuristic globally (PSO, basin hopping) to find a
good basin, then polish with NEWUOA or a Newton method. See
[Metaheuristic](metaheuristic.md).
