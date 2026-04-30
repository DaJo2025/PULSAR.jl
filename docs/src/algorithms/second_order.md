# Second-order methods

Second-order methods exploit curvature information (Hessian or quasi-Hessian)
for faster local convergence near a minimum. PULSAR ships generic and
QOC-specialized variants.

## BFGS / L-BFGS

Quasi-Newton methods that build a Hessian approximation from successive
gradient evaluations.

| Function | Source |
|---|---|
| `bfgs_optimize` | [`src/Optimization/SecondOrder/SecondOrderMethods.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/SecondOrder/SecondOrderMethods.jl) |
| `lbfgs_optimize` | same |
| `lbfgsb_optimize` | [`src/Optimization/Gradient/Generic/QuasiNewton.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Gradient/Generic/QuasiNewton.jl) |

`BFGSConfig` / `LBFGSConfig` carry the standard knobs (memory length, line-search
type, convergence tolerances). L-BFGS-B adds box constraints `[l, u]` per
parameter — ideal for hardware amplitude limits.

## Newton and trust-region Newton

| Function | Notes |
|---|---|
| `newton_optimize` | Pure Newton (full Hessian, factorization step) |
| `gauss_newton_optimize` | Gauss–Newton for least-squares-like objectives |
| `lm_optimize` | Levenberg–Marquardt |
| `trust_region_newton_optimize` | Trust-region Newton with subproblem solver |

Newton-class methods are best with moderate parameter counts (≲ a few hundred)
since they perform an `O(n²)`–`O(n³)` step per iteration.

## High-order optimal control

For QOC-specialized second-order methods (e.g., adaptive
trust-region tailored to the GRAPE landscape):

| Function | Source |
|---|---|
| `oc_trust_region_newton_optimize` | [`src/Optimization/Gradient/QOC/HighOrderOC.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Gradient/QOC/HighOrderOC.jl) |
| `oc_semismooth_newton_optimize` | same |

These wrap the GRAPE gradient with QOC-specific Hessian models.

## Choosing between them

- **Default**: `lbfgs_optimize` — robust, low memory, no Hessian factorization
- **Tight bounds**: `lbfgsb_optimize`
- **Many parameters, smooth**: GRAPE-CG (`grape_cg_optimize`) often beats
  standard L-BFGS for `n_steps × n_controls > 10⁴`
- **Final polishing**: `trust_region_newton_optimize` after a global search

## Configuration example — L-BFGS

```julia
config = LBFGSConfig(
    max_iter           = 200,
    memory_size        = 10,
    convergence_tol    = 1e-10,
    gradient_tol       = 1e-7,
    line_search_method = "wolfe",   # or "backtracking"
    verbose            = false,
)
result = lbfgs_optimize(sys, target, ctrl; config = config,
                         penalty_fns = penalty_fns)
```

## L-BFGS-B with hardware bounds

`lbfgsb_optimize` and the QOC-specialised `grape_lbfgsb_optimize` are
*function-form* optimisers — they accept a callable cost `f`, an in-place
gradient `grad!`, and an initial parameter vector `θ0`, with the box
bounds passed as keyword arguments:

```julia
n_p   = n_controls * n_steps                   # flattened parameter count
lower = -RF_max .* ones(n_p)
upper =  RF_max .* ones(n_p)

result = grape_lbfgsb_optimize(
    f, grad!, θ0;
    lower    = lower,
    upper    = upper,
    memory   = 10,
    max_iter = 200,
    tol      = 1e-7,
    verbose  = false,
)
```

The extra-thin `lbfgsb_optimize` wrapper accepts `use_native = true` to
delegate to the `LBFGSB.jl` Fortran backend; install it with
`pkg> add LBFGSB`. With `use_native = false` (the default if `LBFGSB.jl`
is not loaded) PULSAR uses its own pure-Julia bound-projected L-BFGS.

In MR / QC application code, the higher-level convenience wrapper
`grape_lbfgsb_optimize` is invoked indirectly through
`optimcon(::AbstractMRControl)`; see [NMR](../domains/nmr.md) and the QC
[platforms](../domains/qc_platforms.md) page.
