# GRAPE family

GRAPE (**Gradient Ascent Pulse Engineering**, Khaneja et al., *J. Magn. Reson.*
**172** (2005) 296) is PULSAR's reference first-order algorithm. The core
update is gradient ascent on the fidelity `F(w)` with respect to the
piecewise-constant control waveform `w[c, k]`.

## Variants

| Function | Method | Notes |
|---|---|---|
| `grape_optimize` | First-order GA with optional adaptive step | Default entry |
| `grape_cg_optimize` | Conjugate-gradient GRAPE | Faster on smooth landscapes |
| `grape_lbfgsb_optimize` | L-BFGS-B GRAPE (bound-constrained) | Native amplitude limits |

All three share the same `compute_grape_gradient` analytic kernel but differ
in how they consume the gradient.

## Configuration — `GRAPEConfig`

```julia
config = GRAPEConfig(
    max_iter          = 500,
    step_size         = 0.05,      # initial step
    adapt_step_size   = true,      # backtracking + extension
    min_step_size     = 1e-8,
    max_step_size     = 2.0,
    convergence_tol   = 1e-9,      # |ΔF| stop
    gradient_norm_tol = 1e-7,      # ‖∇F‖ stop
    verbose           = true,
    print_interval    = 100,
    callback          = nothing,   # iteration callback
)
```

## Result — `OptimizationResult`

```julia
result.fidelity              # Float64
result.controls              # [n_controls × n_steps]   (transposed vs ControlSequence)
result.fidelity_history      # Vector{Float64}
result.gradient_norm_history # Vector{Float64}
result.n_iterations          # Int
result.converged             # Bool
result.termination_reason    # Symbol / String
```

## Re-simulation rule

Always independently verify the result:

```julia
ctrl_opt = ControlSequence(Matrix(result.controls'), ctrl.dt, ctrl.n_steps)
F_check  = compute_fidelity(sys, ctrl_opt, target)
@assert abs(F_check - result.fidelity) < 1e-6
```

## Running ensemble GRAPE

For robust pulses over a parametric family, use `grape_optimize_ensemble`
(or the higher-level `robust_optimize`). Both rely on `compute_grape_gradient`
applied member-by-member with mean / worst-case aggregation.

## Penalties

GRAPE accepts a `penalty_fns = [w -> p(w), …]` keyword that adds a soft
penalty term to the objective. See [Penalties](../theory/penalties.md).

## When to use GRAPE vs alternatives

- **GRAPE / GRAPE-CG** — smooth landscapes, gradient-friendly objectives,
  many control parameters
- **L-BFGS-B GRAPE** — when amplitude bounds are hard physical limits
- **Krotov** — better convergence near already-good solutions, monotonic
  ascent guaranteed
- **CMA-ES / PSO** — non-smooth or rugged objectives, low parameter count
- **GOAT / CRAB** — analytic basis-function ansatz with few parameters

See [Algorithm selection](../advanced/uq_sensitivity.md) and the
[`recommend_optimizer`](../api/runtime.md) helper for automatic suggestions.
