# Constrained optimization

When pulses must satisfy hard physical limits (peak amplitude, total energy,
peak amplitude / bandwidth, custom relations), use `constrained_optimize`
from
[`src/Optimization/Constrained/ConstrainedOpt.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Constrained/ConstrainedOpt.jl).

## Built-in constraint types

All constraint constructors below take **positional** arguments and validate
inputs in their inner constructor (e.g. `lower < upper`, `P_max > 0`).

| Type | Constructor | Limit |
|---|---|---|
| `BoundConstraint`     | `BoundConstraint(lower, upper [, control_indices])` | `lower ≤ uⱼ[k] ≤ upper` per element |
| `PowerConstraint`     | `PowerConstraint(P_max)`                            | `Σⱼ,ₖ uⱼ[k]² ≤ P_max` |
| `BandwidthConstraint` | `BandwidthConstraint(BW_max)` (or legacy `(f_max_hz, dt)`) | `maxⱼ,ₖ |uⱼ[k]| ≤ BW_max` |
| `EnergyConstraint`    | `EnergyConstraint(E_max)`                           | `dt · Σⱼ,ₖ uⱼ[k]² ≤ E_max` |
| `CustomConstraint`    | `CustomConstraint(name::String, c_fn, ∇c_fn)`       | User-supplied `c(u) ≤ 0` |

`BoundConstraint`'s third argument `control_indices` selects which control
channels the bound applies to; an empty `Int[]` (the default) means *all*
channels.

`BandwidthConstraint(BW_max)` is a peak-amplitude clamp; the legacy two-arg
form `BandwidthConstraint(f_max_hz, dt)` stores `f_max_hz · dt` as a coarse
spectral cap.

`CustomConstraint` requires both an evaluator
`c_fn(u::Matrix{Float64}) -> Float64` and a gradient
`∇c_fn(u::Matrix{Float64}) -> Matrix{Float64}` of the same shape as `u`.

## Configuration

`ConstrainedConfig` controls the *outer* loop (penalty growth, base
optimizer, convergence). Constraints themselves are passed as a separate
positional `Vector{<:AbstractConstraint}` to `constrained_optimize`.

```julia
config = ConstrainedConfig(
    base_method       = "lbfgs",                  # "grape", "bfgs", "lbfgs"
    constraint_method = "augmented_lagrangian",   # "penalty", "augmented_lagrangian", "projection"
    max_iter          = 50,
    penalty_initial   = 1.0,
    penalty_growth    = 10.0,
    violation_tol     = 1e-6,
    verbose           = false,
)

constraints = AbstractConstraint[
    BoundConstraint(-RF_max, RF_max),
    BandwidthConstraint(50e3),
    EnergyConstraint(1e3),
]

result = constrained_optimize(sys, target, ctrl, constraints; config = config)
```

The optimizer enforces the constraints via the strategy chosen in
`constraint_method`:

- `"penalty"` — quadratic penalty with `λ` growing by `penalty_growth` each outer iteration
- `"augmented_lagrangian"` — Lagrange multipliers + quadratic penalty (default)
- `"projection"` — project onto the feasible set after each base-method step

## Soft vs hard constraints

For *soft* constraints (penalty-style enforcement that is part of the
fidelity gradient), use `SpilloutPenalty` / `EnergyPenalty` /
`SmoothnessPenalty` from
[Penalties](../theory/penalties.md) together with any unconstrained
optimizer.

Use **hard** `constrained_optimize` constraints when:

- the limit reflects a hardware bound that must never be exceeded
- the optimizer would otherwise push the solution into the violation region

Use **soft** penalties when:

- the limit is "preferred but not strict" (smoothness, energy budget)
- you want gradient-friendly behavior across the boundary

## Bound-only problems — L-BFGS-B shortcut

For pure box constraints, `lbfgsb_optimize` or the QOC-specific
`grape_lbfgsb_optimize` are typically faster and simpler than the general
`constrained_optimize`. See [Second-order methods](second_order.md).
