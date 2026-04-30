# Penalty functions

Optimizers in PULSAR maximize fidelity subject to soft constraints expressed
as additive penalties on the control waveform. Penalties live in
[`src/Physics/Penalties.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Physics/Penalties.jl)
and subtype `AbstractPenalty`.

## Built-in penalties

All five live in
[`src/Physics/Penalties.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Physics/Penalties.jl)
and subtype `AbstractPenalty`. The penalty constructor takes the weight as
the **first positional** argument.

| Type | Constructor | Formula |
|---|---|---|
| `NormSquarePenalty`        | `NormSquarePenalty(weight)`                        | `weight · Σⱼ,ₖ wⱼ[k]²` |
| `SpilloutPenalty`          | `SpilloutPenalty(weight; l_bound, u_bound)`        | `weight · Σ max(0, w−u_b)² + max(0, l_b−w)²` |
| `AmplitudeSpilloutPenalty` | `AmplitudeSpilloutPenalty(weight; u_bound)`        | `weight · Σₖ max(0, ‖w[:, k]‖ − u_b)²` |
| `SmoothnessPenalty`        | `SmoothnessPenalty(weight)`                        | `weight · Σⱼ,ₖ (wⱼ[k+1] − wⱼ[k])²` |
| `EnergyPenalty`            | `EnergyPenalty(weight; dt = nothing)`              | `weight · Σⱼ,ₖ wⱼ[k]² · dt[k]` |

Domain-specific penalties live alongside the physics they belong to:

- `Physics/MRPhysics.jl` — `sar_penalty`, `slew_rate_penalty`,
  `band_selective_*`
- `Application/QuantumComputing/...` — leakage, Mølmer–Sørensen,
  filter-function and noise-spectrum penalties

## Usage pattern

Most optimization entry points accept a `penalty_fns` keyword that takes a
vector of `w → ℝ` callables. Build the closure list with
`make_penalty_fns` (and the matching gradient list with
`make_penalty_grad_fns`):

```julia
penalties = AbstractPenalty[
    SpilloutPenalty(1e-3; l_bound = -RF_max, u_bound = RF_max),
    SmoothnessPenalty(1e-4),
    EnergyPenalty(1e-6),
]

result = grape_optimize(sys, target, ctrl;
                        config           = config,
                        penalty_fns      = make_penalty_fns(penalties),
                        penalty_grad_fns = make_penalty_grad_fns(penalties))
```

PULSAR's analytic-gradient kernels also support analytic penalty gradients —
each built-in penalty provides a `∇p(w)` method that the GRAPE backward pass
consumes alongside the fidelity gradient.

## Tuning weights

Penalties operate on the same scale as the *infidelity* `1 − F`, not `F`.
Practical starting points:

- `SpilloutPenalty`: `1e-3` to `1e-2` for hard amplitude clipping
- `SmoothnessPenalty`: `1e-5` to `1e-3` depending on `dt` and `n_steps`
- `EnergyPenalty`: `1e-6` to `1e-4` to discourage runaway amplitude

Inspect the residual penalty contributions after optimization
(`result.penalty_history` if available) to recalibrate.

## Custom penalties

Subtype `AbstractPenalty` and implement:

```julia
struct MyPenalty <: AbstractPenalty
    weight::Float64
    # …
end

function (p::MyPenalty)(w::AbstractMatrix)
    # return scalar penalty value
end

function gradient(p::MyPenalty, w::AbstractMatrix)
    # return same-shape gradient ∂p/∂w
end
```

The optimizer will pick up the new type via the same `penalty_fns` interface.
