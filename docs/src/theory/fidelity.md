# Fidelity metrics

Fidelity functions live in
[`src/Physics/Objectives.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Physics/Objectives.jl).
Pulsar exposes a typed algebra of metrics: every metric is a singleton
struct subtyping `AbstractFidelityMetric`, so the compiler specialises on
the metric without runtime symbol comparisons. A legacy `type=:symbol`
keyword API delegates to the same dispatch.

## Pure-state metrics

For pure-state-to-pure-state transfer with overlap
`z = ⟨ψ_target | ψ(T)⟩`:

| Metric singleton  | Formula            | Range       | Symbol      |
|-------------------|--------------------|-------------|-------------|
| `RealOverlap`     | `Re z`             | `[-1, 1]`   | `:real`     |
| `SquaredOverlap` | `|z|²`             | `[0, 1]`    | `:square`   |
| `ModulusOverlap` | `|z|`              | `[0, 1]`    | `:modulus`  |

Type-dispatched call:

```julia
F = state_fidelity(ψ_target, ψ_final, SquaredOverlap())
F = state_fidelity(ψ_target, ψ_final; type = :square)   # legacy form
```

Exported singletons are `REAL_OVERLAP`, `SQUARED_OVERLAP`, `MODULUS_OVERLAP`.
The default for `compute_fidelity(U, target::QuantumTarget)` on a state
target is `SquaredOverlap`.

## Density-matrix (open-system) metrics

For mixed states / Lindblad evolution:

| Metric singleton    | Formula                                   | Symbol        |
|---------------------|-------------------------------------------|---------------|
| `UhlmannFidelity`   | `(Tr √(√ρ σ √ρ))²`                       | `:dm_uhlmann` |
| `LinearDMFidelity`  | `Re Tr(ρ† σ)` (linear approximation)      | `:dm_linear`  |

```julia
F = state_fidelity(ρ, σ, UHLMANN_FIDELITY)
F = dm_fidelity(ρ, σ)                       # alias for Uhlmann
```

`UhlmannFidelity` is computed via the spectral square root of `ρ`. Pulsar's
Lindblad GRAPE (see [Propagators](propagators.md)) builds the trajectory
`ρ(T)` via Liouville-space propagation and feeds the result back through
this dispatch.

## Gate (unitary) metrics

For unitary synthesis on a `d`-dimensional system, with overlap
`τ = Tr(U_target† U) / d`:

| Metric singleton   | Formula                          | Symbol        |
|--------------------|----------------------------------|---------------|
| `NormalizedGate`   | `|Tr(U_target† U)|² / d²` = `|τ|²` | `:normalized` |
| `RealGate`         | `Re τ`                            | `:real`       |
| `AverageGate`      | `(d · |τ|² + 1) / (d + 1)` (Haar) | `:average`    |

```julia
F = gate_fidelity(U, U_target, NORMALIZED_GATE)
F = gate_fidelity(U, U_target; type = :average)
```

`NormalizedGate` is the default for `compute_fidelity` on a unitary target.

## Subspace and process metrics

For leakage-aware optimization on multi-level platforms:

```julia
m = EssentialSubspaceGate([1, 2])             # qubit subspace inside a 3-level transmon
F = gate_fidelity(U_full, U_target, m)        # |Tr(Π U†_t U Π)|² / |Π|²
```

For `optim_target = "gate, file"`-style cooperative goals (a gate shape
**and** a specific state preparation in one cost function):

```julia
m = CooperativeTargetFidelity(NormalizedGate(), SquaredOverlap();
                              α = 0.6, β = 0.4)
F = cooperative_fidelity(U, U_target, ψ_init, ψ_target, m)
```

For process tomography reconstructed from basis-state transfers:

```julia
F = gate_fidelity(U, U_target, ProcessTomographyFidelity(d))
```

## Krotov co-state boundary

Krotov's algorithm needs `χ(T) = -∂J_T/∂⟨ψ(T)|`. Pulsar derives this
automatically per metric:

| Metric             | `χ(T)`                                |
|--------------------|---------------------------------------|
| `RealOverlap`     | `ψ_target`                            |
| `ModulusOverlap`  | `ψ_target` (up to a phase)            |
| `SquaredOverlap` | `⟨ψ_target | ψ(T)⟩ · ψ_target`        |
| `RealGate`         | `U_target`                            |
| `NormalizedGate`   | `(Tr(U_target† U(T)) / d²) · U_target` |

`make_chi(metric, target, ψ_T)` returns the boundary; `krotov_optimize` calls
this when no `chi_constructor` is supplied.

## GRAPE gradient prefactor

The chain-rule coefficient `∂F/∂w[c, k]` is also dispatched on the metric:

| Metric             | `fidelity_grad_prefactor(z, inner, dt_pwr, m)`     |
|--------------------|----------------------------------------------------|
| `RealOverlap`     | `dt_pwr · Im(inner)`                               |
| `SquaredOverlap` | `2 · dt_pwr · Im(z̄ · inner)`                      |
| `ModulusOverlap` | `dt_pwr · Im(inner / |z|)`  (zero when `|z| < ε`) |

Density-matrix metrics use a Liouville-space adjoint gradient (see
[`Physics/Lindblad.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Physics/Lindblad.jl))
rather than this Hilbert-space prefactor.

## Ensemble fidelity

For a parametric ensemble `{ψ_finals_i, ψ_targs_i}` with optional weights:

```julia
F = ensemble_fidelity(ψ_finals, ψ_targs;
                       weights  = nothing,
                       mode     = :mean,        # :mean, :weighted, :minimax
                       fid_type = :real)
```

Modes:

- `:mean` — `(1/N) Σ Fᵢ` or `Σ wᵢ Fᵢ` if weights are given
- `:weighted` — `Σ wᵢ Fᵢ` (requires weights)
- `:minimax` — `min Fᵢ` (worst-case / robust guarantee)

This is the foundation for [robust optimization](../algorithms/robust.md).

## Band-selective fidelity

For pulses with frequency-dependent passband / stopband requirements (see
[NMR](../domains/nmr.md), [MRI](../domains/mri.md)):

```julia
band_weights = [
    BandWeight(-500.0,  1.0),     # +1 weight at -500 Hz (passband)
    BandWeight(   0.0,  1.0),     # +1 weight on resonance
    BandWeight(+500.0,  1.0),
    BandWeight(+2500.0, -0.5),    # -0.5 weight at 2.5 kHz (stopband)
]

F = band_selective_fidelity(sys, ctrl, target, band_weights)
```

`band_selective_gradient` provides the matching analytic gradient. Both live
in
[`src/Physics/MRPhysics.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Physics/MRPhysics.jl).

## Custom metrics

To introduce a non-standard objective, subtype `AbstractFidelityMetric` and
add methods for `state_fidelity` (or `gate_fidelity`) and, if you want
analytic gradients, `fidelity_grad_prefactor`. The optimizer machinery
specialises automatically.
