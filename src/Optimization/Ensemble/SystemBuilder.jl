# ============================================================================
# Optimization/Ensemble/SystemBuilder.jl
# ============================================================================
# Builder that turns a fixed `Vector{<:AbstractQuantumSystem}` into an
# `EnsembleObjective`. Uses only Physics-layer primitives, so it loads early
# (before GRAPE.jl) and is available for `grape_optimize_ensemble` to delegate
# through.
# ============================================================================

using LinearAlgebra

@inline _reshape_theta(θ::Vector{Float64}, nc::Int, nt::Int) = reshape(θ, nc, nt)
@inline _reshape_theta(θ::AbstractVector{<:Real}, nc::Int, nt::Int) =
    reshape(collect(Float64, θ), nc, nt)

@inline function _vec_to_ctrl(θ::AbstractVector{<:Real},
                               dt::Real, n_timesteps::Int, n_controls::Int)
    amps = _reshape_theta(θ, n_controls, n_timesteps)
    return ControlSequence(amps, Float64(dt), Float64(dt) * n_timesteps, n_timesteps)
end

"""
    build_ensemble_from_systems(systems, target, controls;
                                 aggregator = :mean,
                                 cvar_alpha = 0.2) -> EnsembleObjective

Build an [`EnsembleObjective`](@ref) whose members are indexed by a fixed
`Vector{<:AbstractQuantumSystem}` sharing a common `target` and control layout.

Per-sample closures call [`compute_fidelity`](@ref) and
[`compute_grape_gradient`](@ref) on each `systems[i]` with a `ControlSequence`
rebuilt from the flat parameter vector. The closure shape mirrors
`grape_optimize_ensemble`'s pre-refactor `_ens_F` / `_ens_G` (see
`src/Optimization/GRAPE.jl` lines 981-995) — so swapping in this builder is a
numerical no-op for the `:mean` aggregator.

# Aggregators
- `:mean` — average over all `systems`
- `:worst_case` — objective = worst-performing member's fidelity
- `:cvar` — tail mean over the worst `cvar_alpha · length(systems)` members

# Example
```julia
obj       = build_ensemble_from_systems(systems, target, ctrl; aggregator=:worst_case)
f, grad!  = ensemble_wrap(obj)
θ_opt, F, _ = lbfgs_optimize(f, grad!, vec(ctrl.controls); max_iter=200)
```
"""
function build_ensemble_from_systems(systems::Vector{<:AbstractQuantumSystem},
                                      target::QuantumTarget,
                                      controls::ControlSequence;
                                      aggregator::Symbol = :mean,
                                      cvar_alpha::Real   = 0.2)
    n_sys = length(systems)
    n_sys > 0 || throw(ArgumentError("systems must be non-empty"))

    dt          = controls.dt
    n_timesteps = controls.n_timesteps
    n_controls  = size(controls.controls, 1)

    # Factory closures keep the per-sample closure type concrete across
    # iterations so the resulting Vectors are `Vector{typeof(_f)}` / `Vector{typeof(_g)}`
    # rather than `Vector{Function}` — eliminating jl_apply_generic dispatch in
    # EnsembleObjective's per-sample loop.
    _mk_f(s) = let s = s, dt = dt, n_timesteps = n_timesteps, n_controls = n_controls, target = target
        θ -> begin
            cs = _vec_to_ctrl(θ, dt, n_timesteps, n_controls)
            Float64(compute_fidelity(s, cs, target))
        end
    end
    _mk_g(s) = let s = s, dt = dt, n_timesteps = n_timesteps, n_controls = n_controls, target = target
        (gv, θ) -> begin
            cs = _vec_to_ctrl(θ, dt, n_timesteps, n_controls)
            G  = compute_grape_gradient(s, cs, target)
            copyto!(gv, vec(G))
            return gv
        end
    end
    f_samples    = [_mk_f(systems[i]) for i in 1:n_sys]
    grad_samples = [_mk_g(systems[i]) for i in 1:n_sys]

    return EnsembleObjective(f_samples;
                              grad_samples = grad_samples,
                              aggregator   = aggregator,
                              cvar_alpha   = Float64(cvar_alpha),
                              n_samples    = n_sys)
end
