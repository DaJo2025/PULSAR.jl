"""
    RobustOptimization.jl

Robust quantum control optimization for Pulsar.jl
(Pulse Design Library for Spin Control Algorithms and Rollout).

Provides sample-based robust optimization against:
- Parametric uncertainty: system Hamiltonian parameters are uncertain
- Quantum noise: decoherence and gate noise modelled via Kraus operators
- Parameter drift: slow time-varying systematic shifts in the Hamiltonian

Robustness measures supported:
- `"worst_case"` : minimize the worst-case fidelity over all perturbations
- `"mean"`       : maximize the average fidelity over sampled perturbations
- `"cvar"`       : Conditional Value-at-Risk (tail average of poor outcomes)
"""

using LinearAlgebra
using Random
using Statistics
using Printf

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

"""
    RobustConfig

Configuration for robust quantum control optimization.

# Fields
- `uncertainty_type::String`: type of uncertainty to handle:
  `"parametric"` (Hamiltonian parameter spread), `"noise"` (Kraus noise),
  or `"drift"` (slow systematic drift)
- `uncertainty_magnitude::Float64`: relative magnitude of perturbations
  (e.g. `0.05` for 5% parameter spread)
- `robustness_measure::String`: objective used to aggregate fidelity over the
  ensemble: `"worst_case"`, `"mean"`, or `"cvar"`
- `cvar_alpha::Float64`: tail fraction for CVaR (default `0.2` → worst 20%)
- `n_samples::Int`: number of perturbation samples drawn per outer iteration
- `base_method::String`: unconstrained optimizer for the inner step
  (`"grape"`, `"bfgs"`, `"lbfgs"`)
- `max_iter::Int`: maximum gradient steps
- `step_size::Float64`: initial gradient step size
- `convergence_tol::Float64`: stop when gradient norm < tol
- `verbose::Bool`: print iteration log

# Example
```julia
cfg = RobustConfig(
    uncertainty_type      = "parametric",
    uncertainty_magnitude = 0.05,
    robustness_measure    = "mean",
    n_samples             = 20,
    base_method           = "grape",
    max_iter              = 500,
    verbose               = true
)
```
"""
struct RobustConfig
    uncertainty_type::String
    uncertainty_magnitude::Float64
    robustness_measure::String
    cvar_alpha::Float64
    n_samples::Int
    base_method::String
    max_iter::Int
    step_size::Float64
    convergence_tol::Float64
    verbose::Bool
    seed::Int
    use_threads::Bool
    cache_propagators::Bool

    function RobustConfig(;
        uncertainty_type::String      = "parametric",
        uncertainty_magnitude::Float64 = 0.05,
        robustness_measure::String    = "mean",
        cvar_alpha::Float64           = 0.2,
        n_samples::Int                = 20,
        base_method::String           = "grape",
        max_iter::Int                 = 500,
        step_size::Float64            = 1e-3,
        convergence_tol::Float64      = 1e-6,
        verbose::Bool                 = true,
        seed::Int                     = 42,
        use_threads::Bool             = true,
        # Lesson 6: opt-in propagator cache for shared-Hamiltonian ensembles.
        # Off by default until profiling shows a hit-rate ≥ a few % on the
        # caller's workload. See `src/Computation/PropagatorCache.jl`.
        cache_propagators::Bool       = false,
    )
        uncertainty_type in ("parametric", "noise", "drift") ||
            throw(ArgumentError("uncertainty_type must be: parametric, noise, or drift"))
        robustness_measure in ("worst_case", "mean", "cvar") ||
            throw(ArgumentError("robustness_measure must be: worst_case, mean, or cvar"))
        uncertainty_magnitude >= 0 ||
            throw(ArgumentError("uncertainty_magnitude must be non-negative"))
        0 < cvar_alpha <= 1 ||
            throw(ArgumentError("cvar_alpha must be in (0, 1]"))
        n_samples > 0   || throw(ArgumentError("n_samples must be positive"))
        max_iter  > 0   || throw(ArgumentError("max_iter must be positive"))
        step_size > 0   || throw(ArgumentError("step_size must be positive"))
        new(uncertainty_type, uncertainty_magnitude, robustness_measure,
            cvar_alpha, n_samples, base_method, max_iter, step_size,
            convergence_tol, verbose, seed, use_threads, cache_propagators)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Perturbation sampling
# ──────────────────────────────────────────────────────────────────────────────

"""
    sample_parametric_perturbations(system, n_samples::Int,
                                     magnitude::Float64;
                                     rng::AbstractRNG = Random.default_rng()) -> Vector

Sample `n_samples` random Hermitian perturbation matrices ΔH such that
`‖ΔH‖_F ≤ magnitude · ‖H_drift‖_F`.

The perturbations are drawn as random Hermitian matrices:
    ΔH = (X + X†) / 2,   X ∈ ℂⁿˣⁿ  (entries ~ N(0,1) + i·N(0,1))
then normalized and scaled.

# Arguments
- `system`: quantum system (must have `H_drift` field or `drift_hamiltonian` property)
- `n_samples`: number of independent perturbation matrices to draw
- `magnitude`: relative perturbation magnitude
- `rng`: random number generator (for reproducibility)

# Returns
`Vector` of Hermitian perturbation matrices, each of size matching `H_drift`
"""
function sample_parametric_perturbations(system, n_samples::Int,
                                          magnitude::Float64;
                                          rng::AbstractRNG = Random.default_rng())
    H0  = _get_drift_hamiltonian(system)
    dim = size(H0, 1)
    H0_norm = norm(H0)
    H0_norm < 1e-14 && (H0_norm = 1.0)  # avoid division by zero for zero drift

    perturbations = Vector{Matrix{ComplexF64}}(undef, n_samples)
    for i in 1:n_samples
        X  = (randn(rng, dim, dim) .+ im .* randn(rng, dim, dim)) ./ sqrt(2.0)
        dH = (X .+ X') ./ 2.0             # make Hermitian
        sc = magnitude * H0_norm / max(norm(dH), 1e-14)
        perturbations[i] = dH .* sc
    end
    return perturbations
end

"""
    sample_drift_trajectories(system, n_samples::Int,
                               magnitude::Float64, n_timesteps::Int;
                               rng::AbstractRNG = Random.default_rng()) -> Vector

Sample `n_samples` slowly-varying drift trajectories. Each trajectory is a
`Vector{Matrix{ComplexF64}}` of length `n_timesteps` representing a smoothly
time-varying additive perturbation to the drift Hamiltonian.

The drift is modelled as a linear ramp:
    ΔH(t) = t/T · ΔH_final
where ΔH_final is drawn from the same distribution as parametric perturbations.
"""
function sample_drift_trajectories(system, n_samples::Int,
                                    magnitude::Float64, n_timesteps::Int;
                                    rng::AbstractRNG = Random.default_rng())
    perturb_finals = sample_parametric_perturbations(system, n_samples, magnitude; rng=rng)
    trajectories   = Vector{Vector{Matrix{ComplexF64}}}(undef, n_samples)

    for (s, dH_final) in enumerate(perturb_finals)
        traj = Vector{Matrix{ComplexF64}}(undef, n_timesteps)
        for k in 1:n_timesteps
            t_frac  = (k - 1) / max(1, n_timesteps - 1)
            traj[k] = t_frac .* dH_final
        end
        trajectories[s] = traj
    end
    return trajectories
end

# ──────────────────────────────────────────────────────────────────────────────
# Robustness metric computation
# ──────────────────────────────────────────────────────────────────────────────

"""
    robust_fidelity(system, controls::Matrix{Float64}, target,
                     perturbations::Vector,
                     robustness_measure::String;
                     cvar_alpha::Float64 = 0.2) -> Float64

Compute the robustness metric for `controls` over an ensemble of `perturbations`.

# Robustness measures
- `"worst_case"`: `min_{ΔH} F(u; H_0 + ΔH)` — maximize over the worst perturbation
- `"mean"`: `(1/N) ∑ F(u; H_0 + ΔH_i)` — maximize average fidelity
- `"cvar"`: mean of the worst `cvar_alpha` fraction of fidelities
  (Conditional Value-at-Risk at level `1-cvar_alpha`)

# Arguments
- `system`: nominal quantum system
- `controls`: control amplitudes `[n_controls × n_timesteps]`
- `target`: gate or state target
- `perturbations`: vector of perturbation objects (matrices or trajectories)
- `robustness_measure`: `"worst_case"`, `"mean"`, or `"cvar"`
- `cvar_alpha`: tail fraction for CVaR (default 0.2)
"""
function robust_fidelity(system, controls::Matrix{Float64}, target,
                          perturbations::Vector,
                          robustness_measure::String;
                          cvar_alpha::Float64 = 0.2)::Float64

    n  = length(perturbations)
    fv = Vector{Float64}(undef, n)

    # Lesson 3: independent per-perturbation fidelity evaluations are now
    # threaded with BLAS-thread guard via `@threadsif`.
    @threadsif true for i in 1:n
        fv[i] = _fidelity_with_perturbation(system, controls, target, perturbations[i])
    end

    # Delegate to the shared aggregator surgery (Ensemble/EnsembleObjective.jl).
    return _agg_value(fv, _symbolize_measure(robustness_measure), cvar_alpha)
end

# String → Symbol mapping shared by robust_fidelity / _robust_gradient
@inline function _symbolize_measure(m::String)::Symbol
    m == "mean"       && return :mean
    m == "worst_case" && return :worst_case
    m == "cvar"       && return :cvar
    throw(ArgumentError("Unknown robustness_measure: $m"))
end

"""
    _cvar(fidelities::Vector{Float64}, alpha::Float64) -> Float64

Compute the Conditional Value-at-Risk (CVaR) of a fidelity distribution.

CVaR at level `alpha` is the mean of the worst `alpha` fraction of outcomes.
For quantum control this gives a measure that focuses on improving the tail
of the fidelity distribution.

`alpha = 0.2` → average of worst 20% of samples.
"""
function _cvar(fidelities::Vector{Float64}, alpha::Float64;
                check_invariants::Bool = false)::Float64
    n          = length(fidelities)
    n_tail     = max(1, round(Int, alpha * n))
    sorted_f   = sort(fidelities)             # ascending: worst first
    if check_invariants
        ok, msg = check_cvar_ordering(sorted_f)
        _assert_invariant(ok, msg, :cvar_ordering,
                          (; n=n, n_tail=n_tail, alpha=alpha))
    end
    return mean(sorted_f[1:n_tail])
end

"""
    cvar(fidelities, alpha; check_invariants=false) -> Float64

Public alias of the Conditional Value-at-Risk helper. Identical behaviour to
the internal `_cvar` — exposed so users building custom `EnsembleObjective`s
can compute CVaR without digging into private symbols.
"""
cvar(fidelities::Vector{Float64}, alpha::Float64; check_invariants::Bool=false) =
    _cvar(fidelities, alpha; check_invariants=check_invariants)

"""
    _robust_gradient(system, controls::Matrix{Float64}, target,
                      perturbations::Vector, robustness_measure::String;
                      cvar_alpha::Float64 = 0.2) -> Matrix{Float64}

Compute the gradient of the robustness metric with respect to `controls`.

Uses the interchange of gradient and expectation (valid under mild regularity):
    ∇_u E[F(u; ΔH)] = E[∇_u F(u; ΔH)]

For worst-case: uses the gradient at the worst perturbation sample.
For CVaR: uses the mean gradient over the tail samples.
"""
function _robust_gradient(system, controls::Matrix{Float64}, target,
                            perturbations::Vector, robustness_measure::String;
                            cvar_alpha::Float64 = 0.2)::Matrix{Float64}
    n  = length(perturbations)
    fv = Vector{Float64}(undef, n)
    gv = Vector{Vector{Float64}}(undef, n)   # flat, for shared aggregator

    # Parallel evaluation over perturbation samples (each is independent).
    # Lesson 1+7: `@threadsif` adds BLAS-thread guard around the loop.
    @threadsif true for i in eachindex(perturbations)
        fv[i] = _fidelity_with_perturbation(system, controls, target, perturbations[i])
        gv[i] = vec(_gradient_with_perturbation(system, controls, target, perturbations[i]))
    end

    # Delegate aggregation to the shared surgery (Ensemble/EnsembleObjective.jl).
    out = Vector{Float64}(undef, length(controls))
    _agg_grad!(out, fv, gv, _symbolize_measure(robustness_measure), cvar_alpha)
    return reshape(out, size(controls))
end

# ──────────────────────────────────────────────────────────────────────────────
# Parametric robustness
# ──────────────────────────────────────────────────────────────────────────────

"""
    optimize_robust_parametric(system, target, controls_init::Matrix{Float64},
                                config::RobustConfig) -> NamedTuple

Robust optimization against Hamiltonian parameter uncertainty.

At each iteration, samples `config.n_samples` Hermitian perturbations
ΔH with `‖ΔH‖ ≤ config.uncertainty_magnitude · ‖H_drift‖`, then takes a
gradient step in the direction that improves the chosen robustness measure.

The optimization objective is:
    maximize  ρ(F(u; H_0 + ΔH_1), ..., F(u; H_0 + ΔH_N))

where ρ is the robustness measure (mean, worst-case, or CVaR).

# Returns
Named tuple with `controls`, `fidelity`, `fidelity_history`, `converged`,
`iterations`, `method`, `metadata`.
"""
function optimize_robust_parametric(system, target, controls_init::Matrix{Float64},
                                     config::RobustConfig)
    controls      = copy(controls_init)
    best_controls = copy(controls)
    best_fidelity = -Inf
    fidelity_hist = Float64[]
    converged     = false
    α             = config.step_size
    rng           = Random.MersenneTwister(config.seed)   # reproducible sampling

    for iter in 1:config.max_iter
        # Sample fresh perturbations every iteration for stochastic diversity
        perturbations = sample_parametric_perturbations(system, config.n_samples,
                                                         config.uncertainty_magnitude; rng=rng)

        # Compute robust gradient and fidelity
        rf = robust_fidelity(system, controls, target, perturbations,
                              config.robustness_measure; cvar_alpha = config.cvar_alpha)
        rg = _robust_gradient(system, controls, target, perturbations,
                               config.robustness_measure; cvar_alpha = config.cvar_alpha)

        g_norm = norm(rg)
        push!(fidelity_hist, rf)

        if rf > best_fidelity
            best_fidelity = rf
            best_controls = copy(controls)
        end

        config.verbose && iter % 50 == 0 && @printf(
            "[RobustParam iter=%4d] robust_fidelity=%.6f  |g|=%.2e  α=%.2e\n",
            iter, rf, g_norm, α)

        if g_norm < config.convergence_tol
            converged = true
            break
        end

        # Armijo line search on nominal fidelity for step acceptance
        u_trial   = controls .+ α .* rg
        rf_trial  = robust_fidelity(system, u_trial, target, perturbations,
                                     config.robustness_measure; cvar_alpha = config.cvar_alpha)

        if rf_trial >= rf - 1e-12
            controls = u_trial
            α        = min(α * 1.05, 1.0)
        else
            α = max(α * 0.5, 1e-12)
        end
    end

    return (
        controls         = best_controls,
        fidelity         = best_fidelity,
        fidelity_history = fidelity_hist,
        converged        = converged,
        n_iterations     = length(fidelity_hist),
        method           = "robust_parametric",
        metadata         = Dict{String,Any}(
            "uncertainty_type"      => "parametric",
            "uncertainty_magnitude" => config.uncertainty_magnitude,
            "robustness_measure"    => config.robustness_measure,
            "n_samples"             => config.n_samples)
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Noise robustness
# ──────────────────────────────────────────────────────────────────────────────

"""
    optimize_robust_noise(system, target, controls_init::Matrix{Float64},
                           config::RobustConfig) -> NamedTuple

Robust optimization against quantum noise modelled by Kraus operators.

The noisy gate is:
    E(U) = ∑ᵢ Kᵢ U Kᵢ†

where {Kᵢ} are Kraus operators for depolarizing or dephasing noise with
strength set by `config.uncertainty_magnitude`.

The fidelity under noise is:
    F_noisy = |Tr(U_target† E(U))|² / d²

where `d` is the Hilbert space dimension.

Gradient is estimated by finite difference over `config.n_samples` noise
realizations drawn from the Kraus ensemble.

# Returns
Named tuple with `controls`, `fidelity`, `fidelity_history`, `converged`,
`iterations`, `method`, `metadata`.
"""
function optimize_robust_noise(system, target, controls_init::Matrix{Float64},
                                config::RobustConfig)
    controls      = copy(controls_init)
    best_controls = copy(controls)
    best_fidelity = -Inf
    fidelity_hist = Float64[]
    converged     = false
    α             = config.step_size
    rng           = Random.MersenneTwister(config.seed + 1)
    noise_level   = config.uncertainty_magnitude

    dim  = _get_system_dim(system)
    kraus_ops = _build_depolarizing_kraus(dim, noise_level)
    # Lesson 6: per-iteration propagator cache, opt-in via `cache_propagators`.
    cache = config.cache_propagators ? PropagatorCache() : nothing

    for iter in 1:config.max_iter
        cache === nothing || cache_clear!(cache)

        _propagate = (u) -> begin
            cache === nothing && return _compute_propagator(system, u)
            cached_propagator(cache, system, u) do
                _compute_propagator(system, u)
            end
        end

        # Unitary propagator from current controls
        U = _propagate(controls)

        # Noisy process fidelity: F = (1/d²)|Tr(U_target† · sum_i K_i U K_i†)|²
        # We use the sample-averaged Choi fidelity approximation
        rf = _noisy_fidelity(U, target, kraus_ops, dim)
        rg = _noisy_gradient(system, controls, target, kraus_ops, dim)

        g_norm = norm(rg)
        push!(fidelity_hist, rf)

        if rf > best_fidelity
            best_fidelity = rf
            best_controls = copy(controls)
        end

        config.verbose && iter % 50 == 0 && @printf(
            "[RobustNoise  iter=%4d] noisy_fidelity=%.6f  |g|=%.2e  α=%.2e\n",
            iter, rf, g_norm, α)

        if g_norm < config.convergence_tol
            converged = true
            break
        end

        u_trial  = controls .+ α .* rg
        U_trial  = _propagate(u_trial)
        rf_trial = _noisy_fidelity(U_trial, target, kraus_ops, dim)

        if rf_trial >= rf - 1e-12
            controls = u_trial
            α        = min(α * 1.05, 1.0)
        else
            α = max(α * 0.5, 1e-12)
        end
    end

    return (
        controls         = best_controls,
        fidelity         = best_fidelity,
        fidelity_history = fidelity_hist,
        converged        = converged,
        n_iterations     = length(fidelity_hist),
        method           = "robust_noise",
        metadata         = Dict{String,Any}(
            "uncertainty_type" => "noise",
            "noise_level"      => noise_level,
            "n_kraus_ops"      => length(kraus_ops))
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Drift robustness
# ──────────────────────────────────────────────────────────────────────────────

"""
    optimize_robust_drift(system, target, controls_init::Matrix{Float64},
                           config::RobustConfig) -> NamedTuple

Robust optimization against slow parameter drift.

Models drift as a linear ramp: `H(t) = H_0 + (t/T) · ΔH_final` where
ΔH_final is drawn from the same distribution as parametric perturbations.

The drift trajectory ensemble is re-sampled each iteration to cover the
space of possible drift directions.

# Returns
Named tuple with `controls`, `fidelity`, `fidelity_history`, `converged`,
`iterations`, `method`, `metadata`.
"""
function optimize_robust_drift(system, target, controls_init::Matrix{Float64},
                                config::RobustConfig)
    controls      = copy(controls_init)
    best_controls = copy(controls)
    best_fidelity = -Inf
    fidelity_hist = Float64[]
    converged     = false
    α             = config.step_size
    rng           = Random.MersenneTwister(config.seed + 2)
    n_timesteps   = size(controls, 2)

    for iter in 1:config.max_iter
        # Sample drift trajectories
        drift_trajectories = sample_drift_trajectories(
            system, config.n_samples, config.uncertainty_magnitude, n_timesteps; rng=rng)

        rf = _drift_robust_fidelity(system, controls, target,
                                     drift_trajectories, config.robustness_measure;
                                     cvar_alpha = config.cvar_alpha)
        rg = _drift_robust_gradient(system, controls, target,
                                     drift_trajectories, config.robustness_measure;
                                     cvar_alpha = config.cvar_alpha)

        g_norm = norm(rg)
        push!(fidelity_hist, rf)

        if rf > best_fidelity
            best_fidelity = rf
            best_controls = copy(controls)
        end

        config.verbose && iter % 50 == 0 && @printf(
            "[RobustDrift  iter=%4d] robust_fidelity=%.6f  |g|=%.2e  α=%.2e\n",
            iter, rf, g_norm, α)

        if g_norm < config.convergence_tol
            converged = true
            break
        end

        u_trial  = controls .+ α .* rg
        rf_trial = _drift_robust_fidelity(system, u_trial, target,
                                           drift_trajectories, config.robustness_measure;
                                           cvar_alpha = config.cvar_alpha)

        if rf_trial >= rf - 1e-12
            controls = u_trial
            α        = min(α * 1.05, 1.0)
        else
            α = max(α * 0.5, 1e-12)
        end
    end

    return (
        controls         = best_controls,
        fidelity         = best_fidelity,
        fidelity_history = fidelity_hist,
        converged        = converged,
        n_iterations     = length(fidelity_hist),
        method           = "robust_drift",
        metadata         = Dict{String,Any}(
            "uncertainty_type"      => "drift",
            "uncertainty_magnitude" => config.uncertainty_magnitude,
            "robustness_measure"    => config.robustness_measure)
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ──────────────────────────────────────────────────────────────────────────────

"""
    robust_optimize(system, target, controls_init::Matrix{Float64};
                    config::RobustConfig = RobustConfig()) -> NamedTuple

Top-level robust quantum control optimizer for Pulsar.

Dispatches to the appropriate robust method based on `config.uncertainty_type`:

| `uncertainty_type` | Method                                   |
|--------------------|------------------------------------------|
| `"parametric"`     | Sample-based ensemble optimization       |
| `"noise"`          | Kraus-operator noise-averaged fidelity   |
| `"drift"`          | Ensemble over drift trajectory samples   |

# Arguments
- `system`: quantum system with `H_drift`, `H_controls`, `dt` fields
- `target`: gate or state target
- `controls_init`: initial control amplitudes `[n_controls × n_timesteps]`
- `config`: `RobustConfig` specifying uncertainty and optimization settings

# Returns
Named tuple with fields:
- `controls::Matrix{Float64}` — robust optimal controls
- `fidelity::Float64` — best robustness metric achieved
- `fidelity_history::Vector{Float64}` — robustness metric per iteration
- `converged::Bool`
- `iterations::Int`
- `method::String`
- `metadata::Dict`

# Cross-optimizer alternative

This entry point wraps a hand-rolled steepest-ascent loop with Armijo line
search. If you need the **same** ensemble (`:mean` / `:worst_case` / `:cvar`)
driven by a different optimizer (L-BFGS, CG, Adam, CMA-ES, …), build the
ensemble directly and pick any Pulsar optimizer:

```julia
obj       = build_ensemble_from_perturbations(system, target, ctrl;
                                              uncertainty_type = :parametric,
                                              magnitude        = 0.05,
                                              n_samples        = 20,
                                              aggregator       = :worst_case,
                                              resample         = false)
f, grad!  = ensemble_wrap(obj)
θ_opt, _, _ = lbfgs_optimize(f, grad!, vec(ctrl.controls); max_iter=200)
```

See [`build_ensemble_from_perturbations`](@ref).

# Example
```julia
cfg    = RobustConfig(uncertainty_type="parametric", uncertainty_magnitude=0.05,
                      robustness_measure="mean", n_samples=20, verbose=true)
result = robust_optimize(system, target, controls_init; config=cfg)
println("Robust fidelity: ", result.fidelity)
```
"""
function robust_optimize(system, target, controls_init::Matrix{Float64};
                          config::RobustConfig = RobustConfig())

    t0 = time()
    if config.uncertainty_type == "parametric"
        result = optimize_robust_parametric(system, target, controls_init, config)
    elseif config.uncertainty_type == "noise"
        result = optimize_robust_noise(system, target, controls_init, config)
    elseif config.uncertainty_type == "drift"
        result = optimize_robust_drift(system, target, controls_init, config)
    else
        throw(ArgumentError("Unknown uncertainty_type: '$(config.uncertainty_type)'. " *
                            "Choose from: parametric, noise, drift"))
    end
    return merge(result, (total_time = time() - t0,))
end

# ControlSequence overload — unwrap and delegate to the Matrix{Float64} method
function robust_optimize(system, target, controls_init::ControlSequence;
                          config::RobustConfig = RobustConfig())
    robust_optimize(system, target, controls_init.controls; config=config)
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    _get_drift_hamiltonian(system) -> Matrix{ComplexF64}

Extract the drift Hamiltonian from a system object. Tries `system.H_drift`
and `system.drift_hamiltonian` in that order.
"""
function _get_drift_hamiltonian(system)::Matrix{ComplexF64}
    if hasproperty(system, :H_drift)
        return Matrix{ComplexF64}(system.H_drift)
    elseif hasproperty(system, :drift_hamiltonian)
        return Matrix{ComplexF64}(system.drift_hamiltonian)
    else
        # Fallback: return zero matrix using system dimension
        d = _get_system_dim(system)
        return zeros(ComplexF64, d, d)
    end
end

"""
    _get_system_dim(system) -> Int

Extract the Hilbert space dimension from a system object.
"""
function _get_system_dim(system)::Int
    if hasproperty(system, :dim)
        return Int(system.dim)
    elseif hasproperty(system, :H_drift)
        return size(system.H_drift, 1)
    else
        return 2   # default to qubit
    end
end

"""
    _compute_propagator(system, controls::Matrix{Float64}) -> Matrix{ComplexF64}

Compute the time-ordered propagator U(T, 0) for the given controls.
Uses sequential matrix exponentials (Trotterized propagator).
"""
function _compute_propagator(system, controls::Matrix{Float64})::Matrix{ComplexF64}
    H0          = _get_drift_hamiltonian(system)
    dim         = size(H0, 1)
    dt          = _ro_get_dt(system)
    n_timesteps = size(controls, 2)
    H_ctrls     = _get_control_hamiltonians(system)
    n_controls  = min(size(controls, 1), length(H_ctrls))

    U = Matrix{ComplexF64}(I, dim, dim)
    for k in 1:n_timesteps
        H_k = copy(H0)
        for j in 1:n_controls
            H_k .+= controls[j,k] .* H_ctrls[j]
        end
        U = exp(-im * dt * H_k) * U
    end
    return U
end

"""
    _fidelity_with_perturbation(system, controls, target, perturbation) -> Float64

Compute gate fidelity with an additive Hamiltonian perturbation ΔH.
If perturbation is a `Matrix`, it is added to H_drift uniformly.
If it is a `Vector{Matrix}`, it is applied at each timestep.
"""
function _fidelity_with_perturbation(system, controls::AbstractMatrix{Float64}, target,
                                      perturbation)::Float64
    H0     = _get_drift_hamiltonian(system)
    dim    = size(H0, 1)
    dt     = _ro_get_dt(system)
    n_ts   = size(controls, 2)
    Hcs    = _get_control_hamiltonians(system)
    nc     = min(size(controls, 1), length(Hcs))
    U_tgt  = _get_target_unitary(target, dim)

    U = Matrix{ComplexF64}(I, dim, dim)
    for k in 1:n_ts
        dH_k = isa(perturbation, Vector) ? perturbation[k] : perturbation
        H_k  = H0 .+ dH_k
        for j in 1:nc
            H_k .+= controls[j,k] .* Hcs[j]
        end
        U = exp(-im * dt * H_k) * U
    end

    F = abs(tr(U_tgt' * U))^2 / dim^2
    return clamp(real(F), 0.0, 1.0)
end

"""
    _gradient_with_perturbation(system, controls, target, perturbation) -> Matrix{Float64}

Compute the gradient of fidelity with perturbation using finite differences.
"""
function _gradient_with_perturbation(system, controls::AbstractMatrix{Float64}, target,
                                      perturbation)::Matrix{Float64}
    grad  = zeros(size(controls))
    u_buf = copy(controls)
    central_diff_gradient_2d!(
        grad,
        u -> _fidelity_with_perturbation(system, u, target, perturbation),
        u_buf; eps = 1e-5,
    )
    return grad
end

"""
    _noisy_fidelity(U, target, kraus_ops, dim) -> Float64

Compute gate fidelity of propagator `U` under Kraus noise channel.
"""
function _noisy_fidelity(U::Matrix{ComplexF64}, target, kraus_ops::Vector,
                          dim::Int)::Float64
    U_tgt  = _get_target_unitary(target, dim)
    # Noisy channel: E(U) = sum_i K_i U K_i†
    E_U    = sum(K * U * K' for K in kraus_ops)
    F      = abs(tr(U_tgt' * E_U))^2 / dim^2
    return clamp(real(F), 0.0, 1.0)
end

"""
    _noisy_gradient(system, controls, target, kraus_ops, dim) -> Matrix{Float64}

Finite-difference gradient of noisy fidelity.
"""
function _noisy_gradient(system, controls::Matrix{Float64}, target,
                          kraus_ops::Vector, dim::Int)::Matrix{Float64}
    grad  = zeros(size(controls))
    u_buf = copy(controls)
    central_diff_gradient_2d!(
        grad,
        u -> _noisy_fidelity(_compute_propagator(system, u), target, kraus_ops, dim),
        u_buf; eps = 1e-5,
    )
    return grad
end

"""
    _drift_robust_fidelity(system, controls, target, drift_trajectories,
                            robustness_measure; cvar_alpha) -> Float64
"""
function _drift_robust_fidelity(system, controls::Matrix{Float64}, target,
                                 drift_trajectories::Vector,
                                 robustness_measure::String;
                                 cvar_alpha::Float64 = 0.2)::Float64
    fv = [_fidelity_with_perturbation(system, controls, target, traj)
          for traj in drift_trajectories]
    return _agg_value(fv, _symbolize_measure(robustness_measure), cvar_alpha)
end

"""
    _drift_robust_gradient(system, controls, target, drift_trajectories,
                            robustness_measure; cvar_alpha) -> Matrix{Float64}
"""
function _drift_robust_gradient(system, controls::Matrix{Float64}, target,
                                  drift_trajectories::Vector,
                                  robustness_measure::String;
                                  cvar_alpha::Float64 = 0.2)::Matrix{Float64}
    n   = length(drift_trajectories)
    fv  = Vector{Float64}(undef, n)
    gv  = Vector{Vector{Float64}}(undef, n)
    # Parallel evaluation over drift trajectory samples (each is independent).
    # Lesson 1+7: `@threadsif` adds BLAS-thread guard around the loop.
    @threadsif true for i in eachindex(drift_trajectories)
        fv[i] = _fidelity_with_perturbation(system, controls, target, drift_trajectories[i])
        gv[i] = vec(_gradient_with_perturbation(system, controls, target, drift_trajectories[i]))
    end

    out = Vector{Float64}(undef, length(controls))
    _agg_grad!(out, fv, gv, _symbolize_measure(robustness_measure), cvar_alpha)
    return reshape(out, size(controls))
end

"""
    _build_depolarizing_kraus(dim::Int, noise_level::Float64) -> Vector{Matrix{ComplexF64}}

Build Kraus operators for a depolarizing channel with strength `noise_level`.

For a single qubit (`dim=2`):
    K_0 = sqrt(1 - 3p/4) · I
    K_i = sqrt(p/4) · σᵢ  for i = x, y, z

For higher dimensions, uses a generalized depolarizing channel.
"""
function _build_depolarizing_kraus(dim::Int, noise_level::Float64)::Vector{Matrix{ComplexF64}}
    p = clamp(noise_level, 0.0, 1.0)

    if dim == 2
        # Pauli matrices
        sx = ComplexF64[0 1; 1 0]
        sy = ComplexF64[0 -im; im 0]
        sz = ComplexF64[1 0; 0 -1]
        Id = ComplexF64[1 0; 0 1]

        K0 = sqrt(max(0.0, 1.0 - 3p/4)) .* Id
        K1 = sqrt(p/4) .* sx
        K2 = sqrt(p/4) .* sy
        K3 = sqrt(p/4) .* sz
        return [K0, K1, K2, K3]
    else
        # Generalized: identity channel + uniform depolarizing
        d2    = dim^2
        kraus = Vector{Matrix{ComplexF64}}(undef, d2)
        basis = _generalized_gell_mann(dim)
        K0    = sqrt(max(0.0, 1.0 - p * (d2 - 1) / d2)) .* Matrix{ComplexF64}(I, dim, dim)
        kraus[1] = K0
        for (i, G) in enumerate(basis)
            kraus[i+1] = sqrt(p / d2) .* G
        end
        return kraus
    end
end

"""
    _generalized_gell_mann(dim::Int) -> Vector{Matrix{ComplexF64}}

Generate the `dim²-1` traceless Hermitian generators (generalized Gell-Mann matrices)
for SU(dim), forming an orthogonal basis for the space of traceless Hermitian matrices.
"""
function _generalized_gell_mann(dim::Int)::Vector{Matrix{ComplexF64}}
    basis = Matrix{ComplexF64}[]

    # Symmetric off-diagonal: e_jk + e_kj
    for j in 1:dim, k in (j+1):dim
        M = zeros(ComplexF64, dim, dim)
        M[j,k] = 1.0; M[k,j] = 1.0
        push!(basis, M)
    end

    # Anti-symmetric off-diagonal: -i(e_jk - e_kj)
    for j in 1:dim, k in (j+1):dim
        M = zeros(ComplexF64, dim, dim)
        M[j,k] = -im; M[k,j] = im
        push!(basis, M)
    end

    # Diagonal (l = 1,...,dim-1)
    for l in 1:(dim-1)
        M = zeros(ComplexF64, dim, dim)
        sc = sqrt(2.0 / (l * (l + 1)))
        for j in 1:l
            M[j,j] = sc
        end
        M[l+1, l+1] = -l * sc
        push!(basis, M)
    end

    return basis
end

"""
    _get_control_hamiltonians(system) -> Vector{Matrix{ComplexF64}}

Extract the list of control Hamiltonians from a system object.
"""
function _get_control_hamiltonians(system)::Vector{Matrix{ComplexF64}}
    if hasproperty(system, :H_controls)
        return [Matrix{ComplexF64}(H) for H in system.H_controls]
    elseif hasproperty(system, :control_hamiltonians)
        return [Matrix{ComplexF64}(H) for H in system.control_hamiltonians]
    else
        return Matrix{ComplexF64}[]
    end
end

"""
    _get_target_unitary(target, dim::Int) -> Matrix{ComplexF64}

Extract the target unitary from a target object. Falls back to the identity.
"""
function _get_target_unitary(target, dim::Int)::Matrix{ComplexF64}
    if hasproperty(target, :unitary)
        return Matrix{ComplexF64}(target.unitary)
    elseif hasproperty(target, :target_unitary)
        return Matrix{ComplexF64}(target.target_unitary)
    elseif isa(target, Matrix)
        return Matrix{ComplexF64}(target)
    else
        return Matrix{ComplexF64}(I, dim, dim)
    end
end

"""
    _ro_get_dt(system) -> Float64

Extract timestep `dt` from system; fallback to `1.0`.
"""
function _ro_get_dt(system)::Float64
    hasproperty(system, :dt) ? Float64(system.dt) : 1.0
end
