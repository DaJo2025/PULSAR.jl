# ============================================================================
# Optimization/Ensemble/PerturbationBuilder.jl
# ============================================================================
# Builder that turns a single `AbstractQuantumSystem` + uncertainty
# specification into a stochastic-perturbation `EnsembleObjective`.
#
# Loaded AFTER `Optimization/Robust/RobustOpt.jl` because it delegates to
# `sample_parametric_perturbations`, `sample_drift_trajectories`,
# `_fidelity_with_perturbation`, and `_gradient_with_perturbation` defined
# there. Keeping the dependency direction strict (RobustOpt → Ensemble) avoids
# circular include order.
# ============================================================================

using LinearAlgebra
using Random

"""
    build_ensemble_from_perturbations(system, target, controls;
                                       uncertainty_type = :parametric,
                                       magnitude        = 0.05,
                                       n_samples        = 20,
                                       aggregator       = :mean,
                                       cvar_alpha       = 0.2,
                                       resample         = true,
                                       seed             = 42) -> EnsembleObjective

Build an [`EnsembleObjective`](@ref) for robust optimization by perturbing
a **single** nominal `system`. Replaces the hand-rolled optimizer body in
[`robust_optimize`](@ref) — produces the same ensemble mean / worst-case / CVaR
sample statistics but lets any PULSAR optimizer take the outer step.

# Arguments
- `uncertainty_type` — `:parametric` (Hermitian ΔH added to drift) or
  `:drift` (linear ramp trajectory). `:noise` (Kraus) is still served by
  `robust_optimize` directly because its finite-difference gradient does not
  reuse `compute_grape_gradient`; prefer `robust_optimize` for that case.
- `magnitude` — relative perturbation magnitude (same semantics as
  `RobustConfig.uncertainty_magnitude`).
- `n_samples` — number of perturbation draws per call.
- `resample` — redraw perturbations at the start of every
  `ensemble_value_and_grad` call (stochastic approximation). Set to `false` to
  fix the sample set (recommended with quasi-Newton inner optimizers).
- `seed` — base RNG seed; used verbatim when `resample=false`, and incremented
  per call when `resample=true`.

# Warning on stochastic resampling
`resample=true` invalidates the secant condition used by L-BFGS / BFGS. When
combining resampling with a quasi-Newton inner optimizer, expect noisy
convergence curves. Adam / SGD / plain gradient ascent are the natural fit;
for quasi-Newton inner loops prefer `resample=false`.
"""
function build_ensemble_from_perturbations(system::AbstractQuantumSystem,
                                            target::QuantumTarget,
                                            controls::ControlSequence;
                                            uncertainty_type::Symbol = :parametric,
                                            magnitude::Real           = 0.05,
                                            n_samples::Int            = 20,
                                            aggregator::Symbol        = :mean,
                                            cvar_alpha::Real          = 0.2,
                                            resample::Bool            = true,
                                            seed::Int                 = 42)
    uncertainty_type in (:parametric, :drift) ||
        throw(ArgumentError("uncertainty_type must be :parametric or :drift for " *
                            "this builder; use robust_optimize for :noise"))
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))

    dt          = controls.dt
    n_timesteps = controls.n_timesteps
    n_controls  = size(controls.controls, 1)

    # Mutable perturbation cache, updated in-place by `_draw!`.
    perturbations = Ref{Vector}(Vector{Any}(undef, 0))
    rng_state     = Ref{AbstractRNG}(Random.MersenneTwister(seed))
    call_counter  = Ref(0)

    function _draw!()
        if uncertainty_type === :parametric
            perturbations[] = sample_parametric_perturbations(system, n_samples,
                                                               Float64(magnitude);
                                                               rng = rng_state[])
        else # :drift
            perturbations[] = sample_drift_trajectories(system, n_samples,
                                                         Float64(magnitude),
                                                         n_timesteps;
                                                         rng = rng_state[])
        end
        return nothing
    end

    # Initial draw so the object is usable before the first optimizer call.
    _draw!()

    _perturbed_fidelity = (idx::Int, θ::AbstractVector{<:Real}) -> begin
        w    = _reshape_theta(θ, n_controls, n_timesteps)
        pert = perturbations[][idx]
        return _fidelity_with_perturbation(system, w, target, pert)
    end

    _perturbed_gradient! = (gv::Vector{Float64}, idx::Int, θ::AbstractVector{<:Real}) -> begin
        w    = _reshape_theta(θ, n_controls, n_timesteps)
        pert = perturbations[][idx]
        G    = _gradient_with_perturbation(system, w, target, pert)
        copyto!(gv, vec(G))
        return gv
    end

    _mk_f(i::Int) = let ii = i, _pf = _perturbed_fidelity
        θ -> _pf(ii, θ)
    end
    _mk_g(i::Int) = let ii = i, _pg = _perturbed_gradient!
        (gv, θ) -> _pg(gv, ii, θ)
    end
    f_samples    = [_mk_f(i) for i in 1:n_samples]
    grad_samples = [_mk_g(i) for i in 1:n_samples]

    resample_hook = if resample
        function (_obj)
            call_counter[] += 1
            rng_state[]     = Random.MersenneTwister(seed + call_counter[])
            _draw!()
            return nothing
        end
    else
        nothing
    end

    return EnsembleObjective(f_samples;
                              grad_samples = grad_samples,
                              aggregator   = aggregator,
                              cvar_alpha   = Float64(cvar_alpha),
                              resample!    = resample_hook,
                              n_samples    = n_samples)
end
