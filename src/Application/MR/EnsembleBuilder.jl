# ============================================================================
# Application/MR/EnsembleBuilder.jl
# ============================================================================
# Builder that wraps an MR-layer control (`MRControl` or `LindbladMRControl`)
# in an `EnsembleObjective`. Lives in the Application layer because it
# dispatches on Application-layer types; the core Ensemble module
# (Optimization/Ensemble/*) is kept free of MR dependencies.
#
# Two execution modes:
#
#   :mean aggregator  → batched_eval fast path. Delegates the full
#                        (drifts × pwr × state_pairs) ensemble evaluation to the
#                        existing `grape_state_kernel` / `grape_lindblad_kernel`
#                        — preserving their threaded CPU / batched GPU paths
#                        bit-for-bit.
#
#   :worst_case / :cvar → per-sample closures via
#                        `grape_state_kernel_single` /
#                        `grape_lindblad_kernel_single`. Loses the GPU-batched
#                        fast path (one closure per sample) but is the only way
#                        to evaluate non-mean aggregators.
# ============================================================================

using LinearAlgebra

"""
    build_ensemble_from_mrcontrol(ctrl::AbstractMRControl;
                                   aggregator = :mean,
                                   cvar_alpha = 0.2) -> EnsembleObjective

Build an [`EnsembleObjective`](@ref) from an MR-layer `MRControl` (closed
system) or `LindbladMRControl` (open system). Penalties declared on `ctrl` are
applied once to the aggregated `(F, ∇F)` — consistent with the kernels'
existing `_apply_penalties!` behaviour.

# Modes
- `aggregator = :mean` (default) — uses the batched kernel as a single
  `batched_eval` call. No performance regression relative to calling
  `grape_state_kernel` / `grape_lindblad_kernel` directly. Identical to the
  pre-refactor path used by `optimcon`.

- `aggregator ∈ (:worst_case, :cvar)` — flattens the
  `drifts × pwr_levels × state_pairs` outer product into `n_samples` per-sample
  closures. Each closure builds its own propagators and returns the single-
  member `(F, ∇F)` via `grape_state_kernel_single` /
  `grape_lindblad_kernel_single`. Penalties are applied by the aggregator
  after reduction.

# Arguments
- `ctrl`       — `MRControl` or `LindbladMRControl`
- `aggregator` — `:mean`, `:worst_case`, or `:cvar`
- `cvar_alpha` — tail fraction for `:cvar`

# Example
```julia
# :mean — production-fast (GPU-batched if ctrl.backend != :cpu)
obj       = build_ensemble_from_mrcontrol(ctrl; aggregator = :mean)
f, grad!  = ensemble_wrap(obj)
θ_opt, F, _ = lbfgs_optimize(f, grad!, vec(init_w); max_iter = 200)

# :worst_case — new capability (per-sample evaluation path)
obj       = build_ensemble_from_mrcontrol(ctrl; aggregator = :worst_case)
f, grad!  = ensemble_wrap(obj)
θ_opt, F, _ = adam_optimize(f, grad!, vec(init_w); max_iter = 500)
```

# See also
- [`grape_state_kernel`](@ref), [`grape_lindblad_kernel`](@ref) — batched fast path
- [`grape_state_kernel_single`](@ref), [`grape_lindblad_kernel_single`](@ref) — per-sample
- [`ensemble_wrap`](@ref), [`ensemble_wrap_fonly`](@ref)
"""
function build_ensemble_from_mrcontrol(ctrl::MRControl;
                                        aggregator::Symbol = :mean,
                                        cvar_alpha::Real   = 0.2)
    n_ctrl = length(ctrl.operators)
    n_t    = length(ctrl.pulse_dt)

    if aggregator === :mean
        batched_eval = function (θ::AbstractVector{<:Real})
            w      = _reshape_theta(θ, n_ctrl, n_t)
            F, G   = grape_state_kernel(w, ctrl)
            return F, vec(G)
        end
        N_ens = length(ctrl.drifts) * length(ctrl.pwr_levels) * length(ctrl.rho_init)
        return EnsembleObjective(; batched_eval = batched_eval,
                                   n_samples    = N_ens,
                                   aggregator   = :mean,
                                   cvar_alpha   = Float64(cvar_alpha))
    end

    # Per-sample path for :worst_case / :cvar — flatten (drift, pwr, pair)
    drifts     = ctrl.drifts
    pwrs       = ctrl.pwr_levels
    rho_inits  = ctrl.rho_init
    rho_targs  = ctrl.rho_targ

    samples = Tuple{Matrix{ComplexF64}, Float64, Vector{ComplexF64}, Vector{ComplexF64}}[]
    for H_drift in drifts, pwr in pwrs, s in eachindex(rho_inits)
        push!(samples, (H_drift, pwr, rho_inits[s], rho_targs[s]))
    end
    n_samples = length(samples)

    _mk_f(smp) = let smp = smp, nc = n_ctrl, nt = n_t, c = ctrl
        θ -> begin
            w = _reshape_theta(θ, nc, nt)
            F, _ = grape_state_kernel_single(w, smp[1], smp[2], smp[3], smp[4], c)
            F
        end
    end
    _mk_g(smp) = let smp = smp, nc = n_ctrl, nt = n_t, c = ctrl
        (gv, θ) -> begin
            w = _reshape_theta(θ, nc, nt)
            _, G = grape_state_kernel_single(w, smp[1], smp[2], smp[3], smp[4], c)
            copyto!(gv, vec(G))
            gv
        end
    end
    f_samples    = [_mk_f(samples[i]) for i in 1:n_samples]
    grad_samples = [_mk_g(samples[i]) for i in 1:n_samples]

    return EnsembleObjective(f_samples;
                              grad_samples = grad_samples,
                              aggregator   = aggregator,
                              cvar_alpha   = Float64(cvar_alpha),
                              n_samples    = n_samples)
end

function build_ensemble_from_mrcontrol(ctrl::LindbladMRControl;
                                        aggregator::Symbol = :mean,
                                        cvar_alpha::Real   = 0.2)
    n_ctrl = length(ctrl._L_controls)
    n_t    = length(ctrl.pulse_dt)

    if aggregator === :mean
        batched_eval = function (θ::AbstractVector{<:Real})
            w      = _reshape_theta(θ, n_ctrl, n_t)
            F, G   = grape_lindblad_kernel(w, ctrl)
            return F, vec(G)
        end
        N_ens = length(ctrl._L_drifts) * length(ctrl.pwr_levels) *
                length(ctrl._sigma_init)
        return EnsembleObjective(; batched_eval = batched_eval,
                                   n_samples    = N_ens,
                                   aggregator   = :mean,
                                   cvar_alpha   = Float64(cvar_alpha))
    end

    L_drifts   = ctrl._L_drifts
    pwrs       = ctrl.pwr_levels
    sig_inits  = ctrl._sigma_init
    sig_targs  = ctrl._sigma_targ

    samples = Tuple{Matrix{ComplexF64}, Float64, Vector{ComplexF64}, Vector{ComplexF64}}[]
    for L_drift in L_drifts, pwr in pwrs, s in eachindex(sig_inits)
        push!(samples, (L_drift, pwr, sig_inits[s], sig_targs[s]))
    end
    n_samples = length(samples)

    _mk_f(smp) = let smp = smp, nc = n_ctrl, nt = n_t, c = ctrl
        θ -> begin
            w = _reshape_theta(θ, nc, nt)
            F, _ = grape_lindblad_kernel_single(w, smp[1], smp[2], smp[3], smp[4], c)
            F
        end
    end
    _mk_g(smp) = let smp = smp, nc = n_ctrl, nt = n_t, c = ctrl
        (gv, θ) -> begin
            w = _reshape_theta(θ, nc, nt)
            _, G = grape_lindblad_kernel_single(w, smp[1], smp[2], smp[3], smp[4], c)
            copyto!(gv, vec(G))
            gv
        end
    end
    f_samples    = [_mk_f(samples[i]) for i in 1:n_samples]
    grad_samples = [_mk_g(samples[i]) for i in 1:n_samples]

    return EnsembleObjective(f_samples;
                              grad_samples = grad_samples,
                              aggregator   = aggregator,
                              cvar_alpha   = Float64(cvar_alpha),
                              n_samples    = n_samples)
end
