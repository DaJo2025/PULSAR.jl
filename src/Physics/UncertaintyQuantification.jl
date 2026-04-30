"""
    UncertaintyQuantification.jl

Uncertainty quantification (UQ) for quantum optimal control solutions.

After a GRAPE or L-BFGS optimization converges, the resulting control sequence
is a point estimate of the optimal solution.  This module estimates:

  1. **Control uncertainty** — how sensitive is the optimal control pulse to
     small perturbations in the starting point or objective landscape?
  2. **Fidelity confidence interval** — what range of fidelities can realistically
     be achieved given the landscape curvature or solution variability?

Three complementary methods are provided:

  `:hessian`   — Gaussian approximation based on the inverse Hessian of the
                  fidelity landscape at the optimum. Fast but assumes a
                  quadratic landscape near the optimum.

  `:bootstrap` — Perturb the optimal controls, re-optimize from each perturbed
                  start, and measure the spread of converged solutions.
                  Captures multi-modality and flat directions.

  `:sampling`  — Direct Monte-Carlo sampling of the fidelity landscape near
                  the optimum; gives a non-parametric CI estimate.

# References
  - Goodwin & Kuprov, "Modified Newton-Raphson GRAPE methods for optimal control
    of spin systems", J. Chem. Phys. 144, 204107 (2016).
  - Efron & Tibshirani, "An Introduction to the Bootstrap", Chapman & Hall (1993).
"""

using LinearAlgebra
using Statistics
using Random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""
    UQConfig

Configuration for uncertainty quantification of an optimization result.

# Fields
- `method::Symbol` — UQ method; one of `:hessian`, `:bootstrap`, or `:sampling`.
  Default `:hessian`.
- `confidence_level::Float64` — confidence level for intervals; must be in
  `(0, 1)`. Default `0.95`.
- `n_samples::Int` — number of bootstrap re-optimizations or Monte-Carlo
  samples (depending on `method`). Default `100`.
- `regularization::Float64` — Tikhonov regularization added to the Hessian
  diagonal before inversion to ensure positive-definiteness. Default `1e-8`.
- `random_seed::Union{Int, Nothing}` — seed for reproducible sampling; `nothing`
  uses the global RNG. Default `nothing`.

# Example
```julia
cfg = UQConfig(method=:bootstrap, n_samples=50, confidence_level=0.90)
ures = estimate_uncertainty(result, sys, tgt; config=cfg)
```
"""
struct UQConfig
    method::Symbol
    confidence_level::Float64
    n_samples::Int
    regularization::Float64
    random_seed::Union{Int, Nothing}
end

"""
    UQConfig(; method=:hessian, confidence_level=0.95, n_samples=100,
               regularization=1e-8, random_seed=nothing) -> UQConfig

Keyword constructor for `UQConfig`.

# Keyword Arguments
- `method`           — `:hessian`, `:bootstrap`, or `:sampling`
- `confidence_level` — confidence level in `(0, 1)`
- `n_samples`        — number of samples/reoptimizations
- `regularization`   — Hessian diagonal regularization (≥ 0)
- `random_seed`      — integer seed or `nothing`

# Throws
- `ArgumentError` if `method` is not recognised.
- `ArgumentError` if `confidence_level` is not in `(0, 1)`.
- `ArgumentError` if `n_samples ≤ 0` or `regularization < 0`.
"""
function UQConfig(;
        method::Symbol              = :hessian,
        confidence_level::Float64   = 0.95,
        n_samples::Int              = 100,
        regularization::Float64     = 1e-8,
        random_seed::Union{Int,Nothing} = nothing)::UQConfig

    valid = (:hessian, :bootstrap, :sampling)
    if !(method in valid)
        throw(ArgumentError("method must be one of $valid, got :$method"))
    end
    if !(0.0 < confidence_level < 1.0)
        throw(ArgumentError(
            "confidence_level must be in (0, 1), got $confidence_level"))
    end
    if n_samples <= 0
        throw(ArgumentError("n_samples must be > 0, got $n_samples"))
    end
    if regularization < 0.0
        throw(ArgumentError("regularization must be ≥ 0, got $regularization"))
    end
    return UQConfig(method, confidence_level, n_samples, regularization, random_seed)
end

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

"""
    UncertaintyResult

Encapsulates the output of an uncertainty quantification analysis.

# Fields
- `optimal_controls::Matrix{Float64}` — optimal control amplitudes
  `[n_controls × n_timesteps]` taken from the input `OptimizationResult`.
- `optimal_fidelity::Float64` — fidelity at the optimal controls.
- `control_uncertainty::Matrix{Float64}` — estimated one-sigma (1σ) standard
  deviation for each control amplitude; same shape as `optimal_controls`.
- `fidelity_ci_lower::Float64` — lower bound of the fidelity confidence interval
  at the requested `confidence_level`.
- `fidelity_ci_upper::Float64` — upper bound of the fidelity confidence interval.
- `covariance_matrix::Union{Matrix{Float64}, Nothing}` — full parameter covariance
  matrix of size `[n_params × n_params]` where `n_params = n_controls * n_timesteps`.
  Non-`nothing` only for the `:hessian` method.
- `method_used::Symbol` — the UQ method that produced this result.
- `metadata::Dict{String, Any}` — additional method-specific information (e.g.
  number of iterations, regularization used, Hessian condition number).
"""
struct UncertaintyResult
    optimal_controls::Matrix{Float64}
    optimal_fidelity::Float64
    control_uncertainty::Matrix{Float64}
    fidelity_ci_lower::Float64
    fidelity_ci_upper::Float64
    covariance_matrix::Union{Matrix{Float64}, Nothing}
    method_used::Symbol
    metadata::Dict{String, Any}
end

function Base.show(io::IO, r::UncertaintyResult)
    @printf(io, "UncertaintyResult(method=%s, F=%.6f, CI=[%.6f, %.6f], max_ctrl_unc=%.3e)",
            r.method_used, r.optimal_fidelity,
            r.fidelity_ci_lower, r.fidelity_ci_upper,
            maximum(r.control_uncertainty))
end

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

"""
    estimate_uncertainty(result::OptimizationResult,
                          system::AbstractQuantumSystem,
                          target::QuantumTarget;
                          config::UQConfig = UQConfig())
    -> UncertaintyResult

Estimate uncertainty in an optimization result using the configured method.

# Arguments
- `result`  — output of a previous `grape_optimize` (or similar) call
- `system`  — quantum system used for the original optimization
- `target`  — optimization target used for the original optimization
- `config`  — UQ configuration; see `UQConfig`

# Returns
An `UncertaintyResult` containing control uncertainties, a fidelity confidence
interval, and (for `:hessian`) the full parameter covariance matrix.

# Dispatch
| `config.method`  | Function called            |
|:---------------- |:-------------------------- |
| `:hessian`       | `hessian_uncertainty`      |
| `:bootstrap`     | `bootstrap_uncertainty`    |
| `:sampling`      | `sampling_uncertainty`     |

# Example
```julia
result = grape_optimize(sys, tgt, u_init, dt)
uq_cfg = UQConfig(method=:hessian, confidence_level=0.95)
ures   = estimate_uncertainty(result, sys, tgt; config=uq_cfg)
println("Control std dev: ", ures.control_uncertainty)
println("Fidelity CI: [", ures.fidelity_ci_lower, ", ", ures.fidelity_ci_upper, "]")
```
"""
function estimate_uncertainty(
        result::OptimizationResult,
        system::AbstractQuantumSystem,
        target::QuantumTarget;
        config::UQConfig = UQConfig())::UncertaintyResult

    if config.method == :hessian
        return hessian_uncertainty(result, system, target, config)
    elseif config.method == :bootstrap
        return bootstrap_uncertainty(result, system, target, config)
    else  # :sampling
        return sampling_uncertainty(result, system, target, config)
    end
end

# ---------------------------------------------------------------------------
# Hessian-based UQ
# ---------------------------------------------------------------------------

"""
    hessian_uncertainty(result::OptimizationResult,
                         system::AbstractQuantumSystem,
                         target::QuantumTarget,
                         config::UQConfig)
    -> UncertaintyResult

Estimate parameter uncertainty from the inverse Hessian at the optimum.

# Algorithm
The Hessian H of the fidelity landscape is approximated by finite differences:

    H[i,j] ≈ (F(u + εᵢ + εⱼ) - F(u + εᵢ - εⱼ) -
               F(u - εᵢ + εⱼ) + F(u - εᵢ - εⱼ)) / (4 ε²)

where εᵢ is the unit perturbation in direction i scaled by `h = 1e-4` times
the RMS control amplitude (or `1e-6` if the controls are near zero).

The parameter covariance is then approximated as

    Σ ≈ (-H + λ I)⁻¹

where λ = `config.regularization` is a Tikhonov term ensuring positive-
definiteness. Control uncertainties are `σⱼ = sqrt(Σ[j,j])`.

The fidelity confidence interval uses the Gaussian approximation:

    F ± z_{α/2} * σ_F

where σ_F is estimated by error propagation through the gradient.

# Notes
- The full Hessian requires O(n_params²) fidelity evaluations, which is
  expensive for large control sequences. For n_params > 500 consider using
  `:sampling` instead.
- The regularization `λ` is increased automatically if the regularized Hessian
  is still not positive-definite.
"""
function hessian_uncertainty(result::OptimizationResult,
                               system::AbstractQuantumSystem,
                               target::QuantumTarget,
                               config::UQConfig)::UncertaintyResult
    u_opt  = result.controls
    F_opt  = result.fidelity
    nc, nt = size(u_opt)
    n_params = nc * nt
    dt     = result.dt

    # Finite-difference step scaled to control amplitude
    rms_u = sqrt(mean(abs2, u_opt))
    h = max(1e-4 * rms_u, 1e-6)

    # Build full Hessian via central FD (expensive but exact up to FD error)
    u_flat = vec(u_opt)
    H_mat  = _compute_hessian_fd(u_flat, system, target, nc, nt, dt, h)

    # Regularize and invert: Σ = (-H + λ I)^{-1}
    λ = config.regularization
    neg_H = -H_mat
    reg_H = neg_H + λ * I

    # Ensure positive-definiteness by bumping regularization if needed
    λ_actual = λ
    max_attempts = 10
    Σ = nothing
    for attempt in 1:max_attempts
        try
            F_chol = cholesky(Hermitian(reg_H))
            Σ = Matrix(inv(F_chol))
            break
        catch
            λ_actual *= 10.0
            reg_H = neg_H + λ_actual * I
        end
    end

    if Σ === nothing
        # Fall back to pseudo-inverse if Hessian is severely ill-conditioned
        Σ = pinv(reg_H)
        λ_actual = NaN
    end

    # Control uncertainties: sqrt of diagonal of covariance
    ctrl_var  = max.(diag(Σ), 0.0)   # clamp negatives from numerical noise
    ctrl_std  = sqrt.(ctrl_var)
    ctrl_unc  = reshape(ctrl_std, nc, nt)

    # Gradient at optimum for fidelity uncertainty by error propagation
    seq_opt = ControlSequence(u_opt, dt, dt * nt, nt)
    G = compute_grape_gradient(system, seq_opt, target)
    g_flat = vec(G)
    # σ_F² = g^T Σ g
    sigma_F = sqrt(max(dot(g_flat, Σ * g_flat), 0.0))

    ci_lower, ci_upper = compute_confidence_interval_gaussian(
        F_opt, sigma_F, config.confidence_level)

    # Clamp CI to [0, 1]
    ci_lower = max(ci_lower, 0.0)
    ci_upper = min(ci_upper, 1.0)

    cond_num = cond(reg_H)
    metadata = Dict{String, Any}(
        "hessian_step"         => h,
        "regularization_used"  => λ_actual,
        "hessian_condition"    => cond_num,
        "sigma_fidelity"       => sigma_F,
        "n_params"             => n_params,
    )

    return UncertaintyResult(
        u_opt, F_opt, ctrl_unc,
        ci_lower, ci_upper,
        Σ,
        :hessian,
        metadata,
    )
end

# ---------------------------------------------------------------------------
# Bootstrap UQ
# ---------------------------------------------------------------------------

"""
    bootstrap_uncertainty(result::OptimizationResult,
                           system::AbstractQuantumSystem,
                           target::QuantumTarget,
                           config::UQConfig)
    -> UncertaintyResult

Estimate uncertainty by re-optimizing from perturbed starting points.

# Algorithm
For each of `config.n_samples` bootstrap iterations:
  1. Add zero-mean Gaussian noise with standard deviation `σ_perturb` to the
     optimal controls, where `σ_perturb = 0.05 * rms(u_opt)`.
  2. Run a short GRAPE optimization (200 iterations) starting from the
     perturbed controls.
  3. Record the converged control sequence and fidelity.

After all re-optimizations:
  - `control_uncertainty[j,k] = std(u_bootstrap[:, j, k])` across samples.
  - Fidelity CI is the empirical percentile interval of fidelity values.

# Notes
This method is more robust than the Hessian approach for landscapes with
flat directions or multiple local optima.  It is, however, significantly more
expensive (n_samples × optimizer cost).  Use `n_samples ≤ 50` for quick
diagnostics and `n_samples ≥ 200` for production-quality estimates.
"""
function bootstrap_uncertainty(result::OptimizationResult,
                                 system::AbstractQuantumSystem,
                                 target::QuantumTarget,
                                 config::UQConfig)::UncertaintyResult
    u_opt  = result.controls
    F_opt  = result.fidelity
    nc, nt = size(u_opt)
    dt     = result.dt

    rng = config.random_seed === nothing ? Random.GLOBAL_RNG :
                                           MersenneTwister(config.random_seed)

    rms_u       = sqrt(mean(abs2, u_opt))
    sigma_perturb = 0.05 * max(rms_u, 1e-6)

    boot_controls  = Array{Float64, 3}(undef, config.n_samples, nc, nt)
    boot_fidelities = Vector{Float64}(undef, config.n_samples)

    # Short re-optimization config
    boot_config = GRAPEConfig(
        max_iter        = 200,
        convergence_tol = 1e-8,
        verbose         = false,
    )

    for s in 1:config.n_samples
        u_perturbed = u_opt .+ sigma_perturb .* randn(rng, nc, nt)
        boot_result = grape_optimize(system, target, u_perturbed, dt;
                                      config=boot_config)
        boot_controls[s, :, :]  = boot_result.controls
        boot_fidelities[s]       = boot_result.fidelity
    end

    # Control uncertainty: std over bootstrap samples
    ctrl_unc = dropdims(std(boot_controls; dims=1); dims=1)

    # Fidelity CI: empirical percentile
    alpha = 1.0 - config.confidence_level
    lo_idx = max(1, round(Int, alpha / 2 * config.n_samples))
    hi_idx = min(config.n_samples, round(Int, (1 - alpha / 2) * config.n_samples))
    f_sorted = sort(boot_fidelities)
    ci_lower = f_sorted[lo_idx]
    ci_upper = f_sorted[hi_idx]

    metadata = Dict{String, Any}(
        "n_samples"       => config.n_samples,
        "sigma_perturb"   => sigma_perturb,
        "boot_fidelities" => boot_fidelities,
        "mean_fidelity"   => mean(boot_fidelities),
        "std_fidelity"    => std(boot_fidelities),
    )

    return UncertaintyResult(
        u_opt, F_opt, ctrl_unc,
        ci_lower, ci_upper,
        nothing,
        :bootstrap,
        metadata,
    )
end

# ---------------------------------------------------------------------------
# Sampling-based UQ
# ---------------------------------------------------------------------------

"""
    sampling_uncertainty(result::OptimizationResult,
                          system::AbstractQuantumSystem,
                          target::QuantumTarget,
                          config::UQConfig)
    -> UncertaintyResult

Estimate the fidelity confidence interval by Monte-Carlo sampling near the optimum.

# Algorithm
1. Draw `config.n_samples` random control perturbations from a multivariate
   Gaussian with mean zero and standard deviation `σ = 0.01 * rms(u_opt)`.
2. Evaluate the fidelity at each perturbed control sequence.
3. Estimate control uncertainty as the pointwise standard deviation of fidelity-
   weighted perturbations (a first-order sensitivity measure).
4. Construct a non-parametric fidelity CI from empirical percentiles.

This method does not require re-optimization and is therefore much cheaper than
`:bootstrap`.  It is appropriate when the landscape is approximately quadratic
near the optimum but Hessian computation is too expensive.

# Notes
The returned `covariance_matrix` is `nothing` for this method.  Use `:hessian`
if you need the full covariance.
"""
function sampling_uncertainty(result::OptimizationResult,
                                system::AbstractQuantumSystem,
                                target::QuantumTarget,
                                config::UQConfig)::UncertaintyResult
    u_opt  = result.controls
    F_opt  = result.fidelity
    nc, nt = size(u_opt)
    dt     = result.dt

    rng = config.random_seed === nothing ? Random.GLOBAL_RNG :
                                           MersenneTwister(config.random_seed)

    rms_u       = sqrt(mean(abs2, u_opt))
    sigma_sample = 0.01 * max(rms_u, 1e-6)

    sample_controls   = Array{Float64, 3}(undef, config.n_samples, nc, nt)
    sample_fidelities = Vector{Float64}(undef, config.n_samples)

    for s in 1:config.n_samples
        delta_u = sigma_sample .* randn(rng, nc, nt)
        u_s = u_opt .+ delta_u
        seq_s = ControlSequence(u_s, dt, dt * nt, nt)
        H_s   = build_total_hamiltonian(system, seq_s)
        U_s   = _propagate_total(H_s, dt)
        sample_fidelities[s]     = compute_fidelity(U_s, target)
        sample_controls[s, :, :] = delta_u
    end

    # Control uncertainty: std of perturbations weighted by |ΔF|
    delta_F = abs.(sample_fidelities .- F_opt)
    if sum(delta_F) < 1e-30
        ctrl_unc = zeros(Float64, nc, nt)
    else
        w = delta_F ./ sum(delta_F)
        ctrl_unc = zeros(Float64, nc, nt)
        for s in 1:config.n_samples
            ctrl_unc .+= w[s] .* abs.(sample_controls[s, :, :])
        end
    end

    # Empirical CI
    alpha = 1.0 - config.confidence_level
    lo_idx = max(1, round(Int, alpha / 2 * config.n_samples))
    hi_idx = min(config.n_samples, round(Int, (1 - alpha / 2) * config.n_samples))
    f_sorted = sort(sample_fidelities)
    ci_lower = max(f_sorted[lo_idx], 0.0)
    ci_upper = min(f_sorted[hi_idx], 1.0)

    metadata = Dict{String, Any}(
        "n_samples"       => config.n_samples,
        "sigma_sample"    => sigma_sample,
        "mean_fidelity"   => mean(sample_fidelities),
        "std_fidelity"    => std(sample_fidelities),
    )

    return UncertaintyResult(
        u_opt, F_opt, ctrl_unc,
        ci_lower, ci_upper,
        nothing,
        :sampling,
        metadata,
    )
end

# ---------------------------------------------------------------------------
# Fidelity confidence interval (standalone)
# ---------------------------------------------------------------------------

"""
    fidelity_confidence_interval(result::OptimizationResult,
                                  system::AbstractQuantumSystem,
                                  target::QuantumTarget,
                                  config::UQConfig)
    -> Tuple{Float64, Float64}

Estimate a confidence interval for the achievable fidelity near the optimum.

# Arguments
- `result`  — optimization result
- `system`  — quantum system
- `target`  — optimization target
- `config`  — UQ configuration (uses `n_samples`, `confidence_level`,
  `random_seed`)

# Returns
Tuple `(lower, upper)` giving the lower and upper bounds of the fidelity
confidence interval at `config.confidence_level`.

# Algorithm
Samples `config.n_samples` nearby control sequences drawn from a Gaussian
centred at the optimal controls with standard deviation `0.01 * rms(u_opt)`.
Returns the empirical α/2 and 1-α/2 percentiles of the sampled fidelities.

# Example
```julia
lo, hi = fidelity_confidence_interval(result, sys, tgt, UQConfig(n_samples=200))
println("95% CI: [\$lo, \$hi]")
```
"""
function fidelity_confidence_interval(
        result::OptimizationResult,
        system::AbstractQuantumSystem,
        target::QuantumTarget,
        config::UQConfig)::Tuple{Float64, Float64}

    u_opt  = result.controls
    nc, nt = size(u_opt)
    dt     = result.dt

    rng = config.random_seed === nothing ? Random.GLOBAL_RNG :
                                           MersenneTwister(config.random_seed)

    rms_u  = sqrt(mean(abs2, u_opt))
    sigma  = 0.01 * max(rms_u, 1e-6)

    fids = Vector{Float64}(undef, config.n_samples)
    for s in 1:config.n_samples
        u_s   = u_opt .+ sigma .* randn(rng, nc, nt)
        seq_s = ControlSequence(u_s, dt, dt * nt, nt)
        H_s   = build_total_hamiltonian(system, seq_s)
        U_s   = _propagate_total(H_s, dt)
        fids[s] = compute_fidelity(U_s, target)
    end

    alpha  = 1.0 - config.confidence_level
    lo_idx = max(1, round(Int, alpha / 2 * config.n_samples))
    hi_idx = min(config.n_samples, round(Int, (1 - alpha / 2) * config.n_samples))
    f_sorted = sort(fids)
    return (max(f_sorted[lo_idx], 0.0), min(f_sorted[hi_idx], 1.0))
end

# ---------------------------------------------------------------------------
# Gaussian confidence interval helper
# ---------------------------------------------------------------------------

"""
    compute_confidence_interval_gaussian(value::Float64, uncertainty::Float64,
                                          confidence_level::Float64)
    -> Tuple{Float64, Float64}

Compute a symmetric Gaussian confidence interval.

# Arguments
- `value`            — point estimate (e.g. mean fidelity)
- `uncertainty`      — one-sigma standard deviation
- `confidence_level` — confidence level; e.g. `0.95` for a 95% CI

# Returns
Tuple `(lower, upper)` = `(value - z·σ, value + z·σ)` where `z` is the
standard normal quantile corresponding to `(1 + confidence_level) / 2`.

# Standard z-values
| confidence_level | z      |
|:---------------- |:------ |
| 0.90             | 1.645  |
| 0.95             | 1.960  |
| 0.99             | 2.576  |

# Example
```julia
lo, hi = compute_confidence_interval_gaussian(0.98, 0.005, 0.95)
# (0.9702, 0.9898)
```
"""
function compute_confidence_interval_gaussian(value::Float64,
                                               uncertainty::Float64,
                                               confidence_level::Float64)::Tuple{Float64, Float64}
    # Standard normal quantile by rational approximation (Beasley-Springer-Moro)
    p = (1.0 + confidence_level) / 2.0
    z = _normal_quantile(p)
    lower = value - z * uncertainty
    upper = value + z * uncertainty
    return (lower, upper)
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _compute_hessian_fd(u_flat, system, target, nc, nt, dt, h) -> Matrix{Float64}

Compute the Hessian matrix of the fidelity at `u_flat` using central
finite differences. The (i,j) entry is:

    H[i,j] = (F(u+hᵢ+hⱼ) - F(u+hᵢ-hⱼ) - F(u-hᵢ+hⱼ) + F(u-hᵢ-hⱼ)) / (4h²)

Cost: O(2 * n_params²) fidelity evaluations (only upper triangle is computed;
      symmetry is exploited for the lower triangle).
"""
function _compute_hessian_fd(u_flat::Vector{Float64},
                               system::AbstractQuantumSystem,
                               target::QuantumTarget,
                               nc::Int, nt::Int,
                               dt::Float64,
                               h::Float64)::Matrix{Float64}
    n = length(u_flat)
    H_mat = zeros(Float64, n, n)

    function eval_fidelity(u::Vector{Float64})
        u_mat = reshape(u, nc, nt)
        seq   = ControlSequence(u_mat, dt, dt * nt, nt)
        Ht    = build_total_hamiltonian(system, seq)
        Ut    = _propagate_total(Ht, dt)
        return compute_fidelity(Ut, target)
    end

    # Diagonal terms: H[i,i] = (F(u+2h eᵢ) - 2F(u) + F(u-2h eᵢ)) / (4h²)
    F0 = eval_fidelity(u_flat)
    for i in 1:n
        u_pp = copy(u_flat); u_pp[i] += 2h
        u_mm = copy(u_flat); u_mm[i] -= 2h
        H_mat[i, i] = (eval_fidelity(u_pp) - 2F0 + eval_fidelity(u_mm)) / (4h^2)
    end

    # Off-diagonal (upper triangle, exploit symmetry)
    for i in 1:n
        for j in (i+1):n
            u_pp = copy(u_flat); u_pp[i] += h; u_pp[j] += h
            u_pm = copy(u_flat); u_pm[i] += h; u_pm[j] -= h
            u_mp = copy(u_flat); u_mp[i] -= h; u_mp[j] += h
            u_mm = copy(u_flat); u_mm[i] -= h; u_mm[j] -= h
            val = (eval_fidelity(u_pp) - eval_fidelity(u_pm) -
                   eval_fidelity(u_mp) + eval_fidelity(u_mm)) / (4h^2)
            H_mat[i, j] = val
            H_mat[j, i] = val
        end
    end

    return H_mat
end

"""
    _propagate_total(H_total::Array{ComplexF64,3}, dt::Float64) -> Matrix{ComplexF64}

Compute the full time-ordered propagator from a 3-D Hamiltonian array.
Internal helper used by UQ sampling methods.
"""
function _propagate_total(H_total::Array{ComplexF64,3},
                            dt::Float64)::Matrix{ComplexF64}
    n_t = size(H_total, 1)
    dim = size(H_total, 2)
    U   = Matrix{ComplexF64}(I, dim, dim)
    for k in 1:n_t
        U = compute_propagator(H_total[k, :, :], dt) * U
    end
    return U
end

"""
    _normal_quantile(p::Float64) -> Float64

Approximate the standard-normal quantile function Φ⁻¹(p) using the
rational approximation by Peter Acklam (2002).

Accurate to about 9 significant digits for `p ∈ (0, 1)`.
"""
function _normal_quantile(p::Float64)::Float64
    # Coefficients for the rational approximation
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00)

    p_low  = 0.02425
    p_high = 1.0 - p_low

    if p < p_low
        q = sqrt(-2.0 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    elseif p <= p_high
        q = p - 0.5
        r = q^2
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    end
end
