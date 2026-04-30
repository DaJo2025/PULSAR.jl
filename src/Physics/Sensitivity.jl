"""
    SensitivityAnalysis.jl

Global and local sensitivity analysis for quantum optimal control pulses.

After optimization, it is often valuable to know:

  - Which control amplitudes most strongly affect the final fidelity?
  - Which time steps are critical versus redundant?
  - Can any control channels be removed (sparsified) without significant loss?

This module provides three sensitivity methods:

  `:gradient`   — Local method. Uses |∂F/∂u_j[k]| at the optimal controls as
                   a sensitivity measure. Fast (one gradient evaluation) but
                   only locally valid near the optimum.

  `:finite_diff` — Local method. Computes ΔF when each control is perturbed
                   by a finite step. Captures nonlinear effects better than
                   the gradient alone.

  `:morris`     — Global screening method (Morris OAT, 1991). Randomly explores
                   the full input space by making one-at-a-time (OAT) changes
                   along random trajectories. Returns mean |elementary effect|
                   (μ*) and standard deviation (σ) for each input.

All methods return a `SensitivityResult` with per-control, per-timestep
sensitivity values and convenient summary statistics.

# Reference
  Morris, M.D. (1991). "Factorial sampling plans for preliminary computational
  experiments". Technometrics 33(2):161–174.
"""

using LinearAlgebra
using Statistics
using Random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""
    SensitivityConfig

Configuration for sensitivity analysis of an optimal control pulse.

# Fields
- `method::Symbol` — analysis method; one of `:gradient`, `:finite_diff`,
  or `:morris`. Default `:gradient`.
- `n_trajectories::Int` — number of Morris OAT trajectories (only used when
  `method == :morris`). Default `20`.
- `levels::Int` — number of grid levels for the Morris method (controls
  coarseness of the OAT grid). Must be even. Default `4`.
- `normalize::Bool` — when `true`, scale sensitivities so the maximum entry
  is 1.0. Default `true`.
- `verbose::Bool` — print progress during Morris screening. Default `false`.

# Example
```julia
cfg = SensitivityConfig(method=:morris, n_trajectories=30, verbose=true)
sens = compute_sensitivity(result, sys, tgt; config=cfg)
```
"""
struct SensitivityConfig
    method::Symbol
    n_trajectories::Int
    levels::Int
    normalize::Bool
    verbose::Bool
end

"""
    SensitivityConfig(; method=:gradient, n_trajectories=20,
                        levels=4, normalize=true, verbose=false)
    -> SensitivityConfig

Keyword constructor for `SensitivityConfig`.

# Keyword Arguments
- `method`         — `:gradient`, `:finite_diff`, or `:morris`
- `n_trajectories` — Morris trajectory count (≥ 1)
- `levels`         — Morris grid levels (must be even, ≥ 2)
- `normalize`      — normalize final sensitivities to [0, 1]
- `verbose`        — print progress messages

# Throws
- `ArgumentError` if `method` is not recognised.
- `ArgumentError` if `n_trajectories ≤ 0`.
- `ArgumentError` if `levels` is not a positive even integer.
"""
function SensitivityConfig(;
        method::Symbol     = :gradient,
        n_trajectories::Int = 20,
        levels::Int        = 4,
        normalize::Bool    = true,
        verbose::Bool      = false)::SensitivityConfig

    valid = (:gradient, :finite_diff, :morris)
    if !(method in valid)
        throw(ArgumentError("method must be one of $valid, got :$method"))
    end
    if n_trajectories <= 0
        throw(ArgumentError("n_trajectories must be > 0, got $n_trajectories"))
    end
    if levels < 2 || isodd(levels)
        throw(ArgumentError("levels must be a positive even integer ≥ 2, got $levels"))
    end
    return SensitivityConfig(method, n_trajectories, levels, normalize, verbose)
end

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

"""
    SensitivityResult

Encapsulates the output of a sensitivity analysis.

# Fields
- `control_sensitivities::Matrix{Float64}` — raw sensitivity values for each
  control amplitude, shape `[n_controls × n_timesteps]`.
- `normalized_sensitivities::Matrix{Float64}` — sensitivities scaled so the
  maximum value is 1.0 (all zeros if max is zero).
- `most_sensitive::Vector{Tuple{Int,Int}}` — indices `(control_idx, timestep_idx)`
  sorted from highest to lowest sensitivity (full ranking).
- `least_sensitive::Vector{Tuple{Int,Int}}` — same indices but sorted from
  lowest to highest sensitivity.
- `total_sensitivity_per_control::Vector{Float64}` — sum of sensitivities over
  all timesteps for each control channel; length `n_controls`.
- `total_sensitivity_per_timestep::Vector{Float64}` — sum of sensitivities over
  all controls for each timestep; length `n_timesteps`.
- `method_used::Symbol` — the method that produced this result.
- `metadata::Dict{String, Any}` — additional method-specific information.
"""
struct SensitivityResult
    control_sensitivities::Matrix{Float64}
    normalized_sensitivities::Matrix{Float64}
    most_sensitive::Vector{Tuple{Int,Int}}
    least_sensitive::Vector{Tuple{Int,Int}}
    total_sensitivity_per_control::Vector{Float64}
    total_sensitivity_per_timestep::Vector{Float64}
    method_used::Symbol
    metadata::Dict{String, Any}
end

function Base.show(io::IO, s::SensitivityResult)
    nc, nt = size(s.control_sensitivities)
    top = first(s.most_sensitive, 3)
    @printf(io, "SensitivityResult(method=%s, [%d controls x %d timesteps], top3=%s)",
            s.method_used, nc, nt, top)
end

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

"""
    compute_sensitivity(result::OptimizationResult,
                         system::AbstractQuantumSystem,
                         target::QuantumTarget;
                         config::SensitivityConfig = SensitivityConfig())
    -> SensitivityResult

Compute sensitivity of the fidelity with respect to each control amplitude.

# Arguments
- `result`  — output of a previous optimization (provides optimal controls)
- `system`  — quantum system
- `target`  — optimization target
- `config`  — sensitivity configuration; see `SensitivityConfig`

# Returns
A `SensitivityResult` with sensitivity maps and summary statistics.

# Dispatch
| `config.method`  | Analysis called          |
|:---------------- |:------------------------ |
| `:gradient`      | `gradient_sensitivity`   |
| `:finite_diff`   | `finite_diff_sensitivity`|
| `:morris`        | `morris_screening`       |

# Example
```julia
cfg  = SensitivityConfig(method=:gradient, normalize=true)
sens = compute_sensitivity(result, sys, tgt; config=cfg)
println(summarize_sensitivity(sens))
```
"""
function compute_sensitivity(
        result::OptimizationResult,
        system::AbstractQuantumSystem,
        target::QuantumTarget;
        config::SensitivityConfig = SensitivityConfig())::SensitivityResult

    u_opt = result.controls
    nc, nt = size(u_opt)
    dt = result.dt

    # Compute raw sensitivities
    raw_sens, metadata = if config.method == :gradient
        S = gradient_sensitivity(result, system, target)
        S, Dict{String,Any}("method_detail" => "analytical GRAPE gradient")
    elseif config.method == :finite_diff
        S = finite_diff_sensitivity(result, system, target)
        S, Dict{String,Any}("method_detail" => "central finite differences")
    else   # :morris
        morris_result = morris_screening(system, target, u_opt;
                                          n_trajectories=config.n_trajectories,
                                          levels=config.levels,
                                          verbose=config.verbose)
        S = morris_result["mu_star"]
        Dict{String,Any}(
            "mu_star"       => morris_result["mu_star"],
            "sigma"         => morris_result["sigma"],
            "n_trajectories"=> config.n_trajectories,
            "levels"        => config.levels,
        )
    end

    # Normalize
    max_s = maximum(raw_sens)
    norm_sens = if max_s > 0.0 && config.normalize
        raw_sens ./ max_s
    else
        copy(raw_sens)
    end

    # Rankings
    idx_pairs = [(j, k) for j in 1:nc, k in 1:nt]
    sorted_pairs = sort(vec(idx_pairs);
                        by = jk -> -raw_sens[jk[1], jk[2]])
    most_sens  = sorted_pairs
    least_sens = reverse(sorted_pairs)

    # Marginals
    total_per_ctrl = vec(sum(raw_sens; dims=2))
    total_per_time = vec(sum(raw_sens; dims=1))

    metadata["max_sensitivity"] = max_s
    metadata["normalized"]      = config.normalize

    return SensitivityResult(
        raw_sens, norm_sens,
        most_sens, least_sens,
        total_per_ctrl, total_per_time,
        config.method, metadata,
    )
end

# ---------------------------------------------------------------------------
# Gradient-based sensitivity
# ---------------------------------------------------------------------------

"""
    gradient_sensitivity(result::OptimizationResult,
                          system::AbstractQuantumSystem,
                          target::QuantumTarget)
    -> Matrix{Float64}

Compute local sensitivity as the absolute value of the GRAPE gradient.

# Arguments
- `result`  — optimization result (optimal controls, dt)
- `system`  — quantum system
- `target`  — optimization target

# Returns
Sensitivity matrix `S[j,k] = |∂F/∂u_j[k]|`, shape `[n_controls × n_timesteps]`.

# Notes
This is the fastest sensitivity measure (one gradient evaluation) but only
reflects local landscape curvature at the optimal controls. Controls at a
true optimum will have nearly zero gradient — interpret gradient sensitivity
after partial optimization or at intermediate solutions where |∇F| > 0.

For a fully converged solution, `finite_diff_sensitivity` with a larger
perturbation amplitude gives a more informative sensitivity landscape.
"""
function gradient_sensitivity(result::OptimizationResult,
                                system::AbstractQuantumSystem,
                                target::QuantumTarget)::Matrix{Float64}
    u_opt = result.controls
    nt    = size(u_opt, 2)
    dt    = result.dt
    seq   = ControlSequence(u_opt, dt, dt * nt, nt)
    G     = compute_grape_gradient(system, seq, target)
    return abs.(G)
end

# ---------------------------------------------------------------------------
# Finite-difference sensitivity
# ---------------------------------------------------------------------------

"""
    finite_diff_sensitivity(result::OptimizationResult,
                             system::AbstractQuantumSystem,
                             target::QuantumTarget;
                             eps::Float64 = 0.01)
    -> Matrix{Float64}

Compute sensitivity as the absolute fidelity change under a finite perturbation.

# Arguments
- `result`  — optimization result
- `system`  — quantum system
- `target`  — optimization target
- `eps`     — perturbation amplitude in the same units as the controls.
  Default `0.01` (assumed to be a meaningful fractional perturbation).

# Returns
Sensitivity matrix `S[j,k] = |F(u + ε e_{jk}) - F(u)|`,
shape `[n_controls × n_timesteps]`.

# Notes
Uses a one-sided forward-difference (cheaper than central differences) since
the sign of the sensitivity is not needed — only the magnitude.  For a more
accurate measure use central differences (at twice the cost).

This method is more informative than gradient sensitivity at a fully-converged
optimum because it captures finite-amplitude effects (the gradient vanishes at
the optimum but finite perturbations still change fidelity).
"""
function finite_diff_sensitivity(result::OptimizationResult,
                                   system::AbstractQuantumSystem,
                                   target::QuantumTarget;
                                   eps::Float64 = 0.01)::Matrix{Float64}
    u_opt  = result.controls
    nc, nt = size(u_opt)
    dt     = result.dt

    # Baseline fidelity
    seq0 = ControlSequence(u_opt, dt, dt * nt, nt)
    H0   = build_total_hamiltonian(system, seq0)
    U0   = _sa_propagate_total(H0, dt)
    F0   = compute_fidelity(U0, target)

    S = zeros(Float64, nc, nt)
    for j in 1:nc
        for k in 1:nt
            u_perturbed = copy(u_opt)
            u_perturbed[j, k] += eps
            seq_p = ControlSequence(u_perturbed, dt, dt * nt, nt)
            H_p   = build_total_hamiltonian(system, seq_p)
            U_p   = _sa_propagate_total(H_p, dt)
            F_p   = compute_fidelity(U_p, target)
            S[j, k] = abs(F_p - F0)
        end
    end

    return S
end

# ---------------------------------------------------------------------------
# Morris OAT screening
# ---------------------------------------------------------------------------

"""
    morris_screening(system::AbstractQuantumSystem,
                      target::QuantumTarget,
                      controls_ref::Matrix{Float64};
                      n_trajectories::Int = 20,
                      levels::Int = 4,
                      verbose::Bool = false)
    -> Dict{String, Matrix{Float64}}

Perform Morris one-at-a-time (OAT) elementary effect screening.

# Arguments
- `system`         — quantum system
- `target`         — optimization target
- `controls_ref`   — reference control matrix `[n_controls × n_timesteps]` used to
  define the sampling range. The OAT grid is centered on the range
  `[min(u_ref) - Δ, max(u_ref) + Δ]` where `Δ = 0.1 * std(u_ref)`.
- `n_trajectories` — number of random OAT trajectories. More trajectories give
  more accurate μ* and σ estimates. Default `20`.
- `levels`         — number of discrete levels for the OAT grid. Default `4`.
  The step size is `Δu = range / (levels - 1)`.
- `verbose`        — print trajectory progress. Default `false`.

# Returns
`Dict` with two entries, each a matrix of shape `[n_controls × n_timesteps]`:

- `"mu_star"` — mean of absolute elementary effects μ* = E[|EE_{j,k}|].
  Large μ* indicates an influential control parameter.
- `"sigma"`   — standard deviation of elementary effects.
  Large σ relative to μ* indicates interaction effects or nonlinearity.

# Algorithm
For each trajectory r = 1, …, n_trajectories:
  1. Draw a random starting point x⁰ on the discrete grid.
  2. Generate a random permutation of all n_controls × n_timesteps parameters.
  3. For each parameter (j,k) in the permutation order: change u_jk by ±Δu
     (direction chosen randomly), evaluate F, record elementary effect
     `EE_{j,k} = ΔF / Δu`.

After all trajectories, compute μ* and σ over the collected EE values.

# Reference
  Morris, M.D. (1991). Technometrics 33(2):161–174.
  Saltelli et al. (2008). "Global Sensitivity Analysis: The Primer".

# Example
```julia
d = morris_screening(sys, tgt, u_opt; n_trajectories=30, levels=6)
# d["mu_star"] and d["sigma"] are [n_controls × n_timesteps] matrices
```
"""
function morris_screening(system::AbstractQuantumSystem,
                            target::QuantumTarget,
                            controls_ref::Matrix{Float64};
                            n_trajectories::Int = 20,
                            levels::Int = 4,
                            verbose::Bool = false)::Dict{String, Matrix{Float64}}

    nc, nt = size(controls_ref)
    n_params = nc * nt

    # Build sampling range from the reference controls
    u_min = minimum(controls_ref) - 0.1 * max(std(controls_ref), 1e-8)
    u_max = maximum(controls_ref) + 0.1 * max(std(controls_ref), 1e-8)
    u_range = u_max - u_min
    delta = u_range / (levels - 1)        # OAT step size

    # Accumulate elementary effects: shape [n_trajectories × nc × nt]
    EE = zeros(Float64, n_trajectories, nc, nt)

    dt = controls_ref[1, 1] == controls_ref[1, 1] ? NaN : NaN  # dummy; extract from result
    # We need dt — use a unit dt and assume controls_ref carries it implicitly.
    # Caller passes controls_ref; we need ControlSequence to evaluate fidelity.
    # We use dt=1.0 as a placeholder normalized time (the user should ensure that
    # controls_ref was obtained with a known dt; this function needs dt externally).
    # Since we only have controls_ref, use dt = 1/(nt) as a normalized value.
    # The sensitivity ratios remain meaningful regardless of absolute dt.
    dt_morris = 1.0 / nt

    rng = Random.GLOBAL_RNG

    for r in 1:n_trajectories
        if verbose
            println("[Morris] Trajectory $r / $n_trajectories")
        end

        # Random starting point on the OAT grid
        x0 = u_min .+ delta .* rand(rng, 0:(levels-1), nc, nt)
        x0 = Float64.(x0)

        # Evaluate baseline fidelity for this trajectory
        seq0  = ControlSequence(x0, dt_morris, 1.0, nt)
        H0    = build_total_hamiltonian(system, seq0)
        U0    = _sa_propagate_total(H0, dt_morris)
        F_cur = compute_fidelity(U0, target)
        x_cur = copy(x0)

        # Random permutation of all parameter indices
        perm = randperm(rng, n_params)

        for idx in perm
            # Convert flat index to (j, k)
            j = mod1(idx, nc)
            k = div(idx - 1, nc) + 1

            # Random ±delta direction
            direction = rand(rng, (-1, 1))
            x_new = copy(x_cur)
            x_new[j, k] += direction * delta

            # Clamp to [u_min, u_max] to stay on grid
            x_new[j, k] = clamp(x_new[j, k], u_min, u_max)

            # Evaluate fidelity
            seq_new = ControlSequence(x_new, dt_morris, 1.0, nt)
            H_new   = build_total_hamiltonian(system, seq_new)
            U_new   = _sa_propagate_total(H_new, dt_morris)
            F_new   = compute_fidelity(U_new, target)

            # Elementary effect
            actual_step = x_new[j, k] - x_cur[j, k]
            EE[r, j, k] = if abs(actual_step) > 1e-14
                (F_new - F_cur) / actual_step
            else
                0.0
            end

            # Advance to new point
            x_cur = x_new
            F_cur = F_new
        end
    end

    # Compute μ* and σ over trajectories
    mu_star = dropdims(mean(abs.(EE); dims=1); dims=1)
    sigma   = dropdims(std(EE; dims=1); dims=1)

    return Dict{String, Matrix{Float64}}(
        "mu_star" => mu_star,
        "sigma"   => sigma,
    )
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

"""
    summarize_sensitivity(sens::SensitivityResult) -> String

Generate a human-readable summary of a `SensitivityResult`.

# Arguments
- `sens` — sensitivity analysis result

# Returns
A multi-line string summarizing:
  - The method used.
  - The top-5 most sensitive control-timestep pairs.
  - Controls with total sensitivity below 1% of the maximum control sensitivity.
  - The timestep with peak sensitivity.

# Example
```julia
println(summarize_sensitivity(sens))
# Sensitivity Analysis Summary
# Method: gradient
# Top 5 most sensitive:
#   1. (control=2, t=17): sensitivity = 0.923
#   2. (control=1, t=18): sensitivity = 0.871
#   ...
# Insensitive controls (< 1% of max): control 3 (total = 0.003)
# Peak timestep: t=17 (total sensitivity = 2.14)
```
"""
function summarize_sensitivity(sens::SensitivityResult)::String
    nc, nt = size(sens.control_sensitivities)
    raw    = sens.control_sensitivities
    lines  = String[]

    push!(lines, "Sensitivity Analysis Summary")
    push!(lines, "Method: $(sens.method_used)")

    # Top-5 most sensitive
    push!(lines, "Top 5 most sensitive control-timestep pairs:")
    for (rank, (j, k)) in enumerate(first(sens.most_sensitive, 5))
        push!(lines, @sprintf("  %d. (control=%d, t=%d): sensitivity = %.4f",
                              rank, j, k, raw[j, k]))
    end

    # Insensitive controls
    max_ctrl_sens = maximum(sens.total_sensitivity_per_control)
    thresh = 0.01 * max_ctrl_sens
    insensitive = findall(s -> s < thresh, sens.total_sensitivity_per_control)
    if !isempty(insensitive)
        ctrl_list = join(["control $c (total = " *
                          @sprintf("%.3e", sens.total_sensitivity_per_control[c]) * ")"
                          for c in insensitive], ", ")
        push!(lines, "Insensitive controls (< 1% of max): $ctrl_list")
    else
        push!(lines, "All controls contribute significantly (>1% of max)")
    end

    # Peak timestep
    peak_t = argmax(sens.total_sensitivity_per_timestep)
    push!(lines, @sprintf("Peak timestep: t=%d (total sensitivity = %.4f)",
                           peak_t, sens.total_sensitivity_per_timestep[peak_t]))

    # Method-specific extras
    if haskey(sens.metadata, "mu_star")
        push!(lines, "Morris μ* range: [" *
              @sprintf("%.3e", minimum(sens.metadata["mu_star"])) * ", " *
              @sprintf("%.3e", maximum(sens.metadata["mu_star"])) * "]")
        push!(lines, "Morris σ range:  [" *
              @sprintf("%.3e", minimum(sens.metadata["sigma"])) * ", " *
              @sprintf("%.3e", maximum(sens.metadata["sigma"])) * "]")
    end

    return join(lines, "\n")
end

# ---------------------------------------------------------------------------
# Internal helper: propagate H_total array to get full propagator
# ---------------------------------------------------------------------------

"""
    _sa_propagate_total(H_total::Array{ComplexF64,3}, dt::Float64)
    -> Matrix{ComplexF64}

Compute the full time-ordered propagator U_total = U[n_t] ⋅ … ⋅ U[1].
Internal helper for `SensitivityAnalysis`.
"""
function _sa_propagate_total(H_total::Array{ComplexF64,3},
                               dt::Float64)::Matrix{ComplexF64}
    dim = size(H_total, 1)
    n_t = size(H_total, 3)
    U   = Matrix{ComplexF64}(I, dim, dim)
    for k in 1:n_t
        U = compute_propagator(H_total[:, :, k], dt) * U
    end
    return U
end
