"""
    ConstrainedOptimization.jl

Constrained quantum control optimization for Pulsar.jl
(Pulse Design Library for Spin Control Algorithms and Rollout).

Implements penalty method, augmented Lagrangian, and projection-based approaches
for handling constraints on control amplitudes, power, bandwidth, and energy.

Supported constraint types:
- BoundConstraint: element-wise bounds lower ≤ u_j[k] ≤ upper
- PowerConstraint: ‖u‖₂² ≤ P_max
- BandwidthConstraint: max|u_j[k]| ≤ BW_max
- EnergyConstraint: dt · ∑|u_j[k]|² ≤ E_max
- CustomConstraint: user-defined c(u) ≤ 0
"""

using LinearAlgebra
using Printf

# ──────────────────────────────────────────────────────────────────────────────
# Constraint types
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractConstraint

Abstract supertype for all constraint types in Pulsar constrained optimization.
Every concrete subtype must be evaluable via `constraint_violation` and
projectable via `project_single_constraint`.
"""
abstract type AbstractConstraint end

"""
    BoundConstraint <: AbstractConstraint

Element-wise amplitude bounds: `lower ≤ u_j[k] ≤ upper` for every timestep k.

# Fields
- `lower::Float64`: lower bound on each control amplitude
- `upper::Float64`: upper bound on each control amplitude
- `control_indices::Vector{Int}`: indices of controls to constrain; empty means all controls

# Example
```julia
# Constrain all controls to [-1, 1]
bc = BoundConstraint(-1.0, 1.0, Int[])

# Constrain only controls 1 and 3 to [-0.5, 0.5]
bc2 = BoundConstraint(-0.5, 0.5, [1, 3])
```
"""
struct BoundConstraint <: AbstractConstraint
    lower::Float64
    upper::Float64
    control_indices::Vector{Int}

    function BoundConstraint(lower::Real, upper::Real, control_indices::Vector{Int}=Int[])
        lower < upper || throw(ArgumentError(
            "BoundConstraint: lower ($lower) must be strictly less than upper ($upper)"))
        new(Float64(lower), Float64(upper), control_indices)
    end
end

"""
    PowerConstraint <: AbstractConstraint

Total RF power constraint: `‖u‖₂² ≤ P_max` (sum over all controls and timesteps).

# Fields
- `P_max::Float64`: maximum allowed total power (must be positive)
"""
struct PowerConstraint <: AbstractConstraint
    P_max::Float64

    function PowerConstraint(P_max::Real)
        P_max > 0 || throw(ArgumentError("PowerConstraint: P_max must be positive, got $P_max"))
        new(Float64(P_max))
    end
end

"""
    BandwidthConstraint <: AbstractConstraint

Peak amplitude constraint: `max_{j,k}|u_j[k]| ≤ BW_max`.

# Fields
- `BW_max::Float64`: maximum allowed amplitude (bandwidth limit)
"""
struct BandwidthConstraint <: AbstractConstraint
    BW_max::Float64

    function BandwidthConstraint(BW_max::Real)
        BW_max > 0 || throw(ArgumentError(
            "BandwidthConstraint: BW_max must be positive, got $BW_max"))
        new(Float64(BW_max))
    end
end

# Legacy 2-arg form: BandwidthConstraint(f_max_hz, dt). The pre-discretised
# bandwidth in normalised units of 1/dt is `f_max_hz * dt`; we store this as
# BW_max so the existing peak-amplitude clamp acts as a coarse spectral cap.
function BandwidthConstraint(f_max_hz::Real, dt::Real)
    f_max_hz > 0 || throw(ArgumentError(
        "BandwidthConstraint: f_max_hz must be positive, got $f_max_hz"))
    dt > 0 || throw(ArgumentError(
        "BandwidthConstraint: dt must be positive, got $dt"))
    return BandwidthConstraint(Float64(f_max_hz) * Float64(dt))
end

"""
    EnergyConstraint <: AbstractConstraint

Pulse energy constraint: `dt · ∑_{j,k} |u_j[k]|² ≤ E_max`.

# Fields
- `E_max::Float64`: maximum allowed pulse energy
"""
struct EnergyConstraint <: AbstractConstraint
    E_max::Float64

    function EnergyConstraint(E_max::Real)
        E_max > 0 || throw(ArgumentError(
            "EnergyConstraint: E_max must be positive, got $E_max"))
        new(Float64(E_max))
    end
end

"""
    CustomConstraint <: AbstractConstraint

User-defined inequality constraint: `c(controls) ≤ 0`.

# Fields
- `name::String`: human-readable name for logging
- `constraint_fn::Function`: evaluates `c(controls::Matrix{Float64}) -> Float64`
- `gradient_fn::Function`: evaluates `∇c(controls::Matrix{Float64}) -> Matrix{Float64}`

# Example
```julia
# Peak-to-average power ratio constraint
papr_fn = u -> maximum(abs2, u) / mean(abs2, u) - 3.0
papr_gf = u -> ForwardDiff.gradient(v -> maximum(abs2, v) / mean(abs2, v), vec(u)) |>
               g -> reshape(g, size(u))
cc = CustomConstraint("PAPR", papr_fn, papr_gf)
```
"""
struct CustomConstraint <: AbstractConstraint
    name::String
    constraint_fn::Function
    gradient_fn::Function
end

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

"""
    ConstrainedConfig

Configuration for constrained optimization algorithms.

# Fields
- `base_method::String`: unconstrained optimizer to use inside each outer iteration
  (`"grape"`, `"bfgs"`, or `"lbfgs"`)
- `constraint_method::String`: constraint handling strategy
  (`"penalty"`, `"augmented_lagrangian"`, or `"projection"`)
- `max_iter::Int`: maximum outer iterations
- `penalty_initial::Float64`: initial penalty parameter λ₀
- `penalty_growth::Float64`: multiplicative growth factor for λ at each outer iteration
- `violation_tol::Float64`: declare convergence when max constraint violation < tol
- `verbose::Bool`: print iteration log

# Example
```julia
cfg = ConstrainedConfig(
    base_method         = "lbfgs",
    constraint_method   = "augmented_lagrangian",
    max_iter            = 50,
    penalty_initial     = 1.0,
    penalty_growth      = 10.0,
    violation_tol       = 1e-6,
    verbose             = true
)
```
"""
struct ConstrainedConfig
    base_method::String
    constraint_method::String
    max_iter::Int
    penalty_initial::Float64
    penalty_growth::Float64
    violation_tol::Float64
    verbose::Bool
    check_invariants::Bool

    function ConstrainedConfig(;
        base_method::String       = "grape",
        constraint_method::String = "augmented_lagrangian",
        max_iter::Int             = 100,
        penalty_initial::Float64  = 1.0,
        penalty_growth::Float64   = 10.0,
        violation_tol::Float64    = 1e-6,
        verbose::Bool             = true,
        check_invariants::Bool    = false
    )
        base_method in ("grape", "bfgs", "lbfgs") ||
            throw(ArgumentError("base_method must be one of: grape, bfgs, lbfgs"))
        constraint_method in ("penalty", "augmented_lagrangian", "projection") ||
            throw(ArgumentError(
                "constraint_method must be one of: penalty, augmented_lagrangian, projection"))
        max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
        penalty_initial > 0 || throw(ArgumentError("penalty_initial must be positive"))
        penalty_growth > 1 || throw(ArgumentError("penalty_growth must be > 1"))
        violation_tol > 0 || throw(ArgumentError("violation_tol must be positive"))
        new(base_method, constraint_method, max_iter,
            penalty_initial, penalty_growth, violation_tol, verbose,
            check_invariants)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Constraint evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    _evaluate_constraint(c::AbstractConstraint, controls::Matrix{Float64}, dt::Float64) -> Float64

Return the constraint residual `g(u)` such that feasibility requires `g(u) ≤ 0`.
Values > 0 indicate violation.
"""
function _evaluate_constraint(c::BoundConstraint, controls::Matrix{Float64}, dt::Float64)
    idxs = isempty(c.control_indices) ? axes(controls, 1) : c.control_indices
    viol = 0.0
    @inbounds for j in idxs, k in axes(controls, 2)
        viol = max(viol, controls[j,k] - c.upper, c.lower - controls[j,k])
    end
    return viol
end

function _evaluate_constraint(c::PowerConstraint, controls::Matrix{Float64}, dt::Float64)
    return sum(abs2, controls) - c.P_max
end

function _evaluate_constraint(c::BandwidthConstraint, controls::Matrix{Float64}, dt::Float64)
    return maximum(abs, controls) - c.BW_max
end

function _evaluate_constraint(c::EnergyConstraint, controls::Matrix{Float64}, dt::Float64)
    return dt * sum(abs2, controls) - c.E_max
end

function _evaluate_constraint(c::CustomConstraint, controls::Matrix{Float64}, dt::Float64)
    return c.constraint_fn(controls)
end

"""
    _constraint_gradient(c::AbstractConstraint, controls::Matrix{Float64}, dt::Float64) -> Matrix{Float64}

Return the gradient of the constraint residual with respect to controls.
"""
function _constraint_gradient(c::BoundConstraint, controls::Matrix{Float64}, dt::Float64)
    grad = zeros(size(controls))
    idxs = isempty(c.control_indices) ? axes(controls, 1) : c.control_indices
    @inbounds for j in idxs, k in axes(controls, 2)
        if controls[j,k] > c.upper
            grad[j,k] = 1.0
        elseif controls[j,k] < c.lower
            grad[j,k] = -1.0
        end
    end
    return grad
end

function _constraint_gradient(c::PowerConstraint, controls::Matrix{Float64}, dt::Float64)
    return 2.0 .* controls
end

function _constraint_gradient(c::BandwidthConstraint, controls::Matrix{Float64}, dt::Float64)
    grad = zeros(size(controls))
    idx  = argmax(abs.(controls))
    v    = controls[idx]
    grad[idx] = sign(v)
    return grad
end

function _constraint_gradient(c::EnergyConstraint, controls::Matrix{Float64}, dt::Float64)
    return 2.0 * dt .* controls
end

function _constraint_gradient(c::CustomConstraint, controls::Matrix{Float64}, dt::Float64)
    return c.gradient_fn(controls)
end

# ──────────────────────────────────────────────────────────────────────────────
# Constraint violation aggregate
# ──────────────────────────────────────────────────────────────────────────────

"""
    constraint_violation(controls::Matrix{Float64},
                          constraints::Vector{<:AbstractConstraint};
                          dt::Float64 = 1.0) -> Float64

Compute the total constraint violation, defined as the sum of positive parts:

    V = ∑ᵢ max(0, cᵢ(u))

Returns `0.0` when all constraints are satisfied.

# Arguments
- `controls`: control amplitudes matrix of shape `[n_controls × n_timesteps]`
- `constraints`: list of constraint objects
- `dt`: timestep duration (used by `EnergyConstraint`)
"""
function constraint_violation(controls::Matrix{Float64},
                               constraints::Vector{<:AbstractConstraint};
                               dt::Float64 = 1.0)::Float64
    total = 0.0
    for c in constraints
        v = _evaluate_constraint(c, controls, dt)
        if v > 0.0
            total += v
        end
    end
    return total
end

# ──────────────────────────────────────────────────────────────────────────────
# Projection onto feasible set
# ──────────────────────────────────────────────────────────────────────────────

"""
    project_single_constraint(controls::Matrix{Float64}, c::AbstractConstraint;
                               dt::Float64 = 1.0) -> Matrix{Float64}

Project `controls` onto the feasible set of constraint `c`.

- `BoundConstraint`: clamp element-wise to `[lower, upper]`
- `PowerConstraint`: if `‖u‖² > P_max`, rescale `u ← u · sqrt(P_max)/‖u‖`
- `BandwidthConstraint`: clamp all amplitudes to `[-BW_max, BW_max]`
- `EnergyConstraint`: if energy > `E_max`, rescale to satisfy with equality
- `CustomConstraint`: no closed-form projection; returns controls unchanged with a warning
"""
function project_single_constraint(controls::Matrix{Float64}, c::BoundConstraint;
                                    dt::Float64 = 1.0)
    result = copy(controls)
    idxs   = isempty(c.control_indices) ? axes(controls, 1) : c.control_indices
    @inbounds for j in idxs, k in axes(controls, 2)
        result[j,k] = clamp(result[j,k], c.lower, c.upper)
    end
    return result
end

function project_single_constraint(controls::Matrix{Float64}, c::PowerConstraint;
                                    dt::Float64 = 1.0)
    power = sum(abs2, controls)
    if power > c.P_max
        return controls .* (sqrt(c.P_max) / sqrt(power))
    end
    return copy(controls)
end

function project_single_constraint(controls::Matrix{Float64}, c::BandwidthConstraint;
                                    dt::Float64 = 1.0)
    return clamp.(controls, -c.BW_max, c.BW_max)
end

function project_single_constraint(controls::Matrix{Float64}, c::EnergyConstraint;
                                    dt::Float64 = 1.0)
    energy = dt * sum(abs2, controls)
    if energy > c.E_max
        return controls .* sqrt(c.E_max / energy)
    end
    return copy(controls)
end

function project_single_constraint(controls::Matrix{Float64}, c::CustomConstraint;
                                    dt::Float64 = 1.0)
    @warn "No closed-form projection available for CustomConstraint '$(c.name)'. " *
          "Controls returned unchanged. Consider using penalty or augmented_lagrangian."
    return copy(controls)
end

"""
    project_onto_constraints(controls::Matrix{Float64},
                              constraints::Vector{<:AbstractConstraint};
                              dt::Float64 = 1.0,
                              n_iter::Int = 5) -> Matrix{Float64}

Project controls onto the intersection of all constraint feasible sets using
Dykstra's alternating projection algorithm.

For a single constraint the projection is exact; for multiple constraints
Dykstra's algorithm converges to the nearest point in the intersection.

# Arguments
- `controls`: initial control amplitudes `[n_controls × n_timesteps]`
- `constraints`: list of constraints to satisfy
- `dt`: timestep size (needed for `EnergyConstraint`)
- `n_iter`: number of alternating projection passes (5 is typically sufficient)
"""
function project_onto_constraints(controls::Matrix{Float64},
                                   constraints::Vector{<:AbstractConstraint};
                                   dt::Float64 = 1.0,
                                   n_iter::Int = 5)::Matrix{Float64}
    isempty(constraints) && return copy(controls)

    # Dykstra's algorithm: maintain incremental corrections per constraint
    u = copy(controls)
    p = [zeros(size(controls)) for _ in constraints]

    for _ in 1:n_iter
        for (i, c) in enumerate(constraints)
            y    = u .+ p[i]
            proj = project_single_constraint(y, c; dt=dt)
            p[i] = y .- proj
            u    = proj
        end
    end
    return u
end

# ──────────────────────────────────────────────────────────────────────────────
# Penalty method
# ──────────────────────────────────────────────────────────────────────────────

"""
    penalty_method_optimize(system, target, controls_init::Matrix{Float64},
                             constraints::Vector{<:AbstractConstraint},
                             config::ConstrainedConfig) -> NamedTuple

Quadratic penalty method for constrained optimization.

Solves a sequence of unconstrained problems:

    F_pen(u; λ) = F(u) - λ · ∑ᵢ max(0, cᵢ(u))²

The penalty λ is multiplied by `config.penalty_growth` after each outer
iteration in which any constraint is violated.

# Algorithm
1. Set λ = `config.penalty_initial`
2. Run gradient ascent on `F_pen(u; λ)` for a fixed number of inner steps
3. If `constraint_violation(u) < config.violation_tol`: converge
4. Else: λ ← λ · `config.penalty_growth`, goto 2

Returns a named tuple with fields `controls`, `fidelity`, `fidelity_history`,
`converged`, `iterations`, `method`, and `metadata`.
"""
function penalty_method_optimize(system, target, controls_init::Matrix{Float64},
                                  constraints::Vector{<:AbstractConstraint},
                                  config::ConstrainedConfig)
    controls = copy(controls_init)
    λ        = config.penalty_initial
    dt       = _get_dt(system)

    best_controls = copy(controls)
    best_fidelity = -Inf
    fidelity_hist = Float64[]
    λ_hist        = Float64[λ]
    converged     = false

    for outer in 1:config.max_iter
        # Build augmented objective and its gradient
        function penalized_fidelity(u)
            f   = _compute_fidelity(system, target, u)
            pen = 0.0
            for c in constraints
                v = _evaluate_constraint(c, u, dt)
                v > 0.0 && (pen += v^2)
            end
            return f - λ * pen
        end

        function penalized_gradient(u)
            gf = _compute_gradient(system, target, u)
            for c in constraints
                v = _evaluate_constraint(c, u, dt)
                if v > 0.0
                    gc = _constraint_gradient(c, u, dt)
                    gf .-= 2λ * v .* gc
                end
            end
            return gf
        end

        # Inner gradient ascent with Armijo line search
        n_inner = max(10, 200 ÷ config.max_iter)
        for _ in 1:n_inner
            g        = penalized_gradient(controls)
            α        = _backtracking_line_search_matrix(penalized_fidelity, controls, g, g)
            controls = controls .+ α .* g
        end

        fid  = _compute_fidelity(system, target, controls)
        viol = constraint_violation(controls, constraints; dt=dt)
        push!(fidelity_hist, fid)

        if fid > best_fidelity
            best_fidelity = fid
            best_controls = copy(controls)
        end

        config.verbose && @printf(
            "[Penalty outer=%3d] fidelity=%.6f  violation=%.2e  λ=%.2e\n",
            outer, fid, viol, λ)

        if viol < config.violation_tol
            converged = true
            break
        end

        λ *= config.penalty_growth
        push!(λ_hist, λ)
        if config.check_invariants
            ok, msg = check_penalty_weight_growth(λ_hist)
            _assert_invariant(ok, msg, :penalty_weight_growth,
                              (; outer=outer, λ=λ))
        end
    end

    return (
        controls         = best_controls,
        fidelity         = best_fidelity,
        fidelity_history = fidelity_hist,
        converged        = converged,
        n_iterations     = length(fidelity_hist),
        method           = "penalty",
        metadata         = Dict{String,Any}(
            "final_penalty" => config.penalty_initial *
                               config.penalty_growth ^ max(0, length(fidelity_hist) - 1))
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Augmented Lagrangian method
# ──────────────────────────────────────────────────────────────────────────────

"""
    augmented_lagrangian_optimize(system, target, controls_init::Matrix{Float64},
                                   constraints::Vector{<:AbstractConstraint},
                                   config::ConstrainedConfig) -> NamedTuple

Hestenes–Powell augmented Lagrangian method for constrained optimization.

Maintains Lagrange multiplier estimates `μᵢ` that are updated after each inner
minimization:

    μᵢ ← max(0, μᵢ + λ · cᵢ(u))

The augmented Lagrangian objective is:

    L(u; μ, λ) = F(u) - ∑ᵢ [μᵢ · cᵢ(u) + (λ/2) · max(0, cᵢ(u) + μᵢ/λ)²]

Advantages over pure penalty method: multiplier updates allow convergence
without driving λ → ∞, avoiding numerical ill-conditioning.

Returns a named tuple with fields `controls`, `fidelity`, `fidelity_history`,
`converged`, `iterations`, `method`, and `metadata`.
"""
function augmented_lagrangian_optimize(system, target, controls_init::Matrix{Float64},
                                        constraints::Vector{<:AbstractConstraint},
                                        config::ConstrainedConfig)
    controls = copy(controls_init)
    λ        = config.penalty_initial
    dt       = _get_dt(system)
    nc       = length(constraints)
    μ        = zeros(nc)             # Lagrange multiplier estimates

    best_controls = copy(controls)
    best_fidelity = -Inf
    fidelity_hist = Float64[]
    converged     = false

    for outer in 1:config.max_iter
        μ_snap = copy(μ)

        function auglag_fidelity(u)
            f   = _compute_fidelity(system, target, u)
            pen = 0.0
            for (i, c) in enumerate(constraints)
                ci         = _evaluate_constraint(c, u, dt)
                ci_shifted = ci + μ_snap[i] / λ
                pen += (λ / 2.0) * max(0.0, ci_shifted)^2 - (μ_snap[i]^2) / (2λ)
            end
            return f - pen
        end

        function auglag_gradient(u)
            gf = _compute_gradient(system, target, u)
            for (i, c) in enumerate(constraints)
                ci         = _evaluate_constraint(c, u, dt)
                ci_shifted = ci + μ_snap[i] / λ
                if ci_shifted > 0.0
                    gc = _constraint_gradient(c, u, dt)
                    gf .-= λ * ci_shifted .* gc
                end
            end
            return gf
        end

        # Inner gradient ascent steps
        n_inner = max(10, 200 ÷ config.max_iter)
        for _ in 1:n_inner
            g        = auglag_gradient(controls)
            α        = _backtracking_line_search_matrix(auglag_fidelity, controls, g, g)
            controls = controls .+ α .* g
        end

        # Update Lagrange multipliers (dual ascent)
        for (i, c) in enumerate(constraints)
            ci   = _evaluate_constraint(c, controls, dt)
            μ[i] = max(0.0, μ[i] + λ * ci)
        end

        fid  = _compute_fidelity(system, target, controls)
        viol = constraint_violation(controls, constraints; dt=dt)
        push!(fidelity_hist, fid)

        if fid > best_fidelity
            best_fidelity = fid
            best_controls = copy(controls)
        end

        config.verbose && @printf(
            "[AugLag outer=%3d] fidelity=%.6f  violation=%.2e  λ=%.2e  |μ|=%.2e\n",
            outer, fid, viol, λ, norm(μ))

        if viol < config.violation_tol
            converged = true
            break
        end

        # Grow penalty if violation is not decreasing sufficiently
        prev_viol = outer > 1 ? max(0.0, fidelity_hist[end-1] - fidelity_hist[end]) : Inf
        if outer > 3 && viol > 0.25 * abs(get(fidelity_hist, length(fidelity_hist)-1, viol))
            λ *= config.penalty_growth
        end
    end

    return (
        controls         = best_controls,
        fidelity         = best_fidelity,
        fidelity_history = fidelity_hist,
        converged        = converged,
        n_iterations     = length(fidelity_hist),
        method           = "augmented_lagrangian",
        metadata         = Dict{String,Any}(
            "final_multipliers" => copy(μ),
            "final_penalty"     => λ)
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Projection-based constrained optimization
# ──────────────────────────────────────────────────────────────────────────────

"""
    projection_optimize(system, target, controls_init::Matrix{Float64},
                         constraints::Vector{<:AbstractConstraint},
                         config::ConstrainedConfig) -> NamedTuple

Projected gradient ascent for constrained quantum control optimization.

At each iteration, takes a gradient ascent step and then projects the result
onto the feasible set defined by all constraints:

    u_{k+1} = P_C[u_k + αk · ∇F(u_k)]

Projection uses Dykstra's alternating projection algorithm for intersections
of multiple constraint sets. Most efficient when all constraints have cheap
closed-form projections (bounds, power, energy).

Returns a named tuple with fields `controls`, `fidelity`, `fidelity_history`,
`converged`, `iterations`, `method`, and `metadata`.
"""
function projection_optimize(system, target, controls_init::Matrix{Float64},
                               constraints::Vector{<:AbstractConstraint},
                               config::ConstrainedConfig)
    dt       = _get_dt(system)
    controls = project_onto_constraints(copy(controls_init), constraints; dt=dt)

    best_controls = copy(controls)
    best_fidelity = _compute_fidelity(system, target, controls)
    fidelity_hist = Float64[best_fidelity]
    converged     = false

    α = 0.01  # initial step size (adaptive)

    for iter in 1:config.max_iter
        g         = _compute_gradient(system, target, controls)
        u_trial   = controls .+ α .* g
        u_proj    = project_onto_constraints(u_trial, constraints; dt=dt)
        fid_trial = _compute_fidelity(system, target, u_proj)

        if fid_trial >= best_fidelity - 1e-12
            controls      = u_proj
            best_fidelity = max(best_fidelity, fid_trial)
            if fid_trial >= best_fidelity
                best_controls = copy(controls)
            end
            α = min(α * 1.1, 1.0)
        else
            α = max(α * 0.5, 1e-12)
        end

        push!(fidelity_hist, _compute_fidelity(system, target, controls))
        viol = constraint_violation(controls, constraints; dt=dt)

        config.verbose && iter % 50 == 0 && @printf(
            "[Projection iter=%4d] fidelity=%.6f  violation=%.2e  α=%.2e\n",
            iter, fidelity_hist[end], viol, α)

        if viol < config.violation_tol && α < 1e-11
            converged = true
            break
        end
    end

    return (
        controls         = best_controls,
        fidelity         = best_fidelity,
        fidelity_history = fidelity_hist,
        converged        = converged,
        n_iterations     = length(fidelity_hist),
        method           = "projection",
        metadata         = Dict{String,Any}()
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

"""
    constrained_optimize(system, target, controls_init::Matrix{Float64},
                          constraints::Vector{<:AbstractConstraint};
                          config::ConstrainedConfig = ConstrainedConfig()) -> NamedTuple

Top-level constrained optimization dispatcher for Pulsar.

Selects the algorithm based on `config.constraint_method`:

| `constraint_method`       | Best for                                                 |
|---------------------------|----------------------------------------------------------|
| `"projection"`            | Bound/bandwidth/power/energy with cheap projections      |
| `"penalty"`               | Simple general constraints; may need high penalty        |
| `"augmented_lagrangian"`  | General constraints; best convergence, avoids ill-cond.  |

# Arguments
- `system`: quantum system object (must provide `dt`, `fidelity_fn`, `gradient_fn`)
- `target`: optimization target (gate or state)
- `controls_init`: initial control amplitudes `[n_controls × n_timesteps]`
- `constraints`: `Vector{<:AbstractConstraint}` of active constraints
- `config`: algorithm configuration (`ConstrainedConfig`)

# Returns
Named tuple:
- `controls::Matrix{Float64}` — optimal constrained controls
- `fidelity::Float64` — achieved fidelity
- `fidelity_history::Vector{Float64}` — fidelity per outer iteration
- `converged::Bool` — whether constraint tolerance was met
- `iterations::Int` — number of outer iterations performed
- `method::String` — algorithm used
- `metadata::Dict` — algorithm-specific diagnostics

# Example
```julia
constraints = AbstractConstraint[
    BoundConstraint(-1.0, 1.0),
    EnergyConstraint(10.0)
]
cfg    = ConstrainedConfig(constraint_method = "augmented_lagrangian", verbose = true)
result = constrained_optimize(system, target, controls_init, constraints; config = cfg)
println("Fidelity: ", result.fidelity)
println("Converged: ", result.converged)
```
"""
function constrained_optimize(system, target, controls_init::Matrix{Float64},
                                constraints::Vector{<:AbstractConstraint};
                                config::ConstrainedConfig = ConstrainedConfig())

    t0     = time()
    method = config.constraint_method

    if method == "projection"
        result = projection_optimize(system, target, controls_init, constraints, config)
    elseif method == "penalty"
        result = penalty_method_optimize(system, target, controls_init, constraints, config)
    elseif method == "augmented_lagrangian"
        result = augmented_lagrangian_optimize(system, target, controls_init, constraints, config)
    else
        throw(ArgumentError("Unknown constraint_method: '$method'. " *
                            "Choose from: projection, penalty, augmented_lagrangian"))
    end

    # Final feasibility projection: penalty / augmented-Lagrangian methods may
    # return slightly infeasible iterates. Project onto the constraint set and
    # re-evaluate the fidelity so the user always sees feasible controls.
    dt           = _get_dt(system)
    proj_ctrls   = project_onto_constraints(copy(result.controls), constraints; dt=dt)
    proj_fid     = _compute_fidelity(system, target, proj_ctrls)
    result       = merge(result, (controls = proj_ctrls, fidelity = proj_fid))

    return merge(result, (total_time = time() - t0,))
end

# ControlSequence overload — unwrap and delegate to the Matrix{Float64} method
function constrained_optimize(system, target, controls_init::ControlSequence,
                                constraints::Vector{<:AbstractConstraint};
                                config::ConstrainedConfig = ConstrainedConfig())
    constrained_optimize(system, target,
                         Matrix{Float64}(controls_init.controls),
                         constraints; config=config)
end

# Generic AbstractMatrix overload — covers `Adjoint`, `Transpose`, etc.
function constrained_optimize(system, target,
                                controls_init::AbstractMatrix{<:Real},
                                constraints::Vector{<:AbstractConstraint};
                                config::ConstrainedConfig = ConstrainedConfig())
    constrained_optimize(system, target,
                         Matrix{Float64}(controls_init),
                         constraints; config=config)
end

# Backward-compatible Matrix + dt overload.
# Stash dt in the system's metadata so the internal _get_dt /
# _compute_fidelity / _compute_gradient helpers can reach it without
# threading the argument through every constrained optimizer body.
function constrained_optimize(system, target,
                                controls_init::AbstractMatrix{<:Real},
                                dt::Real,
                                constraints::Vector{<:AbstractConstraint};
                                config::ConstrainedConfig = ConstrainedConfig())
    if system isa AbstractQuantumSystem && hasproperty(system, :metadata)
        system.metadata["dt"] = Float64(dt)
    end
    n_c, n_t = size(controls_init)
    seq = ControlSequence(Matrix{Float64}(controls_init), Float64(dt),
                          Float64(dt) * n_t, n_t)
    constrained_optimize(system, target, seq, constraints; config=config)
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (thin wrappers — real implementations live in Core)
# ──────────────────────────────────────────────────────────────────────────────

"""
    _get_dt(system) -> Float64

Extract the timestep `dt` from a quantum system object.
Falls back to `1.0` if the field is not present.
"""
function _get_dt(system)::Float64
    if hasproperty(system, :dt)
        return Float64(system.dt)
    end
    if hasproperty(system, :metadata) && haskey(system.metadata, "dt")
        return Float64(system.metadata["dt"])
    end
    return 1.0
end

"""
    _compute_fidelity(system, target, controls::Matrix{Float64}) -> Float64

Evaluate the fidelity for `controls`. Calls `system.fidelity_fn(controls)` if
available; otherwise returns `0.0` (useful for isolated unit testing).
"""
function _compute_fidelity(system, target, controls::Matrix{Float64})::Float64
    if hasproperty(system, :fidelity_fn)
        return system.fidelity_fn(controls)
    end
    if system isa AbstractQuantumSystem && target isa QuantumTarget
        return evaluate_fidelity(system, target, controls, _get_dt(system))
    end
    return 0.0
end

"""
    _compute_gradient(system, target, controls::Matrix{Float64}) -> Matrix{Float64}

Evaluate the fidelity gradient ∂F/∂u. Calls `system.gradient_fn(controls)` if
available; otherwise returns a zero matrix.
"""
function _compute_gradient(system, target, controls::Matrix{Float64})::Matrix{Float64}
    if hasproperty(system, :gradient_fn)
        return system.gradient_fn(controls)
    end
    if system isa AbstractQuantumSystem && target isa QuantumTarget
        return grape_gradient(system, target, controls, _get_dt(system))
    end
    return zeros(size(controls))
end

"""
    _backtracking_line_search_matrix(f, u, direction, g;
                                      c = 0.1, rho = 0.5, α0 = 1.0) -> Float64

Armijo backtracking line search for matrix-valued iterate `u`.

Finds the largest `α ∈ {α0, α0·ρ, α0·ρ², ...}` such that the sufficient
decrease (Armijo) condition holds:

    f(u + α·d) ≥ f(u) + c·α·⟨g, d⟩_F

# Arguments
- `f`: objective function `Matrix{Float64} -> Float64`
- `u`: current iterate
- `direction`: search direction
- `g`: gradient at `u` (used to compute slope)
- `c`: Armijo constant (default `0.1`)
- `rho`: backtracking factor (default `0.5`)
- `α0`: initial step size (default `1.0`)
"""
function _backtracking_line_search_matrix(f, u::Matrix{Float64},
                                           direction::Matrix{Float64},
                                           g::Matrix{Float64};
                                           c::Float64   = 0.1,
                                           rho::Float64 = 0.5,
                                           α0::Float64  = 1.0)::Float64
    f0    = f(u)
    slope = dot(g, direction)
    α     = α0

    for _ in 1:50
        f(u .+ α .* direction) >= f0 + c * α * slope && return α
        α *= rho
    end
    return α
end

# ──────────────────────────────────────────────────────────────────────────────
# Constraint diagnostics
# ──────────────────────────────────────────────────────────────────────────────

"""
    constraint_summary(controls::Matrix{Float64},
                        constraints::Vector{<:AbstractConstraint};
                        dt::Float64 = 1.0) -> String

Return a human-readable summary of constraint satisfaction status.

# Example output
```
Constraint Status:
  [1] BoundConstraint   : SATISFIED (margin = 0.0234)
  [2] EnergyConstraint  : VIOLATED  (excess = 0.4170)
  Total violation: 0.4170
```
"""
function constraint_summary(controls::Matrix{Float64},
                              constraints::Vector{<:AbstractConstraint};
                              dt::Float64 = 1.0)::String
    lines = String["Constraint Status:"]
    total = 0.0
    for (i, c) in enumerate(constraints)
        v    = _evaluate_constraint(c, controls, dt)
        kind = string(nameof(typeof(c)))
        if v <= 0.0
            push!(lines, @sprintf("  [%d] %-28s SATISFIED (margin = %.4g)", i, kind, -v))
        else
            push!(lines, @sprintf("  [%d] %-28s VIOLATED  (excess = %.4g)", i, kind,  v))
            total += v
        end
    end
    push!(lines, @sprintf("  Total violation: %.6g", total))
    return join(lines, "\n")
end
