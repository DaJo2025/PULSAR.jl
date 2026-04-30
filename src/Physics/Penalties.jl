# Physics/Penalties.jl
# Penalty/regularization functions for quantum control optimization.
# Extracted from Core/Fidelity.jl.

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3b — Penalty functor hierarchy
#
# Callable structs that replace runtime Symbol dispatch.  Create a functor once
# (baking in the parameters), then call it on control matrices:
#
#   pen = SmoothnessPenalty(0.01)       # weight = 0.01
#   P   = pen(w)                        # → Float64 penalty value
#   G   = gradient(pen, w)              # → Matrix{Float64} gradient
#   P, G = value_and_gradient(pen, w)   # combined (avoids recompute)
#
# The legacy Symbol-based API (penalty_value, penalty_gradient) is preserved
# as thin wrappers and continues to work unchanged.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AbstractPenalty

Abstract supertype for all penalty functors.  Concrete subtypes are callable:
`pen(w::Matrix{Float64}) -> Float64`.
"""
abstract type AbstractPenalty end

"""
    NormSquarePenalty(weight=1.0)

Penalty functor for the norm-square (L2) regularization:

    P = weight × Σ_{j,k} w[j,k]²

Equivalent to `penalty_value(w; type=:NS, weight=weight)`.
"""
struct NormSquarePenalty <: AbstractPenalty
    weight :: Float64
    NormSquarePenalty(weight::Float64 = 1.0) = new(weight)
end

"""
    SpilloutPenalty(weight=1.0; l_bound=-1.0, u_bound=1.0)

Penalty functor for amplitude-bound violations (SNS penalty):

    P = weight × Σ_{j,k} [max(0, w−u_b)² + max(0, l_b−w)²]

Equivalent to `penalty_value(w; type=:SNS, weight=weight, ...)`.
"""
struct SpilloutPenalty <: AbstractPenalty
    weight  :: Float64
    l_bound :: Float64
    u_bound :: Float64
    function SpilloutPenalty(weight::Float64 = 1.0;
                              l_bound::Float64 = -1.0,
                              u_bound::Float64 = +1.0)
        l_bound < u_bound || throw(ArgumentError("l_bound must be < u_bound"))
        new(weight, l_bound, u_bound)
    end
end

"""
    AmplitudeSpilloutPenalty(weight=1.0; u_bound=1.0)

Penalty functor for circular (total-RF-amplitude) bound violation (SNSA penalty):

    P = weight × Σ_k max(0, ‖w[:, k]‖ − u_bound)²

Equivalent to `penalty_value(w; type=:SNSA, weight=weight, u_bound=u_bound)`.
"""
struct AmplitudeSpilloutPenalty <: AbstractPenalty
    weight  :: Float64
    u_bound :: Float64
    function AmplitudeSpilloutPenalty(weight::Float64 = 1.0; u_bound::Float64 = 1.0)
        u_bound > 0.0 || throw(ArgumentError("u_bound must be positive"))
        new(weight, u_bound)
    end
end

"""
    SmoothnessPenalty(weight=1.0)

Penalty functor for waveform smoothness (derivative norm-square, DNS):

    P = weight × Σ_{j,k} (w[j,k+1] − w[j,k])²

Equivalent to `penalty_value(w; type=:DNS, weight=weight)`.
"""
struct SmoothnessPenalty <: AbstractPenalty
    weight :: Float64
    SmoothnessPenalty(weight::Float64 = 1.0) = new(weight)
end

"""
    EnergyPenalty(weight=1.0; dt=nothing)

Penalty functor for pulse energy (dt-weighted norm-square):

    P = weight × Σ_{j,k} w[j,k]² × dt[k]

`dt` can be `nothing` (uniform unit steps), a scalar, or a `Vector{Float64}`.
Equivalent to `penalty_value(w; type=:energy, weight=weight, dt=dt)`.
"""
struct EnergyPenalty <: AbstractPenalty
    weight :: Float64
    dt     :: Union{Nothing, Vector{Float64}}
    function EnergyPenalty(weight::Float64 = 1.0; dt = nothing)
        _dt = isnothing(dt) ? nothing :
              (dt isa AbstractVector ? Vector{Float64}(dt) :
               throw(ArgumentError("dt must be a vector or nothing")))
        new(weight, _dt)
    end
end

# ── Callable interface ────────────────────────────────────────────────────────

(pen::NormSquarePenalty)(w::Matrix{Float64})         = pen.weight * sum(abs2, w)
(pen::SpilloutPenalty)(w::Matrix{Float64})           =
    pen.weight * _pen_scalar(w, :SNS, pen.l_bound, pen.u_bound, nothing)
(pen::AmplitudeSpilloutPenalty)(w::Matrix{Float64})  =
    pen.weight * _pen_scalar(w, :SNSA, -Inf, pen.u_bound, nothing)
(pen::SmoothnessPenalty)(w::Matrix{Float64})         =
    pen.weight * _pen_scalar(w, :DNS, -Inf, Inf, nothing)
(pen::EnergyPenalty)(w::Matrix{Float64})             =
    pen.weight * _pen_scalar(w, :energy, -Inf, Inf, pen.dt)

# ── Gradient interface ────────────────────────────────────────────────────────

"""
    gradient(pen::AbstractPenalty, w::Matrix{Float64}) -> Matrix{Float64}

Return the gradient ∂P/∂w scaled by `pen.weight`.
"""
function gradient(pen::AbstractPenalty, w::Matrix{Float64})::Matrix{Float64}
    _sym, lb, ub, dt_ = _pen_params(pen)
    return pen.weight .* _pen_grad_mat(w, _sym, lb, ub, dt_)
end

"""
    value_and_gradient(pen::AbstractPenalty, w::Matrix{Float64})
    -> (Float64, Matrix{Float64})

Compute penalty value and gradient in a single pass.
"""
function value_and_gradient(pen::AbstractPenalty, w::Matrix{Float64})
    _sym, lb, ub, dt_ = _pen_params(pen)
    P, G = _pen_val_grad(w, _sym, lb, ub, dt_)
    return pen.weight * P, pen.weight .* G
end

"""
    make_penalty_fns(ps::Vector{<:AbstractPenalty}) -> Vector{Function}
    make_penalty_fns(p::AbstractPenalty)            -> Vector{Function}

Build the `penalty_fns` closure list expected by `grape_optimize` et al. from
one or more [`AbstractPenalty`](@ref) functors. Each closure has the shape
`w -> p(w)::Float64`.

# Example
```julia
ps = AbstractPenalty[SmoothnessPenalty(1e-3), EnergyPenalty(1e-4, dt_vec)]
result = grape_optimize(sys, target, ctrl;
                        penalty_fns = make_penalty_fns(ps))
```
"""
make_penalty_fns(ps::Vector{<:AbstractPenalty}) = [w -> p(w) for p in ps]
make_penalty_fns(p::AbstractPenalty)            = [w -> p(w)]

"""
    make_penalty_grad_fns(ps::Vector{<:AbstractPenalty}) -> Vector{Function}
    make_penalty_grad_fns(p::AbstractPenalty)            -> Vector{Function}

Companion to [`make_penalty_fns`](@ref) that builds the matching
`penalty_grad_fns` list: each closure is `w -> gradient(p, w)::Matrix{Float64}`.
"""
make_penalty_grad_fns(ps::Vector{<:AbstractPenalty}) = [w -> gradient(p, w) for p in ps]
make_penalty_grad_fns(p::AbstractPenalty)            = [w -> gradient(p, w)]

# Internal helper: extract (symbol, l_b, u_b, dt) from a functor
_pen_params(pen::NormSquarePenalty)          = (:NS,     -Inf,         Inf,          nothing)
_pen_params(pen::SpilloutPenalty)            = (:SNS,    pen.l_bound,  pen.u_bound,  nothing)
_pen_params(pen::AmplitudeSpilloutPenalty)   = (:SNSA,   -Inf,         pen.u_bound,  nothing)
_pen_params(pen::SmoothnessPenalty)          = (:DNS,    -Inf,         Inf,          nothing)
_pen_params(pen::EnergyPenalty)              = (:energy, -Inf,         Inf,          pen.dt)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Penalty Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PENALTY_TYPES :: Tuple

All valid penalty type symbols for [`penalty_value`](@ref) and
[`penalty_gradient`](@ref).

| Symbol    | Name                      | Formula                                                         |
|:--------- |:------------------------- |:--------------------------------------------------------------- |
| `:NS`     | Norm-square               | Σ_{k,n} w[k,n]²                                                 |
| `:SNS`    | Spillout norm-sq          | Σ max(0, w−u_b)² + max(0, l_b−w)²  (per channel)               |
| `:SNSA`   | Spillout norm-sq amplitude| Σ_n max(0, A[n]−u_b)²  where A[n]=‖w[:,n]‖  (circular bound)  |
| `:DNS`    | Derivative norm-sq        | Σ_{k,n<N} (w[k,n+1] − w[k,n])²                                 |
| `:energy` | Energy (dt-weighted)      | Σ_{k,n} w[k,n]² × dt[n]                                        |
"""
const PENALTY_TYPES = (:NS, :SNS, :SNSA, :DNS, :energy)

"""
    penalty_value(w; type, weight=1.0, l_bound=-1.0, u_bound=1.0, dt=nothing)
    -> Float64

Penalty scalar P ≥ 0 for normalised waveform `w` [n_ctrl × n_t].
The effective contribution to the total objective is `−weight × P`.

# Keyword arguments
- `type`    — penalty type; see [`PENALTY_TYPES`](@ref)
- `weight`  — scalar multiplier (default 1.0)
- `l_bound` — lower bound for `:SNS` (default −1.0)
- `u_bound` — upper bound for `:SNS` (default +1.0)
- `dt`      — time-step vector (length n_t) for `:energy`; uniform if `nothing`

# Example
```julia
P = penalty_value(w; type=:DNS, weight=0.01)
```
"""
function penalty_value(w::Matrix{Float64};
                       type    :: Symbol,
                       weight  :: Float64 = 1.0,
                       l_bound :: Float64 = -1.0,
                       u_bound :: Float64 = +1.0,
                       dt                 = nothing)::Float64
    weight ≈ 0.0 && return 0.0
    return weight * _pen_scalar(w, type, l_bound, u_bound, dt)
end

"""
    penalty_gradient(w; type, weight=1.0, l_bound=-1.0, u_bound=1.0, dt=nothing)
    -> Matrix{Float64}

Gradient ∂P/∂w (same shape as `w`) scaled by `weight`.
The contribution to the GRAPE waveform update is `−weight × ∂P/∂w`.
"""
function penalty_gradient(w::Matrix{Float64};
                           type    :: Symbol,
                           weight  :: Float64 = 1.0,
                           l_bound :: Float64 = -1.0,
                           u_bound :: Float64 = +1.0,
                           dt                 = nothing)::Matrix{Float64}
    weight ≈ 0.0 && return zeros(size(w))
    return weight .* _pen_grad_mat(w, type, l_bound, u_bound, dt)
end

"""
    penalty_value_and_gradient(w; type, weight=1.0, l_bound=-1.0, u_bound=1.0, dt=nothing)
    -> (Float64, Matrix{Float64})

Compute penalty value **and** gradient in one pass (avoids recomputing shared sums).
Returns `(weight × P, weight × ∂P/∂w)`.
"""
function penalty_value_and_gradient(w::Matrix{Float64};
                                     type    :: Symbol,
                                     weight  :: Float64 = 1.0,
                                     l_bound :: Float64 = -1.0,
                                     u_bound :: Float64 = +1.0,
                                     dt                 = nothing)
    weight ≈ 0.0 && return (0.0, zeros(size(w)))
    P, G = _pen_val_grad(w, type, l_bound, u_bound, dt)
    return weight * P, weight .* G
end

# ── Internal implementations (single-pass value+gradient) ─────────────────

function _pen_val_grad(w::Matrix{Float64}, type::Symbol,
                        l_b::Float64, u_b::Float64, dt)
    n_ctrl, n_t = size(w)
    G = zeros(n_ctrl, n_t)
    F = 0.0

    if type == :NS
        @inbounds for idx in eachindex(w)
            F += w[idx]^2
            G[idx] = 2 * w[idx]
        end

    elseif type == :SNS
        @inbounds for idx in eachindex(w)
            v = w[idx]
            if v > u_b
                e = v - u_b;  F += e^2;  G[idx] = 2e
            elseif v < l_b
                e = v - l_b;  F += e^2;  G[idx] = 2e
            end
        end

    elseif type == :SNSA
        # Penalise total RF amplitude A[n] = ‖w[:,n]‖ exceeding u_bound.
        # P = Σ_n max(0, A[n] − u_b)²
        # ∂P/∂w[k,n] = 2 max(0, A[n]−u_b) · w[k,n] / A[n]   (0 when A[n]=0)
        @inbounds for n in 1:n_t
            A = sqrt(sum(abs2(w[k, n]) for k in 1:n_ctrl))
            spillout = max(0.0, A - u_b)
            F += spillout^2
            if spillout > 0.0 && A > 1e-14
                scale = 2.0 * spillout / A
                for k in 1:n_ctrl
                    G[k, n] += scale * w[k, n]
                end
            end
        end

    elseif type == :DNS
        @inbounds for k in 1:n_ctrl, n in 1:(n_t - 1)
            d = w[k, n + 1] - w[k, n]
            F           += d^2
            G[k, n]     -= 2d
            G[k, n + 1] += 2d
        end

    elseif type == :energy
        _dt = isnothing(dt) ? ones(Float64, n_t) : Float64.(dt)
        @inbounds for k in 1:n_ctrl, n in 1:n_t
            F       += w[k, n]^2 * _dt[n]
            G[k, n]  = 2 * w[k, n] * _dt[n]
        end

    else
        throw(ArgumentError(
            "Unknown penalty ':$type'. Valid: " * join(string.(PENALTY_TYPES), ", ")))
    end

    return F, G
end

# Scalar-only path (used when gradient is not needed)
function _pen_scalar(w::Matrix{Float64}, type::Symbol,
                      l_b::Float64, u_b::Float64, dt)::Float64
    if type == :NS
        return sum(abs2, w)
    elseif type == :SNS
        F = 0.0
        @inbounds for v in w
            v > u_b && (F += (v - u_b)^2)
            v < l_b && (F += (v - l_b)^2)
        end
        return F
    elseif type == :SNSA
        n_ctrl_s, n_t_s = size(w);  F = 0.0
        @inbounds for n in 1:n_t_s
            A = sqrt(sum(abs2(w[k, n]) for k in 1:n_ctrl_s))
            F += max(0.0, A - u_b)^2
        end
        return F
    elseif type == :DNS
        n_ctrl, n_t = size(w);  F = 0.0
        @inbounds for k in 1:n_ctrl, n in 1:(n_t - 1)
            F += (w[k, n + 1] - w[k, n])^2
        end
        return F
    elseif type == :energy
        _dt = isnothing(dt) ? ones(Float64, size(w, 2)) : Float64.(dt)
        F = 0.0
        @inbounds for k in 1:size(w,1), n in 1:size(w,2)
            F += w[k, n]^2 * _dt[n]
        end
        return F
    else
        throw(ArgumentError(
            "Unknown penalty ':$type'. Valid: " * join(string.(PENALTY_TYPES), ", ")))
    end
end

# Gradient-only path
function _pen_grad_mat(w::Matrix{Float64}, type::Symbol,
                        l_b::Float64, u_b::Float64, dt)::Matrix{Float64}
    n_ctrl, n_t = size(w)
    G = zeros(n_ctrl, n_t)
    if type == :NS
        @. G = 2 * w
    elseif type == :SNS
        @inbounds for idx in eachindex(w)
            v = w[idx]
            G[idx] = v > u_b ? 2(v - u_b) : v < l_b ? 2(v - l_b) : 0.0
        end
    elseif type == :SNSA
        n_ctrl_g, n_t_g = size(w)
        @inbounds for n in 1:n_t_g
            A = sqrt(sum(abs2(w[k, n]) for k in 1:n_ctrl_g))
            spillout = max(0.0, A - u_b)
            if spillout > 0.0 && A > 1e-14
                scale = 2.0 * spillout / A
                for k in 1:n_ctrl_g
                    G[k, n] += scale * w[k, n]
                end
            end
        end
    elseif type == :DNS
        @inbounds for k in 1:n_ctrl, n in 1:(n_t - 1)
            d = w[k, n + 1] - w[k, n]
            G[k, n]     -= 2d
            G[k, n + 1] += 2d
        end
    elseif type == :energy
        _dt = isnothing(dt) ? ones(Float64, n_t) : Float64.(dt)
        @inbounds for k in 1:n_ctrl, n in 1:n_t
            G[k, n] = 2 * w[k, n] * _dt[n]
        end
    else
        throw(ArgumentError(
            "Unknown penalty ':$type'. Valid: " * join(string.(PENALTY_TYPES), ", ")))
    end
    return G
end

# MRI-specific penalties (sar_penalty, slew_rate_penalty, etc.) are in Physics/MRPhysics.jl
# (loaded after Types/BlochSystem.jl)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4b — External-package-inspired penalty functors (waveform-only)
#
# Quandary, Spinach and qopt provide several penalty styles PULSAR did not
# previously expose as first-class functors.  The five subtypes below all
# operate on the waveform alone, so they slot into the existing AbstractPenalty
# interface and the `make_penalty_fns` / `make_penalty_grad_fns` helpers.
#
# Trajectory-dependent (state variance), parameterisation-coefficient-dependent
# (Tikhonov on θ), and spectral-density-weighted penalties from the same
# packages are deferred until the parameterisation layer (Theme 2) and a
# trajectory-aware penalty interface land.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TotalEnergyBudget(weight, E_max; dt=nothing)

One-sided penalty enforcing a hard cap on integrated drive energy
(Quandary `gamma_energy`):

    P = weight × max(0, E(w) − E_max)²,    E(w) = Σ_{j,k} w[j,k]² × dt[k]

`dt` follows the same convention as [`EnergyPenalty`](@ref): `nothing` for
uniform unit steps, or a `Vector{Float64}` of per-step durations.  Use this
when a soft `EnergyPenalty` is insufficient and only over-budget pulses
should be punished.
"""
struct TotalEnergyBudget <: AbstractPenalty
    weight :: Float64
    E_max  :: Float64
    dt     :: Union{Nothing, Vector{Float64}}
    function TotalEnergyBudget(weight::Float64, E_max::Float64;
                                dt = nothing)
        E_max ≥ 0.0 || throw(ArgumentError("E_max must be non-negative"))
        _dt = isnothing(dt) ? nothing :
              (dt isa AbstractVector ? Vector{Float64}(dt) :
               throw(ArgumentError("dt must be a vector or nothing")))
        new(weight, E_max, _dt)
    end
end

"""
    MirrorSymmetryPenalty(weight)

Enforce time-reversal symmetry of the waveform (Spinach MSY):

    P = weight × Σ_{j,k} (w[j,k] − w[j, N+1−k])²

Useful when the desired pulse profile is palindromic (e.g. self-refocusing
pulses, BIR-4 envelopes).  Minimum is achieved when `w[j,k] = w[j, N+1−k]`
for every channel and time index.
"""
struct MirrorSymmetryPenalty <: AbstractPenalty
    weight :: Float64
    MirrorSymmetryPenalty(weight::Float64 = 1.0) = new(weight)
end

"""
    AsymmetryPenalty(weight)

Enforce time-antisymmetric waveforms (Spinach ASY):

    P = weight × Σ_{j,k} (w[j,k] + w[j, N+1−k])²

Minimum is achieved when `w[j,k] = −w[j, N+1−k]`.  Useful for refocusing
gradient lobes and odd-symmetry shaped pulses.
"""
struct AsymmetryPenalty <: AbstractPenalty
    weight :: Float64
    AsymmetryPenalty(weight::Float64 = 1.0) = new(weight)
end

"""
    CrossCouplingPenalty(weight)

Discourage simultaneous activation of distinct control channels (synthesis
of leakage / spillover ideas):

    P = weight × Σ_k Σ_{i<j} (w[i,k] · w[j,k])²

Per time step, every pair `(i, j)` of distinct channels contributes the
square of their product.  Minimum is achieved when at most one control is
non-zero at any given time slice — useful when channels share hardware or
crosstalk is destructive.
"""
struct CrossCouplingPenalty <: AbstractPenalty
    weight :: Float64
    CrossCouplingPenalty(weight::Float64 = 1.0) = new(weight)
end

"""
    InterpolatedTikhonov(weight, w_ref)

Tikhonov-style regularisation toward a reference waveform
(Quandary `gamma_tik0_interpolate`):

    P = weight × Σ_{j,k} (w[j,k] − w_ref[j,k])²

When warm-starting from a known good waveform, this keeps the optimiser from
drifting too far while still allowing fine-tuning.  When `w_ref` is zero
this reduces to plain L2 ([`NormSquarePenalty`](@ref)).
"""
struct InterpolatedTikhonov <: AbstractPenalty
    weight :: Float64
    w_ref  :: Matrix{Float64}
    function InterpolatedTikhonov(weight::Float64, w_ref::AbstractMatrix{<:Real})
        new(weight, Matrix{Float64}(w_ref))
    end
end

# ── Callable interface ────────────────────────────────────────────────────────

function (pen::TotalEnergyBudget)(w::Matrix{Float64})::Float64
    pen.weight ≈ 0.0 && return 0.0
    n_t = size(w, 2)
    _dt = isnothing(pen.dt) ? nothing : pen.dt
    _dt === nothing || length(_dt) == n_t ||
        throw(DimensionMismatch("dt length $(length(_dt)) ≠ n_t $n_t"))
    E = 0.0
    @inbounds for k in 1:n_t
        dk = _dt === nothing ? 1.0 : _dt[k]
        for j in axes(w, 1)
            E += w[j, k]^2 * dk
        end
    end
    over = E - pen.E_max
    return over > 0.0 ? pen.weight * over^2 : 0.0
end

function (pen::MirrorSymmetryPenalty)(w::Matrix{Float64})::Float64
    pen.weight ≈ 0.0 && return 0.0
    n_ctrl, n_t = size(w)
    F = 0.0
    @inbounds for k in 1:n_t
        kr = n_t + 1 - k
        kr < k && break
        # Off-diagonal pairs (k < kr) appear twice in the canonical
        # full-grid sum; the midpoint (k == kr, only when n_t is odd)
        # appears once. Match by weighting accordingly.
        mult = (k == kr) ? 1.0 : 2.0
        for j in 1:n_ctrl
            d = w[j, k] - w[j, kr]
            F += mult * d^2
        end
    end
    return pen.weight * F
end

function (pen::AsymmetryPenalty)(w::Matrix{Float64})::Float64
    pen.weight ≈ 0.0 && return 0.0
    n_ctrl, n_t = size(w)
    F = 0.0
    @inbounds for k in 1:n_t
        kr = n_t + 1 - k
        kr < k && break
        mult = (k == kr) ? 1.0 : 2.0
        for j in 1:n_ctrl
            s = w[j, k] + w[j, kr]
            F += mult * s^2
        end
    end
    return pen.weight * F
end

function (pen::CrossCouplingPenalty)(w::Matrix{Float64})::Float64
    pen.weight ≈ 0.0 && return 0.0
    n_ctrl, n_t = size(w)
    n_ctrl ≥ 2 || return 0.0
    F = 0.0
    @inbounds for k in 1:n_t, i in 1:(n_ctrl - 1), j in (i + 1):n_ctrl
        F += (w[i, k] * w[j, k])^2
    end
    return pen.weight * F
end

function (pen::InterpolatedTikhonov)(w::Matrix{Float64})::Float64
    pen.weight ≈ 0.0 && return 0.0
    size(w) == size(pen.w_ref) ||
        throw(DimensionMismatch("waveform size $(size(w)) ≠ reference $(size(pen.w_ref))"))
    F = 0.0
    @inbounds for idx in eachindex(w)
        F += (w[idx] - pen.w_ref[idx])^2
    end
    return pen.weight * F
end

# ── Gradient interface ────────────────────────────────────────────────────────

function gradient(pen::TotalEnergyBudget, w::Matrix{Float64})::Matrix{Float64}
    G = zeros(size(w))
    pen.weight ≈ 0.0 && return G
    n_t = size(w, 2)
    _dt = isnothing(pen.dt) ? nothing : pen.dt
    _dt === nothing || length(_dt) == n_t ||
        throw(DimensionMismatch("dt length $(length(_dt)) ≠ n_t $n_t"))
    E = 0.0
    @inbounds for k in 1:n_t
        dk = _dt === nothing ? 1.0 : _dt[k]
        for j in axes(w, 1)
            E += w[j, k]^2 * dk
        end
    end
    over = E - pen.E_max
    over > 0.0 || return G
    coeff = pen.weight * 4.0 * over   # ∂P/∂E = weight × 2 × over;  ∂E/∂w = 2 w dt
    @inbounds for k in 1:n_t
        dk = _dt === nothing ? 1.0 : _dt[k]
        for j in axes(w, 1)
            G[j, k] = coeff * w[j, k] * dk
        end
    end
    return G
end

function gradient(pen::MirrorSymmetryPenalty, w::Matrix{Float64})::Matrix{Float64}
    G = zeros(size(w))
    pen.weight ≈ 0.0 && return G
    n_ctrl, n_t = size(w)
    coeff = 4.0 * pen.weight   # 2× from off-diagonal-pair counting × 2 from d/dw of d²
    @inbounds for k in 1:n_t
        kr = n_t + 1 - k
        kr < k && break
        if k == kr
            # midpoint of an odd-length grid contributes (w − w)² = 0 with zero gradient
            continue
        end
        for j in 1:n_ctrl
            d = w[j, k] - w[j, kr]
            G[j, k]  += coeff * d
            G[j, kr] -= coeff * d
        end
    end
    return G
end

function gradient(pen::AsymmetryPenalty, w::Matrix{Float64})::Matrix{Float64}
    G = zeros(size(w))
    pen.weight ≈ 0.0 && return G
    n_ctrl, n_t = size(w)
    coeff = 4.0 * pen.weight
    @inbounds for k in 1:n_t
        kr = n_t + 1 - k
        kr < k && break
        if k == kr
            # midpoint contributes (2 w[k])² = 4 w[k]² with gradient 8 w[k];
            # the canonical doubled-sum convention treats this self-pair once.
            for j in 1:n_ctrl
                G[j, k] += 4.0 * pen.weight * 2.0 * w[j, k]
            end
            continue
        end
        for j in 1:n_ctrl
            s = w[j, k] + w[j, kr]
            G[j, k]  += coeff * s
            G[j, kr] += coeff * s
        end
    end
    return G
end

function gradient(pen::CrossCouplingPenalty, w::Matrix{Float64})::Matrix{Float64}
    G = zeros(size(w))
    pen.weight ≈ 0.0 && return G
    n_ctrl, n_t = size(w)
    n_ctrl ≥ 2 || return G
    coeff = 2.0 * pen.weight
    @inbounds for k in 1:n_t, i in 1:(n_ctrl - 1), j in (i + 1):n_ctrl
        # ∂[(wi wj)²]/∂wi = 2 wi wj²;  ∂/∂wj = 2 wj wi²
        wi = w[i, k]; wj = w[j, k]
        G[i, k] += coeff * wi * wj^2
        G[j, k] += coeff * wj * wi^2
    end
    return G
end

function gradient(pen::InterpolatedTikhonov, w::Matrix{Float64})::Matrix{Float64}
    G = zeros(size(w))
    pen.weight ≈ 0.0 && return G
    size(w) == size(pen.w_ref) ||
        throw(DimensionMismatch("waveform size $(size(w)) ≠ reference $(size(pen.w_ref))"))
    coeff = 2.0 * pen.weight
    @inbounds for idx in eachindex(w)
        G[idx] = coeff * (w[idx] - pen.w_ref[idx])
    end
    return G
end

# ── Combined value-and-gradient (delegate to the two halves) ──────────────────
# These benefit less from shared state than the legacy _pen_val_grad path, so
# the simple delegation is fine and keeps the new code easy to audit.

value_and_gradient(pen::TotalEnergyBudget,    w::Matrix{Float64}) = (pen(w), gradient(pen, w))
value_and_gradient(pen::MirrorSymmetryPenalty, w::Matrix{Float64}) = (pen(w), gradient(pen, w))
value_and_gradient(pen::AsymmetryPenalty,      w::Matrix{Float64}) = (pen(w), gradient(pen, w))
value_and_gradient(pen::CrossCouplingPenalty,  w::Matrix{Float64}) = (pen(w), gradient(pen, w))
value_and_gradient(pen::InterpolatedTikhonov,  w::Matrix{Float64}) = (pen(w), gradient(pen, w))

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Quantum Computing Penalties
# ═══════════════════════════════════════════════════════════════════════════════

"""
    leakage_penalty(U_total, leakage_indices; weight=1.0) -> Float64

Penalise population transferred into leakage levels outside the computational
subspace.  `leakage_indices` is a vector of 1-based row indices (in the full
Hilbert space) that correspond to leakage levels (e.g. |2⟩, |3⟩, … of a
transmon).

    P_leak = weight × Σ_{i ∈ leakage_indices} Σ_j |U_total[i,j]|²

This counts the total column-norm of the rows that map into leakage.

# Arguments
- `U_total`         — dim × dim propagator matrix at end of pulse
- `leakage_indices` — vector of row indices (1-based) for leakage levels
- `weight`          — scalar multiplier (default 1.0)

# Returns
Non-negative Float64 penalty value.

# Example
```julia
# 3-level transmon: |0⟩,|1⟩ computational, |2⟩ leakage
P = leakage_penalty(U_total, [3]; weight=0.1)
```
"""
function leakage_penalty(U_total       :: Matrix{ComplexF64},
                          leakage_indices :: Vector{Int};
                          weight          :: Float64 = 1.0)::Float64
    weight ≈ 0.0 && return 0.0
    P = 0.0
    for i in leakage_indices
        for j in axes(U_total, 2)
            P += abs2(U_total[i, j])
        end
    end
    return weight * P
end

"""
    leakage_gradient(P_fwd, Q_bwd, H_controls, leakage_indices, dt; weight=1.0)
    -> Matrix{Float64}

Analytical GRAPE gradient of [`leakage_penalty`](@ref) with respect to
piecewise-constant control amplitudes.

Using the GRAPE adjoint method with an auxiliary backward propagator seeded by
the leakage projector Π_L = Σ_{i ∈ leakage_indices} |i⟩⟨i|:

    ∂P_leak/∂u_j[k] = weight × 2 Re[ Tr(Q_bwd[k]† (−i H_j dt) P_fwd[k]) ]

where the backward propagator Q_bwd is seeded with U_total^† Π_L instead of
the usual U_target.

# Arguments
- `P_fwd`           — Array{ComplexF64,3} of size (n_t+1, dim, dim): forward propagators
- `Q_bwd`           — Array{ComplexF64,3} of size (n_t+1, dim, dim): backward propagators
  seeded at U_total† Π_L U_total for leakage
- `H_controls`      — Vector of dim×dim Hermitian control Hamiltonians
- `leakage_indices` — row indices of leakage levels
- `dt`              — time step in seconds
- `weight`          — scalar multiplier (default 1.0)

# Returns
Matrix{Float64} of size (n_controls × n_timesteps).
"""
function leakage_gradient(P_fwd           :: Array{ComplexF64,3},
                           Q_bwd           :: Array{ComplexF64,3},
                           H_controls      :: Vector{Matrix{ComplexF64}},
                           leakage_indices :: Vector{Int},
                           dt              :: Float64;
                           weight          :: Float64 = 1.0)::Matrix{Float64}
    weight ≈ 0.0 && return zeros(length(H_controls), size(P_fwd, 1) - 1)
    n_t    = size(P_fwd, 1) - 1
    n_ctrl = length(H_controls)
    G      = zeros(n_ctrl, n_t)
    for k in 1:n_t
        Pk = @view P_fwd[k, :, :]
        Qk = @view Q_bwd[k, :, :]
        for j in 1:n_ctrl
            # ∂P_leak/∂u_j[k] = 2 Re[ Tr( Qk† (−i Hj dt) Pk ) ]
            inner = -im * dt * tr(Qk' * H_controls[j] * Pk)
            G[j, k] = weight * 2.0 * real(inner)
        end
    end
    return G
end

# ─── MS gate penalties (Mølmer-Sørensen, trapped-ion) ─────────────────────────

"""
    ms_closure_penalty(motional_amplitudes; weight=1.0) -> Float64

Penalise failure of the trapped-ion motional bus to return to the vacuum state
after an MS gate.

Under the Lamb-Dicke approximation the motional displacement at the end of the
gate sequence is characterised by a complex amplitude `α_m` for each sideband
mode `m`.  The penalty is:

    P_close = weight × Σ_m |α_m|²

Perfect closure (motional vacuum at gate end) gives P_close = 0.

# Arguments
- `motional_amplitudes` — Vector{ComplexF64} of displacement amplitudes α_m
- `weight`              — scalar multiplier (default 1.0)

# Example
```julia
# Two COM modes; both should return to origin
α = [0.02 + 0.01im, -0.005 + 0.003im]
P = ms_closure_penalty(α; weight=1.0)
```
"""
function ms_closure_penalty(motional_amplitudes :: AbstractVector{ComplexF64};
                              weight              :: Float64 = 1.0)::Float64
    weight ≈ 0.0 && return 0.0
    return weight * sum(abs2, motional_amplitudes)
end

"""
    ms_phase_penalty(acquired_phase, target_phase; weight=1.0) -> Float64

Penalise deviation of the accumulated geometric phase from the MS gate target.
The ideal MS(π/4) gate acquires a geometric phase of π/4.

    P_phase = weight × (acquired_phase − target_phase)²

# Arguments
- `acquired_phase` — Float64 geometric phase (radians) accumulated by pulse
- `target_phase`   — Float64 desired phase (default π/4 for MS gate)
- `weight`         — scalar multiplier (default 1.0)

# Example
```julia
P = ms_phase_penalty(Φ; target_phase=π/4, weight=0.5)
```
"""
function ms_phase_penalty(acquired_phase :: Float64,
                           target_phase   :: Float64 = π/4;
                           weight         :: Float64  = 1.0)::Float64
    weight ≈ 0.0 && return 0.0
    return weight * (acquired_phase - target_phase)^2
end

"""
    ms_closure_gradient(dα_dw, motional_amplitudes; weight=1.0) -> Matrix{Float64}

Gradient of [`ms_closure_penalty`](@ref) with respect to waveform amplitudes.

    ∂P_close/∂w[j,k] = weight × 2 Re[ conj(α_m) × ∂α_m/∂w[j,k] ]

# Arguments
- `dα_dw`               — Array{ComplexF64,3} of size (n_modes, n_ctrl, n_t): Jacobian
  ∂α_m/∂w[j,k] for each mode m, control j, time step k
- `motional_amplitudes` — Vector{ComplexF64} current amplitudes α_m
- `weight`              — scalar multiplier (default 1.0)
"""
function ms_closure_gradient(dα_dw              :: Array{ComplexF64,3},
                              motional_amplitudes :: AbstractVector{ComplexF64};
                              weight             :: Float64 = 1.0)::Matrix{Float64}
    n_modes, n_ctrl, n_t = size(dα_dw)
    G = zeros(n_ctrl, n_t)
    weight ≈ 0.0 && return G
    for m in 1:n_modes, j in 1:n_ctrl, k in 1:n_t
        G[j, k] += weight * 2.0 * real(conj(motional_amplitudes[m]) * dα_dw[m, j, k])
    end
    return G
end

"""
    ms_phase_gradient(dΦ_dw, acquired_phase, target_phase; weight=1.0)
    -> Matrix{Float64}

Gradient of [`ms_phase_penalty`](@ref) with respect to waveform amplitudes.

    ∂P_phase/∂w[j,k] = weight × 2(Φ − Φ_targ) × ∂Φ/∂w[j,k]

# Arguments
- `dΦ_dw`          — Matrix{Float64} of size (n_ctrl, n_t): Jacobian ∂Φ/∂w[j,k]
- `acquired_phase` — current accumulated phase
- `target_phase`   — target phase (default π/4)
- `weight`         — scalar multiplier (default 1.0)
"""
function ms_phase_gradient(dΦ_dw          :: Matrix{Float64},
                            acquired_phase  :: Float64,
                            target_phase    :: Float64 = π/4;
                            weight          :: Float64  = 1.0)::Matrix{Float64}
    weight ≈ 0.0 && return zeros(size(dΦ_dw))
    return (weight * 2.0 * (acquired_phase - target_phase)) .* dΦ_dw
end

# ─── Filter function penalty (dynamical decoupling / noise spectroscopy) ──────

"""
    filter_function_penalty(F_filter, S_noise, dω; weight=1.0) -> Float64

Spectral penalty based on the filter function formalism: the dephasing induced
by a noise spectrum S(ω) is

    χ = (1/π) ∫ dω S(ω) |F(ω)|² / ω²

approximated by the discrete sum:

    P_ff = weight × (1/π) Σ_k S_noise[k] × |F_filter[k]|² / max(ω_k², ε) × dω

Minimising P_ff makes the pulse a dynamical decoupling sequence that rejects the
given noise spectrum.

# Arguments
- `F_filter` — Vector{ComplexF64} filter function values F(ω_k) evaluated at
  positive frequencies ω_k (rad/s)
- `S_noise`  — Vector{Float64} one-sided noise power spectral density S(ω_k)
  (rad²/s units)
- `dω`       — Float64 frequency resolution (rad/s)
- `weight`   — scalar multiplier (default 1.0)

# Example
```julia
ω  = range(1e3, 1e7; length=512) .* 2π
S  = 1.0 ./ ω.^2    # 1/f noise
F  = filter_function(pulse_propagators, dt, ω)
P  = filter_function_penalty(F, S, step(ω); weight=1.0)
```
"""
function filter_function_penalty(F_filter :: AbstractVector{ComplexF64},
                                  S_noise  :: AbstractVector{Float64},
                                  dω       :: Float64;
                                  weight   :: Float64 = 1.0)::Float64
    weight ≈ 0.0 && return 0.0
    ε = 1e-20
    χ = 0.0
    for k in eachindex(F_filter)
        ω_k = dω * k   # approximate ω_k as index × dω
        χ += S_noise[k] * abs2(F_filter[k]) / max(ω_k^2, ε)
    end
    return weight * χ * dω / π
end

"""
    filter_function_gradient(dF_dw, F_filter, S_noise, dω; weight=1.0)
    -> Matrix{Float64}

Gradient of [`filter_function_penalty`](@ref) with respect to waveform amplitudes.

    ∂P_ff/∂w[j,k] = weight × (2/π) Σ_m S(ω_m)/ω_m² × Re[ conj(F[m]) × ∂F[m]/∂w[j,k] ] × dω

# Arguments
- `dF_dw`    — Array{ComplexF64,3} size (n_freq, n_ctrl, n_t): Jacobian ∂F(ω_m)/∂w[j,k]
- `F_filter` — Vector{ComplexF64} current filter function values
- `S_noise`  — Vector{Float64} noise PSD
- `dω`       — frequency resolution (rad/s)
- `weight`   — scalar multiplier (default 1.0)
"""
function filter_function_gradient(dF_dw    :: Array{ComplexF64,3},
                                   F_filter  :: AbstractVector{ComplexF64},
                                   S_noise   :: AbstractVector{Float64},
                                   dω        :: Float64;
                                   weight    :: Float64 = 1.0)::Matrix{Float64}
    n_freq, n_ctrl, n_t = size(dF_dw)
    G = zeros(n_ctrl, n_t)
    weight ≈ 0.0 && return G
    ε = 1e-20
    for m in 1:n_freq
        ω_m   = dω * m
        coeff = weight * 2.0 * dω / π * S_noise[m] / max(ω_m^2, ε)
        for j in 1:n_ctrl, k in 1:n_t
            G[j, k] += coeff * real(conj(F_filter[m]) * dF_dw[m, j, k])
        end
    end
    return G
end

