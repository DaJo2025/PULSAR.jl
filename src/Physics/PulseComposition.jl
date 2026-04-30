# ============================================================================
# Physics/PulseComposition.jl  —  Theme 9: hardware-aware pulse composition
#
# Lets the user wrap an optimised "middle" segment of a sequence with a fixed
# **prefix** pulse, a fixed **suffix** pulse, and an optional **dead-time**
# free-evolution gap.  The composition is plumbed into the optimiser via
# **boundary shifting**, which is mathematically equivalent to inserting the
# fixed propagators into the chain but does not require any change to the
# GRAPE / Krotov kernels:
#
#     ⟨ψ_target | U_suffix · U_dead · U_opt(θ) · U_prefix | ψ_init⟩
#       =  ⟨(U_suffix · U_dead)† · ψ_target | U_opt(θ) | U_prefix · ψ_init⟩
#
# So an optimisation with a `PulseComposition` is identical to an optimisation
# on the un-composited problem after replacing
#
#     ψ_init   →  ψ_init_eff   :=  U_prefix · ψ_init
#     ψ_target →  ψ_target_eff :=  (U_suffix · U_dead)† · ψ_target
#
# This file provides:
#   • [`compose_hard_pulse_propagator`](@ref) — build U from a sequence of
#     `CompositePulseSegment`s on a control basis (no drift evolution)
#   • [`dead_time_propagator`](@ref) — exp(−i H_drift · dead_time)
#   • [`PulseComposition`](@ref) — bundle of `(U_prefix, U_suffix, U_dead)`
#   • [`compose_effective_boundary`](@ref) — shift `(ρ_init, ρ_target)` pairs
#
# Reference (motivation):
#   Hogben, Krzystyniak, Charnock, Kuprov, "Spinach — a software library for
#   simulation of spin dynamics in large spin systems", J. Magn. Reson. 208,
#   179 (2011), §`optimcon` `prefix` / `suffix` / `deadtime` options.
# ============================================================================

using LinearAlgebra

# ---------------------------------------------------------------------------
# PulseComposition — bundle of fixed propagators surrounding the optimised
# segment.  Any of the three components may be `nothing` (= identity).
# ---------------------------------------------------------------------------

"""
    PulseComposition(; prefix=nothing, suffix=nothing, dead=nothing)

Container for fixed propagators surrounding the optimised segment of a
sequence.  All three fields are `Union{Nothing, Matrix{ComplexF64}}`.

* `prefix`  — `U_prefix` applied to the initial state *before* the optimised
              waveform.
* `suffix`  — `U_suffix` applied to the final state *after* the optimised
              waveform (and after `dead`, if any).
* `dead`    — `U_dead = exp(−i H_drift · t_dead)` free-evolution propagator
              inserted between the optimised segment and the suffix.

The convention is `U_full = U_suffix · U_dead · U_opt(θ) · U_prefix`.

Construct via the keyword constructor; use the `compose_*` helpers to build
the component matrices from `CompositePulseSegment` arrays or drift
Hamiltonians.
"""
struct PulseComposition
    prefix :: Union{Nothing, Matrix{ComplexF64}}
    suffix :: Union{Nothing, Matrix{ComplexF64}}
    dead   :: Union{Nothing, Matrix{ComplexF64}}
end

PulseComposition(; prefix = nothing, suffix = nothing, dead = nothing) =
    PulseComposition(_to_cmat(prefix), _to_cmat(suffix), _to_cmat(dead))

@inline _to_cmat(::Nothing)         = nothing
@inline _to_cmat(M::AbstractMatrix) = Matrix{ComplexF64}(M)

# ---------------------------------------------------------------------------
# compose_hard_pulse_propagator
# ---------------------------------------------------------------------------

"""
    compose_hard_pulse_propagator(segments, operators; rf_hz, x_index=1, y_index=2)
        → Matrix{ComplexF64}

Build the propagator for a sequence of hard composite-pulse segments.  Each
[`CompositePulseSegment`](@ref) has a flip angle (deg) and a phase (deg); the
segment Hamiltonian is

    H_seg = Ω · (cos φ · operators[x_index] + sin φ · operators[y_index])

with `Ω = 2π · rf_hz` and the segment duration set so that the flip angle
matches: `dt_seg = θ_rad / Ω`.

`operators` is the same vector of control operators that the optimiser uses
(typically `[Ix, Iy]` for spin-1/2 or `[σx/2, σy/2]` for a qubit).  No drift
evolution is included — use [`dead_time_propagator`](@ref) for that and
combine via [`PulseComposition`](@ref).

Returns the cumulative propagator `U_N · … · U_1` (segments applied
left-to-right in time order).
"""
function compose_hard_pulse_propagator(
    segments  :: AbstractVector{CompositePulseSegment},
    operators :: AbstractVector{<:AbstractMatrix};
    rf_hz     :: Real,
    x_index   :: Int = 1,
    y_index   :: Int = 2,
)
    rf_hz > 0 || throw(ArgumentError("rf_hz must be positive (got $rf_hz)"))
    1 ≤ x_index ≤ length(operators) ||
        throw(ArgumentError("x_index $x_index out of range 1:$(length(operators))"))
    1 ≤ y_index ≤ length(operators) ||
        throw(ArgumentError("y_index $y_index out of range 1:$(length(operators))"))
    isempty(segments) &&
        throw(ArgumentError("segments must be non-empty"))

    Ox = Matrix{ComplexF64}(operators[x_index])
    Oy = Matrix{ComplexF64}(operators[y_index])
    dim = size(Ox, 1)
    size(Oy) == (dim, dim) ||
        throw(DimensionMismatch("operators[$x_index] and operators[$y_index] " *
                                "must share dimensions"))
    Ω = 2π * Float64(rf_hz)
    U = Matrix{ComplexF64}(I, dim, dim)
    for seg in segments
        θ_rad = deg2rad(seg.flip_deg)
        φ_rad = deg2rad(seg.phase_deg)
        dt    = θ_rad / Ω
        H_seg = Ω .* (cos(φ_rad) .* Ox .+ sin(φ_rad) .* Oy)
        U     = compute_propagator(H_seg, dt) * U
    end
    return U
end

# ---------------------------------------------------------------------------
# dead_time_propagator
# ---------------------------------------------------------------------------

"""
    dead_time_propagator(H_drift, dead_time_s) → Matrix{ComplexF64}

Free-evolution propagator `exp(−i H_drift · dead_time_s)`.  `dead_time_s ≥ 0`
is required; `dead_time_s == 0` returns the identity.
"""
function dead_time_propagator(H_drift::AbstractMatrix, dead_time_s::Real)
    dead_time_s ≥ 0 ||
        throw(ArgumentError("dead_time_s must be ≥ 0 (got $dead_time_s)"))
    if dead_time_s == 0
        d = size(H_drift, 1)
        return Matrix{ComplexF64}(I, d, d)
    end
    return compute_propagator(Matrix{ComplexF64}(H_drift), Float64(dead_time_s))
end

# ---------------------------------------------------------------------------
# compose_effective_boundary
# ---------------------------------------------------------------------------

"""
    compose_effective_boundary(comp::PulseComposition, ρ_init, ρ_target)
        → (ρ_init_eff, ρ_target_eff)

Shift the boundary states for a `(ρ_init, ρ_target)` pair so that running an
optimiser on `(ρ_init_eff, ρ_target_eff)` is mathematically identical to
running on the original `(ρ_init, ρ_target)` with `comp` wrapping the
optimised segment.

The shift is

    ρ_init_eff   =  U_prefix · ρ_init
    ρ_target_eff =  (U_suffix · U_dead)† · ρ_target

with any `nothing` component treated as the identity.  Accepts both pure
states (`Vector{ComplexF64}`) and density matrices (`Matrix{ComplexF64}`):

* Pure state:    ρ_init_eff = U · ρ_init
* Density mat:   ρ_init_eff = U · ρ_init · U†

Vector / matrix dispatch is automatic.
"""
function compose_effective_boundary(comp::PulseComposition,
                                    ρ_init::AbstractVector{<:Number},
                                    ρ_target::AbstractVector{<:Number})
    U_pre   = comp.prefix
    U_post  = _compose_post(comp)
    ρ_i_eff = _apply_left(U_pre,  ρ_init)
    ρ_t_eff = _apply_left_adj(U_post, ρ_target)
    return ρ_i_eff, ρ_t_eff
end

function compose_effective_boundary(comp::PulseComposition,
                                    ρ_init::AbstractMatrix{<:Number},
                                    ρ_target::AbstractMatrix{<:Number})
    U_pre   = comp.prefix
    U_post  = _compose_post(comp)
    ρ_i_eff = _apply_left(U_pre,  ρ_init)
    ρ_t_eff = _apply_left_adj(U_post, ρ_target)
    return ρ_i_eff, ρ_t_eff
end

"""
    compose_effective_boundary(comp::PulseComposition,
                                ρ_inits::AbstractVector{<:AbstractVecOrMat},
                                ρ_targs::AbstractVector{<:AbstractVecOrMat})
        → (ρ_inits_eff, ρ_targs_eff)

Vector-of-pairs convenience overload.  Applies the boundary shift element-wise.
Lengths must match.
"""
function compose_effective_boundary(comp::PulseComposition,
                                    ρ_inits::AbstractVector{<:AbstractVecOrMat{<:Number}},
                                    ρ_targs::AbstractVector{<:AbstractVecOrMat{<:Number}})
    length(ρ_inits) == length(ρ_targs) ||
        throw(ArgumentError("ρ_inits and ρ_targs must have the same length " *
                            "(got $(length(ρ_inits)) and $(length(ρ_targs)))"))
    out_i = similar(ρ_inits)
    out_t = similar(ρ_targs)
    for k in eachindex(ρ_inits)
        out_i[k], out_t[k] = compose_effective_boundary(comp, ρ_inits[k], ρ_targs[k])
    end
    return out_i, out_t
end

# Internal: combined post-segment propagator U_suffix · U_dead.
function _compose_post(comp::PulseComposition)
    if comp.suffix === nothing && comp.dead === nothing
        return nothing
    elseif comp.suffix === nothing
        return comp.dead
    elseif comp.dead === nothing
        return comp.suffix
    else
        return comp.suffix * comp.dead
    end
end

# Internal: U · ψ for vectors, U · ρ · U† for matrices, identity if U === nothing.
@inline _apply_left(::Nothing, x) = x
@inline _apply_left(U::AbstractMatrix, ψ::AbstractVector) = U * ψ
@inline _apply_left(U::AbstractMatrix, ρ::AbstractMatrix) = U * ρ * adjoint(U)

@inline _apply_left_adj(::Nothing, x) = x
@inline _apply_left_adj(U::AbstractMatrix, ψ::AbstractVector) = adjoint(U) * ψ
@inline _apply_left_adj(U::AbstractMatrix, ρ::AbstractMatrix) =
    adjoint(U) * ρ * U
