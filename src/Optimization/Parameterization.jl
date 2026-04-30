# ============================================================
# Pulsar.jl — Control parameterization layer (Theme 2)
# ============================================================
#
# Maps an unconstrained optimisation vector `θ` to a piecewise-constant
# waveform `w :: Matrix{Float64}` with shape `[n_ctrl × n_t]`.
#
# Why a hierarchy?
#   • TanhParam / TanhSqParam / LogisticParam enforce |w| ≤ u_max with no
#     clipping and a smooth gradient near the bound — exactly the stability
#     fix the BM08 CNOT runs need.
#   • BSplineParam / HermiteParam / FourierParam / ChebyshevParam reduce the
#     degrees of freedom (Quandary, SIMPSON, GOAT) without changing the
#     downstream waveform consumers.
#   • PiecewiseConstant remains the default — `to_waveform(θ, ::PiecewiseConstant)`
#     is the identity, `waveform_jacobian(θ, ::PiecewiseConstant)` is `I`.
#     Existing code paths therefore see *no* behaviour change.
#
# Phase-1 status: PiecewiseConstant + Tanh / TanhSq / Logistic are wired up
# end-to-end (forward map, inverse map, Jacobian).  The basis-reduction
# parameterisations (BSpline, Hermite, Fourier, Chebyshev, Slepian, CRAB)
# are reserved — their constructors are validated but `to_waveform` etc.
# throw a guarded `ErrorException` until Phase 2/3 fills them in.
# ============================================================

"""
    AbstractControlParameterization

Supertype for forward maps `θ ↦ w(t)` consumed by GRAPE / L-BFGS / Krotov.
Subtypes provide three methods:

    to_waveform(θ, p, n_ctrl, n_t)         :: Matrix{Float64}
    from_waveform(w, p)                     :: Vector{Float64}
    waveform_jacobian(θ, p, n_ctrl, n_t)    :: AbstractMatrix{Float64}

Optimisers chain through `∇_θ F = J(θ)ᵀ · ∇_w F`.  For the default
[`PiecewiseConstant`](@ref), `J = I`, so existing code is unchanged.
"""
abstract type AbstractControlParameterization end

# ─────────────────────────────────────────────────────────────────────────────
# Built-in parameterisations
# ─────────────────────────────────────────────────────────────────────────────

"""
    PiecewiseConstant()

Identity parameterisation: `w = reshape(θ, n_ctrl, n_t)`.  The Jacobian is
the identity.  This is the legacy default and is exactly equivalent to
the existing waveform-as-parameter convention.
"""
struct PiecewiseConstant <: AbstractControlParameterization end

"""
    TanhParam(u_max::Float64)

Smooth bounded map  `w = u_max · tanh(θ)`.  Enforces `|w| ≤ u_max`
without clipping, with a non-zero gradient at the bound.

Inverse: `θ = atanh(w / u_max)`; defined only on `(-u_max, u_max)`.
"""
struct TanhParam <: AbstractControlParameterization
    u_max :: Float64
    function TanhParam(u_max::Real)
        u_max > 0 ||
            throw(ArgumentError("TanhParam u_max must be > 0, got $u_max"))
        return new(Float64(u_max))
    end
end

"""
    TanhSqParam(u_max::Float64)

`w = u_max · tanh(θ)²` — enforces `0 ≤ w ≤ u_max`.  Suitable for
non-negative drive amplitudes (RF magnitude in polar coordinates).
"""
struct TanhSqParam <: AbstractControlParameterization
    u_max :: Float64
    function TanhSqParam(u_max::Real)
        u_max > 0 ||
            throw(ArgumentError("TanhSqParam u_max must be > 0, got $u_max"))
        return new(Float64(u_max))
    end
end

"""
    LogisticParam(u_max::Float64; beta::Float64 = 1.0)

`w = u_max · (2 σ(β θ) − 1)` with `σ(x) = 1 / (1 + e^{-x})`.  Equivalent
to `TanhParam(u_max)` up to a `β` rescaling but with the steepness
adjustable independently of the bound.
"""
struct LogisticParam <: AbstractControlParameterization
    u_max :: Float64
    beta  :: Float64
    function LogisticParam(u_max::Real; beta::Real = 1.0)
        u_max > 0 ||
            throw(ArgumentError("LogisticParam u_max must be > 0, got $u_max"))
        beta > 0 ||
            throw(ArgumentError("LogisticParam beta must be > 0, got $beta"))
        return new(Float64(u_max), Float64(beta))
    end
end

# ── Reserved: basis-reduction parameterisations (Phase 2/3) ─────────────────

"""
    PhaseOnlyParam(amplitude::Float64,
                   phase_pairs::Vector{Tuple{Int,Int}})

Constant-amplitude phase-modulated parameterisation. The optimisation
variable is one phase per timestep per RF carrier; the underlying physics
still sees a 2-channel waveform.

For each tuple `(cx, cy)` in `phase_pairs`, raw control rows `cx` and
`cy` are tied together as
`w[cx, k] = A·cos(φ[k])`, `w[cy, k] = A·sin(φ[k])`.

Channels not appearing in any pair stay free (Cartesian).

θ layout:
- Length `(n_pairs + n_free) · n_t`, reshaped to
  `[(n_pairs + n_free), n_t]`.
- First `n_pairs` rows are phases (in the order `phase_pairs` is given).
- Remaining `n_free` rows are unconstrained Cartesian amplitudes (in
  ascending raw-channel index).

The Jacobian `∂w/∂θ` is sparse, not diagonal — phase ↔ (Cx, Cy) couples
two outputs to one input.  Use [`apply_jacobian_transpose!`](@ref) to
lift `∇_w F → ∇_θ F` without materialising the sparse matrix on the
hot path.
"""
struct PhaseOnlyParam <: AbstractControlParameterization
    amplitude   :: Float64
    phase_pairs :: Vector{Tuple{Int,Int}}
    function PhaseOnlyParam(amplitude::Real,
                            phase_pairs::AbstractVector)
        amplitude > 0 ||
            throw(ArgumentError(
                "PhaseOnlyParam amplitude must be > 0, got $amplitude"))
        isempty(phase_pairs) &&
            throw(ArgumentError(
                "PhaseOnlyParam phase_pairs must be non-empty"))
        used = Set{Int}()
        pairs_clean = Vector{Tuple{Int,Int}}(undef, length(phase_pairs))
        for (i, pair) in enumerate(phase_pairs)
            length(pair) == 2 ||
                throw(ArgumentError(
                    "PhaseOnlyParam phase_pairs entries must be 2-tuples"))
            cx = Int(pair[1]); cy = Int(pair[2])
            (cx ≥ 1 && cy ≥ 1) ||
                throw(ArgumentError(
                    "PhaseOnlyParam phase_pairs indices must be ≥ 1, got ($cx,$cy)"))
            cx == cy &&
                throw(ArgumentError(
                    "PhaseOnlyParam Cx/Cy indices must differ (got ($cx,$cy))"))
            (cx in used || cy in used) &&
                throw(ArgumentError(
                    "PhaseOnlyParam phase_pairs indices reused across pairs"))
            push!(used, cx); push!(used, cy)
            pairs_clean[i] = (cx, cy)
        end
        return new(Float64(amplitude), pairs_clean)
    end
end

"""
    BSplineParam(knot_spacing::Int, order::Int = 2)

B-spline basis of given `order` (0 = piecewise-constant; 2 = quadratic),
controlled by knots placed every `knot_spacing` time steps.  Number of
parameters per channel ≈ `n_t / knot_spacing + order`.  Reserved.
"""
struct BSplineParam <: AbstractControlParameterization
    knot_spacing :: Int
    order        :: Int
    function BSplineParam(knot_spacing::Integer, order::Integer = 2)
        knot_spacing > 0 ||
            throw(ArgumentError("BSplineParam knot_spacing must be > 0"))
        order in (0, 1, 2, 3) ||
            throw(ArgumentError("BSplineParam order must be 0, 1, 2, or 3"))
        return new(Int(knot_spacing), Int(order))
    end
end

"""
    HermiteParam(n_knots::Int)

Piecewise Hermite polynomials between `n_knots` optimisation knots, as in
SIMPSON `oc_grape_hermite`.  Smooth `C¹` waveform.  Reserved.
"""
struct HermiteParam <: AbstractControlParameterization
    n_knots :: Int
    function HermiteParam(n_knots::Integer)
        n_knots > 1 ||
            throw(ArgumentError("HermiteParam n_knots must be > 1"))
        return new(Int(n_knots))
    end
end

"""
    FourierParam(n_freq::Int)

Truncated Fourier expansion in `n_freq` modes per channel — equivalent to
GOAT's basis but defined uniformly with the rest of the parameterisation
hierarchy.  Reserved (plan: refactor existing `goat_optimize` onto this).
"""
struct FourierParam <: AbstractControlParameterization
    n_freq :: Int
    function FourierParam(n_freq::Integer)
        n_freq > 0 ||
            throw(ArgumentError("FourierParam n_freq must be > 0"))
        return new(Int(n_freq))
    end
end

"""
    ChebyshevParam(n_cheb::Int)

Chebyshev-`T_k` expansion truncated at `n_cheb` polynomials.  Better
end-point behaviour than Fourier for non-periodic waveforms.  Reserved.
"""
struct ChebyshevParam <: AbstractControlParameterization
    n_cheb :: Int
    function ChebyshevParam(n_cheb::Integer)
        n_cheb > 0 ||
            throw(ArgumentError("ChebyshevParam n_cheb must be > 0"))
        return new(Int(n_cheb))
    end
end

"""
    SlepianParam(n_slep::Int, bw::Float64)

Discrete prolate spheroidal sequences (Slepian functions) of bandwidth
`bw`, truncated to `n_slep` modes.  Optimal time-frequency localisation.
Reserved.
"""
struct SlepianParam <: AbstractControlParameterization
    n_slep :: Int
    bw     :: Float64
    function SlepianParam(n_slep::Integer, bw::Real)
        n_slep > 0 ||
            throw(ArgumentError("SlepianParam n_slep must be > 0"))
        bw > 0 ||
            throw(ArgumentError("SlepianParam bw must be > 0"))
        return new(Int(n_slep), Float64(bw))
    end
end

"""
    CRABRandomParam(basis::Symbol, n_modes::Int; rng_seed::Int = 0)

QuTiP-style chopped random basis: `n_modes` randomly drawn frequencies
of the chosen `basis` (`:fourier`, `:legendre`, `:chebyshev`).  Reserved.
"""
struct CRABRandomParam <: AbstractControlParameterization
    basis    :: Symbol
    n_modes  :: Int
    rng_seed :: Int
    function CRABRandomParam(basis::Symbol, n_modes::Integer;
                              rng_seed::Integer = 0)
        basis in (:fourier, :legendre, :chebyshev) ||
            throw(ArgumentError(
                "CRABRandomParam basis must be one of " *
                ":fourier, :legendre, :chebyshev (got $basis)"))
        n_modes > 0 ||
            throw(ArgumentError("CRABRandomParam n_modes must be > 0"))
        return new(basis, Int(n_modes), Int(rng_seed))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward / inverse maps
# ─────────────────────────────────────────────────────────────────────────────

"""
    to_waveform(θ, p::AbstractControlParameterization, n_ctrl, n_t) -> Matrix{Float64}

Forward map from optimisation variables `θ` to the `[n_ctrl × n_t]`
waveform consumed by the GRAPE / L-BFGS / Krotov kernels.
"""
function to_waveform end

"""
    from_waveform(w::AbstractMatrix{<:Real}, p::AbstractControlParameterization) -> Vector{Float64}

Inverse map (where defined) from a target waveform back to the closest
`θ` representable under `p`.  Used to seed warm-starts.
"""
function from_waveform end

"""
    waveform_jacobian(θ, p::AbstractControlParameterization, n_ctrl, n_t) -> AbstractMatrix{Float64}

Jacobian `J = ∂w / ∂θ` flattened to a `(n_ctrl·n_t) × length(θ)` matrix.
The optimiser lifts `∇_w F` to `∇_θ F = Jᵀ · ∇_w F`.

For the diagonal parameterisations (Tanh/TanhSq/Logistic) this Jacobian
is sparse — a column-vector of per-element derivatives — so we return it
as a `Diagonal` to avoid materialising `n_t² · n_ctrl²` zeros.
"""
function waveform_jacobian end

# ── PiecewiseConstant: identity ─────────────────────────────────────────────
function to_waveform(θ::AbstractVector{<:Real},
                     ::PiecewiseConstant,
                     n_ctrl::Integer, n_t::Integer)::Matrix{Float64}
    length(θ) == n_ctrl * n_t ||
        throw(DimensionMismatch(
            "PiecewiseConstant: length(θ)=$(length(θ)) ≠ n_ctrl·n_t=$(n_ctrl*n_t)"))
    return reshape(Vector{Float64}(θ), n_ctrl, n_t)
end

from_waveform(w::AbstractMatrix{<:Real}, ::PiecewiseConstant)::Vector{Float64} =
    vec(Matrix{Float64}(w))

waveform_jacobian(θ::AbstractVector{<:Real}, ::PiecewiseConstant,
                   n_ctrl::Integer, n_t::Integer) =
    Diagonal(ones(Float64, n_ctrl * n_t))

# ── TanhParam: w = u_max · tanh(θ) ──────────────────────────────────────────
function to_waveform(θ::AbstractVector{<:Real}, p::TanhParam,
                     n_ctrl::Integer, n_t::Integer)::Matrix{Float64}
    length(θ) == n_ctrl * n_t ||
        throw(DimensionMismatch(
            "TanhParam: length(θ)=$(length(θ)) ≠ n_ctrl·n_t=$(n_ctrl*n_t)"))
    return p.u_max .* tanh.(reshape(Vector{Float64}(θ), n_ctrl, n_t))
end

function from_waveform(w::AbstractMatrix{<:Real}, p::TanhParam)::Vector{Float64}
    # atanh blows up at ±1; clamp slightly inside the bound for numerical safety.
    s = 1 - 1e-12
    return [atanh(clamp(w[i] / p.u_max, -s, s)) for i in eachindex(w)]
end

function waveform_jacobian(θ::AbstractVector{<:Real}, p::TanhParam,
                            n_ctrl::Integer, n_t::Integer)
    length(θ) == n_ctrl * n_t ||
        throw(DimensionMismatch(
            "TanhParam: length(θ)=$(length(θ)) ≠ n_ctrl·n_t=$(n_ctrl*n_t)"))
    # d/dθ [u_max tanh(θ)] = u_max · sech²(θ) = u_max · (1 − tanh²(θ))
    diag = p.u_max .* (1 .- tanh.(Vector{Float64}(θ)).^2)
    return Diagonal(diag)
end

# ── TanhSqParam: w = u_max · tanh(θ)² ───────────────────────────────────────
function to_waveform(θ::AbstractVector{<:Real}, p::TanhSqParam,
                     n_ctrl::Integer, n_t::Integer)::Matrix{Float64}
    length(θ) == n_ctrl * n_t ||
        throw(DimensionMismatch(
            "TanhSqParam: length(θ)=$(length(θ)) ≠ n_ctrl·n_t=$(n_ctrl*n_t)"))
    return p.u_max .* tanh.(reshape(Vector{Float64}(θ), n_ctrl, n_t)).^2
end

function from_waveform(w::AbstractMatrix{<:Real}, p::TanhSqParam)::Vector{Float64}
    s = 1 - 1e-12
    out = Vector{Float64}(undef, length(w))
    @inbounds for i in eachindex(w)
        ratio = clamp(w[i] / p.u_max, 0.0, s)
        out[i] = atanh(sqrt(ratio))
    end
    return out
end

function waveform_jacobian(θ::AbstractVector{<:Real}, p::TanhSqParam,
                            n_ctrl::Integer, n_t::Integer)
    length(θ) == n_ctrl * n_t ||
        throw(DimensionMismatch(
            "TanhSqParam: length(θ)=$(length(θ)) ≠ n_ctrl·n_t=$(n_ctrl*n_t)"))
    # d/dθ [u_max tanh²(θ)] = 2 u_max · tanh(θ) · sech²(θ)
    th   = tanh.(Vector{Float64}(θ))
    diag = (2 * p.u_max) .* th .* (1 .- th.^2)
    return Diagonal(diag)
end

# ── LogisticParam: w = u_max · (2 σ(β θ) − 1) ────────────────────────────────
@inline _sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))

function to_waveform(θ::AbstractVector{<:Real}, p::LogisticParam,
                     n_ctrl::Integer, n_t::Integer)::Matrix{Float64}
    length(θ) == n_ctrl * n_t ||
        throw(DimensionMismatch(
            "LogisticParam: length(θ)=$(length(θ)) ≠ n_ctrl·n_t=$(n_ctrl*n_t)"))
    σ = _sigmoid.(p.beta .* reshape(Vector{Float64}(θ), n_ctrl, n_t))
    return p.u_max .* (2 .* σ .- 1)
end

function from_waveform(w::AbstractMatrix{<:Real}, p::LogisticParam)::Vector{Float64}
    s = 1 - 1e-12
    out = Vector{Float64}(undef, length(w))
    @inbounds for i in eachindex(w)
        # σ_inv(y) = log(y / (1 − y)); y = (w/u_max + 1)/2
        y = clamp((w[i] / p.u_max + 1) / 2, (1 - s)/2, (1 + s)/2)
        out[i] = log(y / (1 - y)) / p.beta
    end
    return out
end

function waveform_jacobian(θ::AbstractVector{<:Real}, p::LogisticParam,
                            n_ctrl::Integer, n_t::Integer)
    length(θ) == n_ctrl * n_t ||
        throw(DimensionMismatch(
            "LogisticParam: length(θ)=$(length(θ)) ≠ n_ctrl·n_t=$(n_ctrl*n_t)"))
    # d/dθ [u_max (2 σ(β θ) − 1)] = 2 u_max β · σ(β θ) (1 − σ(β θ))
    σ    = _sigmoid.(p.beta .* Vector{Float64}(θ))
    diag = (2 * p.u_max * p.beta) .* σ .* (1 .- σ)
    return Diagonal(diag)
end

# ── Reserved parameterisations: guarded errors ──────────────────────────────
const _RESERVED_PARAM_TYPES = Union{BSplineParam,HermiteParam,FourierParam,
                                     ChebyshevParam,SlepianParam,CRABRandomParam}

function to_waveform(::AbstractVector{<:Real}, p::_RESERVED_PARAM_TYPES,
                     ::Integer, ::Integer)
    error("$(typeof(p)) is a reserved parameterisation (Theme 2 / Phase 2). " *
          "Use PiecewiseConstant(), TanhParam, TanhSqParam, or LogisticParam.")
end

function from_waveform(::AbstractMatrix{<:Real}, p::_RESERVED_PARAM_TYPES)
    error("$(typeof(p)) is a reserved parameterisation (Theme 2 / Phase 2).")
end

function waveform_jacobian(::AbstractVector{<:Real}, p::_RESERVED_PARAM_TYPES,
                            ::Integer, ::Integer)
    error("$(typeof(p)) is a reserved parameterisation (Theme 2 / Phase 2).")
end

# ─────────────────────────────────────────────────────────────────────────────
# PhaseOnlyParam: forward / inverse / Jacobian
# ─────────────────────────────────────────────────────────────────────────────

# Free (Cartesian) raw-channel indices: every channel not used by any pair,
# in ascending order.  Computed once per call (cheap; n_ctrl is small).
function _phase_only_free_rows(p::PhaseOnlyParam, n_ctrl::Integer)::Vector{Int}
    used = Set{Int}()
    for (cx, cy) in p.phase_pairs
        push!(used, cx); push!(used, cy)
    end
    return [c for c in 1:n_ctrl if !(c in used)]
end

function _phase_only_validate_dims(p::PhaseOnlyParam, n_ctrl::Integer)
    n_p = length(p.phase_pairs)
    for (cx, cy) in p.phase_pairs
        (cx ≤ n_ctrl && cy ≤ n_ctrl) ||
            throw(DimensionMismatch(
                "PhaseOnlyParam pair ($cx,$cy) exceeds n_ctrl=$n_ctrl"))
    end
    n_free = n_ctrl - 2 * n_p
    n_free ≥ 0 ||
        throw(DimensionMismatch(
            "PhaseOnlyParam: 2·n_pairs=$(2*n_p) > n_ctrl=$n_ctrl"))
    return n_p, n_free
end

function to_waveform(θ::AbstractVector{<:Real}, p::PhaseOnlyParam,
                     n_ctrl::Integer, n_t::Integer)::Matrix{Float64}
    n_p, n_free = _phase_only_validate_dims(p, n_ctrl)
    length(θ) == (n_p + n_free) * n_t ||
        throw(DimensionMismatch(
            "PhaseOnlyParam: length(θ)=$(length(θ)) ≠ (n_p+n_free)·n_t=" *
            "$((n_p + n_free) * n_t)"))
    Θ = reshape(Vector{Float64}(θ), n_p + n_free, n_t)
    w = Matrix{Float64}(undef, n_ctrl, n_t)
    free_rows = _phase_only_free_rows(p, n_ctrl)
    A = p.amplitude
    @inbounds for k in 1:n_t
        for i in 1:n_p
            cx, cy = p.phase_pairs[i]
            φ = Θ[i, k]
            w[cx, k] = A * cos(φ)
            w[cy, k] = A * sin(φ)
        end
        for j in 1:n_free
            w[free_rows[j], k] = Θ[n_p + j, k]
        end
    end
    return w
end

function from_waveform(w::AbstractMatrix{<:Real}, p::PhaseOnlyParam)::Vector{Float64}
    n_ctrl, n_t = size(w)
    n_p, n_free = _phase_only_validate_dims(p, n_ctrl)
    Θ = Matrix{Float64}(undef, n_p + n_free, n_t)
    free_rows = _phase_only_free_rows(p, n_ctrl)
    @inbounds for k in 1:n_t
        for i in 1:n_p
            cx, cy = p.phase_pairs[i]
            Θ[i, k] = atan(w[cy, k], w[cx, k])
        end
        for j in 1:n_free
            Θ[n_p + j, k] = w[free_rows[j], k]
        end
    end
    return vec(Θ)
end

# Sparse Jacobian for callers that need an explicit matrix (analytical
# Hessian work).  The hot path uses apply_jacobian_transpose! instead.
function waveform_jacobian(θ::AbstractVector{<:Real}, p::PhaseOnlyParam,
                            n_ctrl::Integer, n_t::Integer)
    n_p, n_free = _phase_only_validate_dims(p, n_ctrl)
    length(θ) == (n_p + n_free) * n_t ||
        throw(DimensionMismatch(
            "PhaseOnlyParam: length(θ)=$(length(θ)) ≠ (n_p+n_free)·n_t=" *
            "$((n_p + n_free) * n_t)"))
    Θ = reshape(Vector{Float64}(θ), n_p + n_free, n_t)
    free_rows = _phase_only_free_rows(p, n_ctrl)
    A = p.amplitude

    n_θ_total = (n_p + n_free) * n_t
    n_w_total = n_ctrl * n_t
    nnz = (2 * n_p + n_free) * n_t
    I_idx = Vector{Int}(undef, nnz)
    J_idx = Vector{Int}(undef, nnz)
    V     = Vector{Float64}(undef, nnz)
    pos = 0
    @inbounds for k in 1:n_t
        for i in 1:n_p
            cx, cy = p.phase_pairs[i]
            φ = Θ[i, k]
            row_θ = (k - 1) * (n_p + n_free) + i
            pos += 1
            I_idx[pos] = (k - 1) * n_ctrl + cx
            J_idx[pos] = row_θ
            V[pos]     = -A * sin(φ)
            pos += 1
            I_idx[pos] = (k - 1) * n_ctrl + cy
            J_idx[pos] = row_θ
            V[pos]     = A * cos(φ)
        end
        for j in 1:n_free
            row_θ = (k - 1) * (n_p + n_free) + n_p + j
            pos += 1
            I_idx[pos] = (k - 1) * n_ctrl + free_rows[j]
            J_idx[pos] = row_θ
            V[pos]     = 1.0
        end
    end
    return sparse(I_idx, J_idx, V, n_w_total, n_θ_total)
end

# ─────────────────────────────────────────────────────────────────────────────
# apply_jacobian_transpose! — lift ∇_w F to ∇_θ F in-place
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_jacobian_transpose!(g_θ, g_w, θ, p::AbstractControlParameterization,
                              n_ctrl, n_t) -> g_θ

Lift the waveform-space gradient `g_w :: [n_ctrl × n_t]` to the
parameter-space gradient `g_θ` via the chain rule
`∇_θ F = Jᵀ · ∇_w F`.

Default implementation calls [`waveform_jacobian`](@ref) and applies it
explicitly — efficient for the diagonal element-wise parameterisations
(`PiecewiseConstant`, `TanhParam`, `TanhSqParam`, `LogisticParam`).

Parameterisations with non-diagonal Jacobians (e.g.
[`PhaseOnlyParam`](@ref)) override this method to apply the structure
in-place without materialising the sparse matrix.
"""
function apply_jacobian_transpose!(g_θ::AbstractVector{<:Real},
                                    g_w::AbstractMatrix{<:Real},
                                    θ::AbstractVector{<:Real},
                                    p::AbstractControlParameterization,
                                    n_ctrl::Integer, n_t::Integer)
    J = waveform_jacobian(θ, p, n_ctrl, n_t)
    mul!(g_θ, transpose(J), vec(g_w))
    return g_θ
end

# Identity short-circuit: PiecewiseConstant Jacobian is I, so the chain
# rule is just `g_θ ← vec(g_w)`.
function apply_jacobian_transpose!(g_θ::AbstractVector{<:Real},
                                    g_w::AbstractMatrix{<:Real},
                                    ::AbstractVector{<:Real},
                                    ::PiecewiseConstant,
                                    n_ctrl::Integer, n_t::Integer)
    length(g_θ) == n_ctrl * n_t ||
        throw(DimensionMismatch(
            "apply_jacobian_transpose! length(g_θ)=$(length(g_θ)) ≠ " *
            "n_ctrl·n_t=$(n_ctrl*n_t)"))
    @inbounds for i in eachindex(g_w)
        g_θ[i] = g_w[i]
    end
    return g_θ
end

# PhaseOnlyParam: structured chain rule, no sparse alloc.
function apply_jacobian_transpose!(g_θ::AbstractVector{<:Real},
                                    g_w::AbstractMatrix{<:Real},
                                    θ::AbstractVector{<:Real},
                                    p::PhaseOnlyParam,
                                    n_ctrl::Integer, n_t::Integer)
    n_p, n_free = _phase_only_validate_dims(p, n_ctrl)
    length(θ) == (n_p + n_free) * n_t ||
        throw(DimensionMismatch(
            "apply_jacobian_transpose! length(θ)=$(length(θ)) ≠ " *
            "(n_p+n_free)·n_t=$((n_p + n_free) * n_t)"))
    length(g_θ) == length(θ) ||
        throw(DimensionMismatch(
            "apply_jacobian_transpose! length(g_θ)=$(length(g_θ)) ≠ length(θ)"))
    size(g_w) == (n_ctrl, n_t) ||
        throw(DimensionMismatch(
            "apply_jacobian_transpose! size(g_w)=$(size(g_w)) ≠ (n_ctrl,n_t)"))
    Θ = reshape(Vector{Float64}(θ), n_p + n_free, n_t)
    G = reshape(g_θ, n_p + n_free, n_t)
    free_rows = _phase_only_free_rows(p, n_ctrl)
    A = p.amplitude
    @inbounds for k in 1:n_t
        for i in 1:n_p
            cx, cy = p.phase_pairs[i]
            φ = Θ[i, k]
            G[i, k] = -A * sin(φ) * g_w[cx, k] + A * cos(φ) * g_w[cy, k]
        end
        for j in 1:n_free
            G[n_p + j, k] = g_w[free_rows[j], k]
        end
    end
    return g_θ
end
