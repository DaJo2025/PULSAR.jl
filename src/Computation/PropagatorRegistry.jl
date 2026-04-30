# ============================================================
# PULSAR.jl — Propagator dispatch hierarchy (Theme 1)
# ============================================================
#
# Pluggable single-step propagator backends.  The legacy 2-arg
#
#   compute_propagator(H, dt)
#
# remains the default entry point and continues to use a Hermitian
# eigendecomposition.  This file adds a 3-arg overload
#
#   compute_propagator(H, dt, ::AbstractPropagator)
#
# so callers (MRControl / QCControl, GRAPE kernels, etc.) can choose a
# different time-evolution backend without touching the calling code.
#
# Built-in backends:
#   • EigenPropagator        — Hermitian eigendecomposition (default).
#   • PadePropagator         — scaling-and-squaring Padé (LAPACK `exp`).
#   • ChebyshevPropagator    — Tal-Ezer / Kosloff polynomial expansion.
#   • NewtonPropagator       — Newton polynomial w/ Leja interpolation.
#   • MagnusPropagator       — Magnus expansion for time-dependent H.
#
# Only EigenPropagator and PadePropagator are wired up in Phase 1.
# The other types are reserved (parsed + exported) so that they can
# be filled in by Phase 2/3 work without touching the public API.
# ============================================================

"""
    AbstractPropagator

Supertype for time-evolution backends consumed by
[`compute_propagator`](@ref).  All backends compute the same
mathematical object — the unitary `U = exp(-i H dt)` — but trade off
accuracy, memory, and structure exploitation differently.
"""
abstract type AbstractPropagator end

"""
    EigenPropagator()

Hermitian eigendecomposition `H = V diag(λ) V†` with
`U = V diag(exp(-i λ dt)) V†`.  Exact for Hermitian `H`, cost `O(d³)`,
machine-precision unitarity.  This is the legacy PULSAR default.
"""
struct EigenPropagator <: AbstractPropagator end

"""
    PadePropagator(; balance::Bool = true)

Scaling-and-squaring Padé approximation via `LinearAlgebra.exp`.  Does
not require `H` to be Hermitian — useful for non-Hermitian effective
Hamiltonians (e.g. PT-symmetric or amplified systems) that the
eigendecomposition path rejects.

`balance = true` prebalances the matrix prior to the Padé approximation
(the default for `LinearAlgebra.exp` on Julia 1.10+).
"""
struct PadePropagator <: AbstractPropagator
    balance :: Bool
    PadePropagator(; balance::Bool = true) = new(balance)
end

"""
    ChebyshevPropagator(order::Int, spectrum_bounds::Tuple{Float64,Float64})

Tal-Ezer / Kosloff Chebyshev expansion of `exp(-i H dt)`:

    exp(-i α R̂) ≈ Σ_n a_n(α) T_n(R̂)        with  R̂ = (H − E_avg I) / ΔE/2

where the coefficients `a_n` are Bessel functions.  Requires *a priori*
spectrum bounds `(E_min, E_max)` for accurate normalisation; recovering
unitarity to `~1e-12` typically needs `order ≈ 1.5 · α + 20`.

Reserved for Phase 2 — calling [`compute_propagator`](@ref) with this
backend currently throws `ErrorException("not yet implemented")`.
"""
struct ChebyshevPropagator <: AbstractPropagator
    order           :: Int
    spectrum_bounds :: Tuple{Float64,Float64}
    function ChebyshevPropagator(order::Integer,
                                  spectrum_bounds::Tuple{<:Real,<:Real})
        order > 0 ||
            throw(ArgumentError("ChebyshevPropagator order must be > 0"))
        first(spectrum_bounds) < last(spectrum_bounds) ||
            throw(ArgumentError(
                "spectrum_bounds must be (E_min, E_max) with E_min < E_max"))
        return new(Int(order),
                   (Float64(spectrum_bounds[1]), Float64(spectrum_bounds[2])))
    end
end

"""
    NewtonPropagator(n_interp::Int)

Newton-polynomial interpolation of `exp(-i z dt)` at Leja points across
the spectrum of `H`.  Suitable for non-Hermitian `H` where Chebyshev
fails.  Reserved for Phase 2.
"""
struct NewtonPropagator <: AbstractPropagator
    n_interp :: Int
    function NewtonPropagator(n_interp::Integer)
        n_interp > 0 ||
            throw(ArgumentError("NewtonPropagator n_interp must be > 0"))
        return new(Int(n_interp))
    end
end

"""
    MagnusPropagator(order::Int)

Magnus expansion `Ω(t) = Σ_k Ω_k(t)` truncated at order `k = order`,
producing `U = exp(Ω(t))`.  Targets time-dependent `H(t)` over a slice
of duration `dt`.  Reserved for Phase 2.
"""
struct MagnusPropagator <: AbstractPropagator
    order :: Int
    function MagnusPropagator(order::Integer)
        order in (2, 4, 6) ||
            throw(ArgumentError(
                "MagnusPropagator order must be 2, 4, or 6 (got $order)"))
        return new(Int(order))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_propagator(H, dt, propagator::AbstractPropagator) -> Matrix{ComplexF64}

Backend-aware overload of the single-step propagator.  Equivalent to
the legacy 2-arg call when `propagator = EigenPropagator()` (the
default).  Other backends preserve the same signature so callers can
swap implementations without other code changes.
"""
function compute_propagator(H::AbstractMatrix{ComplexF64}, dt::Real,
                             ::EigenPropagator)::Matrix{ComplexF64}
    # Delegate to the legacy 2-arg implementation (already in this file's
    # peer Propagators.jl).  Defining this overload here means future
    # callers can write
    #     compute_propagator(H, dt, ctrl.propagator)
    # without branching on the backend type.
    return compute_propagator(H, dt)
end

function compute_propagator(H::AbstractMatrix{ComplexF64}, dt::Real,
                             p::PadePropagator)::Matrix{ComplexF64}
    dt = Float64(dt)
    m, n = size(H)
    m == n ||
        throw(ArgumentError("H must be square, got $m × $n"))
    if dt == 0.0
        return Matrix{ComplexF64}(I, m, m)
    end
    # `exp` on a dense complex matrix already uses Higham's
    # scaling-and-squaring with order-13 Padé; balancing is handled
    # internally on Julia 1.10+.  We pass through `p.balance` for parity
    # with non-balanced reference behaviour requested by the user.
    return p.balance ? exp(Matrix(-im * dt .* H)) :
                       exp!(Matrix(-im * dt .* H))
end

function compute_propagator(::AbstractMatrix{ComplexF64}, ::Real,
                             ::ChebyshevPropagator)::Matrix{ComplexF64}
    error("ChebyshevPropagator is reserved (Theme 1 / Phase 2). " *
          "Use EigenPropagator() or PadePropagator() until then.")
end

function compute_propagator(::AbstractMatrix{ComplexF64}, ::Real,
                             ::NewtonPropagator)::Matrix{ComplexF64}
    error("NewtonPropagator is reserved (Theme 1 / Phase 2). " *
          "Use EigenPropagator() or PadePropagator() until then.")
end

function compute_propagator(::AbstractMatrix{ComplexF64}, ::Real,
                             ::MagnusPropagator)::Matrix{ComplexF64}
    error("MagnusPropagator is reserved (Theme 1 / Phase 2/3). " *
          "Use EigenPropagator() or PadePropagator() until then.")
end

# Backwards-compatible fallback: any future AbstractPropagator subtype
# that forgets to provide its own method falls through here with a clear
# message rather than the generic `MethodError`.
function compute_propagator(::AbstractMatrix{ComplexF64}, ::Real,
                             p::AbstractPropagator)::Matrix{ComplexF64}
    error("compute_propagator: no method registered for backend " *
          "$(typeof(p)).  Define `compute_propagator(H, dt, ::$(typeof(p)))` " *
          "to add support.")
end

# Internal: support for the `balance = false` branch on Julia versions
# where `exp!` is not exported.
@static if !isdefined(LinearAlgebra, :exp!)
    exp!(A::AbstractMatrix) = exp(A)
end
