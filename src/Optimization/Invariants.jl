# ============================================================================
# Optimization/Invariants.jl — Runtime self-checks for optimizer correctness
#
# Every check returns `(ok::Bool, msg::String)`.  When called from an optimizer
# with its config's `check_invariants = true`, a failure raises
# `InvariantViolationError`.  Checks are opt-in so production runs pay no cost.
# ============================================================================

using LinearAlgebra

# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

"""
    InvariantViolationError(check, message, context)

Raised when a runtime self-check fails in an optimizer with
`check_invariants=true`.  Fields:

* `check::Symbol`   — identifier, e.g. `:bfgs_curvature`
* `message::String` — human-readable failure description
* `context::NamedTuple` — optimizer-supplied context, e.g. `(iter=42, k=7)`
"""
struct InvariantViolationError <: Exception
    check   :: Symbol
    message :: String
    context :: NamedTuple
end

function Base.showerror(io::IO, e::InvariantViolationError)
    print(io, "InvariantViolationError(", e.check, "): ", e.message)
    isempty(e.context) || print(io, "  context=", e.context)
end

# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------

@inline function _assert_invariant(ok::Bool, msg::String, check::Symbol,
                                    context::NamedTuple = NamedTuple())
    ok && return nothing
    throw(InvariantViolationError(check, msg, context))
end

# ---------------------------------------------------------------------------
# Line-search / gradient-descent checks
# ---------------------------------------------------------------------------

"""
    check_armijo(f_new, f_old, c, α, slope; atol=1e-12) → (ok, msg)

Armijo sufficient-decrease condition for minimisation:
`f_new ≤ f_old + c · α · slope`, where `slope = ∇f·d < 0` for a descent
direction.  A small `atol` absorbs floating-point noise near equality.
"""
function check_armijo(f_new::Real, f_old::Real, c::Real, α::Real, slope::Real;
                      atol::Real = 1e-12)
    rhs = f_old + c * α * slope + atol
    ok  = f_new ≤ rhs
    msg = ok ? "armijo ok" :
          "armijo violated: f_new=$(f_new) > f_old + c·α·slope = $(rhs) " *
          "(f_old=$(f_old), c=$(c), α=$(α), slope=$(slope))"
    return ok, msg
end

"""
    check_wolfe_curvature(slope_new, slope_old, c2) → (ok, msg)

Strong-Wolfe curvature condition: `|slope_new| ≤ c2 · |slope_old|`.
"""
function check_wolfe_curvature(slope_new::Real, slope_old::Real, c2::Real)
    ok  = abs(slope_new) ≤ c2 * abs(slope_old) + eps()
    msg = ok ? "wolfe curvature ok" :
          "wolfe curvature violated: |slope_new|=$(abs(slope_new)) > " *
          "c2·|slope_old|=$(c2 * abs(slope_old))"
    return ok, msg
end

# ---------------------------------------------------------------------------
# Quasi-Newton (BFGS / L-BFGS) checks
# ---------------------------------------------------------------------------

"""
    check_bfgs_curvature(s, y; tol=1e-14) → (ok, msg)

BFGS curvature condition `y'·s > tol·‖s‖²` required for the BFGS Hessian
update to remain positive definite.
"""
function check_bfgs_curvature(s::AbstractVector, y::AbstractVector; tol::Real = 1e-14)
    ys    = dot(y, s)
    sn2   = dot(s, s)
    bound = tol * sn2
    ok    = ys > bound
    msg   = ok ? "bfgs curvature ok (y'·s=$(ys))" :
            "bfgs curvature failed: y'·s=$(ys) ≤ tol·‖s‖²=$(bound)"
    return ok, msg
end

"""
    check_lbfgs_pair_positive(ρ; k=0) → (ok, msg)

L-BFGS two-loop recursion requires `ρ_k = 1/(y_k'·s_k) > 0` for every
stored pair.
"""
function check_lbfgs_pair_positive(ρ::Real; k::Integer = 0)
    ok  = isfinite(ρ) && ρ > 0
    msg = ok ? "lbfgs ρ_$(k) ok" :
          "lbfgs pair $(k) has non-positive ρ=$(ρ)"
    return ok, msg
end

# ---------------------------------------------------------------------------
# Monotonicity / trust-region / penalty checks
# ---------------------------------------------------------------------------

"""
    check_monotone_ascent(history; tol=1e-12) → (ok, msg)

For Krotov-class methods fidelity is expected to grow monotonically.
Accepts a vector of fidelities and tests `history[i+1] ≥ history[i] - tol`.
"""
function check_monotone_ascent(history::AbstractVector{<:Real}; tol::Real = 1e-12)
    length(history) ≤ 1 && return true, "monotone ascent trivially ok"
    @inbounds for i in 2:length(history)
        if history[i] < history[i-1] - tol
            return false, "monotone ascent violated at step $(i): " *
                          "history[$(i)]=$(history[i]) < " *
                          "history[$(i-1)]=$(history[i-1]) - tol"
        end
    end
    return true, "monotone ascent ok"
end

"""
    check_trust_region_ratio(ρ) → (ok, msg)

Trust-region actual-vs-predicted-reduction ratio must be finite.
NaN indicates a degenerate model or zero predicted reduction.
"""
function check_trust_region_ratio(ρ::Real)
    ok  = !isnan(ρ)
    msg = ok ? "trust-region ratio ok (ρ=$(ρ))" :
          "trust-region ratio is NaN — degenerate model or zero predicted reduction"
    return ok, msg
end

"""
    check_penalty_weight_growth(λ_hist) → (ok, msg)

Penalty-method outer loops must have monotonically non-decreasing penalty
weights.
"""
function check_penalty_weight_growth(λ_hist::AbstractVector{<:Real})
    length(λ_hist) ≤ 1 && return true, "penalty weight growth trivially ok"
    @inbounds for i in 2:length(λ_hist)
        if λ_hist[i] < λ_hist[i-1]
            return false, "penalty weight decreased at outer iter $(i): " *
                          "$(λ_hist[i-1]) → $(λ_hist[i])"
        end
    end
    return true, "penalty weight growth ok"
end

# ---------------------------------------------------------------------------
# Direct-search checks
# ---------------------------------------------------------------------------

"""
    check_simplex_shape(simplex; atol=1e-10) → (ok, msg)

Nelder-Mead simplex must contain `n+1` distinct vertices.
`simplex` is a vector of vertices (each an `AbstractVector`).
"""
function check_simplex_shape(simplex::AbstractVector; atol::Real = 1e-10)
    n = length(simplex)
    n ≥ 2 || return false, "simplex has fewer than 2 vertices"
    for i in 1:n-1, j in i+1:n
        if norm(simplex[i] .- simplex[j]) < atol
            return false, "simplex degenerate: vertex $(i) ≈ vertex $(j)"
        end
    end
    return true, "simplex shape ok (n=$(n) distinct vertices)"
end

# ---------------------------------------------------------------------------
# CMA-ES checks
# ---------------------------------------------------------------------------

"""
    check_cma_covariance(C; atol=1e-12) → (ok, msg)

CMA-ES covariance must stay symmetric positive-semidefinite.  Tests
symmetry `‖C - C'‖ < atol · ‖C‖` and minimum eigenvalue `≥ -atol`.
"""
function check_cma_covariance(C::AbstractMatrix; atol::Real = 1e-12)
    nC = norm(C)
    if norm(C .- C') > atol * max(nC, one(nC))
        return false, "CMA-ES covariance not symmetric: ‖C - C'‖ = $(norm(C .- C'))"
    end
    λmin = minimum(real.(eigvals(Symmetric(Matrix(C)))))
    if λmin < -atol
        return false, "CMA-ES covariance not PSD: λ_min = $(λmin)"
    end
    return true, "CMA covariance ok (λ_min=$(λmin))"
end

# ---------------------------------------------------------------------------
# Robust / CVaR checks
# ---------------------------------------------------------------------------

"""
    check_cvar_ordering(sorted_f) → (ok, msg)

The tail averaged by CVaR must come from an ascendingly-sorted sample set.
"""
function check_cvar_ordering(sorted_f::AbstractVector{<:Real})
    length(sorted_f) ≤ 1 && return true, "cvar ordering trivially ok"
    @inbounds for i in 2:length(sorted_f)
        if sorted_f[i] < sorted_f[i-1]
            return false, "cvar ordering violated at index $(i): " *
                          "$(sorted_f[i-1]) > $(sorted_f[i])"
        end
    end
    return true, "cvar ordering ok"
end

# ---------------------------------------------------------------------------
# Physics-level checks (optional, O(dim²))
# ---------------------------------------------------------------------------

"""
    check_unitary_invariant(U; atol=1e-10) → (ok, msg)

Verify `‖U† U - I‖ < atol`.  Cost is O(dim²); enable only on the debug path.
(The `_invariant` suffix distinguishes this from
`check_unitary(U; tol)::Bool` in `Utilities/ParameterValidation.jl`.)
"""
function check_unitary_invariant(U::AbstractMatrix; atol::Real = 1e-10)
    n = size(U, 1)
    size(U, 2) == n || return false, "check_unitary_invariant: U is not square ($(size(U)))"
    err = opnorm(U' * U - I)
    ok  = err < atol
    msg = ok ? "unitary ok (‖U†U - I‖=$(err))" :
          "unitary check failed: ‖U†U - I‖=$(err) ≥ atol=$(atol)"
    return ok, msg
end

"""
    check_pure_state_norm(ψ; atol=1e-10) → (ok, msg)

Verify `|‖ψ‖² - 1| < atol` for a pure state.
"""
function check_pure_state_norm(ψ::AbstractVector; atol::Real = 1e-10)
    n2  = sum(abs2, ψ)
    err = abs(n2 - 1)
    ok  = err < atol
    msg = ok ? "state norm ok (‖ψ‖²=$(n2))" :
          "state norm check failed: |‖ψ‖² - 1|=$(err) ≥ atol=$(atol)"
    return ok, msg
end
