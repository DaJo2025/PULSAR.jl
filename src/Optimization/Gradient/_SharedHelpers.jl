# ============================================================================
# Gradient/_SharedHelpers.jl
# Small shared helpers used across the QOC and Generic optimizer hierarchies.
# Every helper is numerically identical to the inlined code it replaces.
# ============================================================================

using LinearAlgebra

# ---------------------------------------------------------------------------
# Nonlinear CG β coefficient
#
# Callers supply the current gradient g_cur, previous gradient g_prev, and
# current search direction d (the latter only needed for :HS / :DY).  The
# :PR branch implements PR+ (the max with 0 restart).
# ---------------------------------------------------------------------------

@inline function _cg_beta(g_cur::AbstractVector{<:Real},
                          g_prev::AbstractVector{<:Real},
                          d::AbstractVector{<:Real},
                          method::Symbol)
    gg_old = dot(g_prev, g_prev)
    gg_old < 1e-30 && return 0.0
    if method === :FR
        return dot(g_cur, g_cur) / gg_old
    elseif method === :PR
        return max(0.0, dot(g_cur, g_cur .- g_prev) / gg_old)
    elseif method === :HS
        dy    = g_cur .- g_prev
        denom = dot(dy, d)
        return abs(denom) < 1e-30 ? 0.0 : dot(g_cur, dy) / denom
    elseif method === :DY
        dy    = g_cur .- g_prev
        denom = dot(dy, d)
        return abs(denom) < 1e-30 ? 0.0 : dot(g_cur, g_cur) / denom
    else
        error("_cg_beta: unknown method $method; choose :FR, :PR, :HS, :DY")
    end
end

# ---------------------------------------------------------------------------
# Trust-region radius update (Nocedal & Wright Alg. 4.1)
#
#   ρ < 0.25            ⇒ Δ ← max(Δ/4, 1e-10)
#   ρ > 0.75 & near Δ   ⇒ Δ ← min(2Δ, Δ_max)
#   otherwise           ⇒ Δ unchanged
# ---------------------------------------------------------------------------

@inline function _update_tr_radius(Δ::Float64, ρ::Float64,
                                    d_norm::Float64, Δ_max::Float64)
    if ρ < 0.25
        return max(Δ * 0.25, 1e-10)
    elseif ρ > 0.75 && d_norm >= 0.9 * Δ
        return min(2.0 * Δ, Δ_max)
    end
    return Δ
end
