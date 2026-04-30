# ============================================================================
# Gradient/_LineSearch.jl
# Shared strong-Wolfe bracket-and-zoom line search.
#
# Consolidates three previously duplicated implementations:
#   - _gf_wolfe_ls  (GRAPEFamily.jl)   simple bracket, α_max=5.0,  zoom 1:30, ε=1e-13
#   - _bm_wolfe_ls  (BasisMethods.jl)  simple bracket, α_max=10.0, zoom 1:25, no ε-break
#   - _qn_wolfe_ls  (QuasiNewton.jl)   Nocedal bracket, α_max=10.0, zoom 1:40, ε=1e-14
#
# Call sites retain their legacy numerical behavior bit-for-bit by passing the
# appropriate kwargs.  Buffers θ_t / g_buf are supplied by the caller so the
# helper is zero-alloc per invocation.
# ============================================================================

using LinearAlgebra

"""
    wolfe_line_search!(θ_t, g_buf, f, grad!, θ, d, g0, f0;
                       c1=1e-4, c2=0.9, α_max=10.0,
                       max_iter=60, zoom_iter=40, zoom_eps=1e-14,
                       two_point_bracket=true) -> (α::Float64, f_α::Float64)

Strong-Wolfe line search.  Requires caller-supplied buffers `θ_t` (trial point,
length `length(θ)`) and `g_buf` (gradient scratch, length `length(g0)`).

`two_point_bracket=true` — Nocedal algorithm: on Armijo violation OR
`f_α >= f_prev` for i>1, zoom with `(α_prev, α)` and `f_lo = f_prev`; on
positive curvature, zoom with `(α, α_prev)` and `f_lo = f_α`.

`two_point_bracket=false` — Simple algorithm: on Armijo violation, zoom with
`(0, α)` and `f_lo = f0`; on positive curvature, break without zooming.

`zoom_eps` — terminate zoom when `abs(α_hi - α_lo) < zoom_eps`.  Pass `0.0`
to disable the early break.
"""
function wolfe_line_search!(
    θ_t      :: AbstractVector{Float64},
    g_buf    :: AbstractVector{Float64},
    f        :: Function,
    grad!    :: Function,
    θ        :: AbstractVector{Float64},
    d        :: AbstractVector{Float64},
    g0       :: AbstractVector{Float64},
    f0       :: Float64;
    c1                :: Float64 = 1e-4,
    c2                :: Float64 = 0.9,
    α_max             :: Float64 = 10.0,
    max_iter          :: Int     = 60,
    zoom_iter         :: Int     = 40,
    zoom_eps          :: Float64 = 1e-14,
    two_point_bracket :: Bool    = true,
)
    dg0 = dot(d, g0)
    dg0 >= 0.0 && return 1e-8, f0

    α      = 1.0
    α_prev = 0.0
    f_prev = f0

    @inline _set_trial!(a) = @. θ_t = θ + a * d

    for i in 1:max_iter
        _set_trial!(α); f_α = f(θ_t)

        if two_point_bracket
            if f_α > f0 + c1 * α * dg0 || (i > 1 && f_α >= f_prev)
                return _wolfe_zoom!(θ_t, g_buf, f, grad!, θ, d, f0,
                                     α_prev, α, f_prev, dg0,
                                     c1, c2, zoom_iter, zoom_eps)
            end
            grad!(g_buf, θ_t)
            dg_α = dot(d, g_buf)
            abs(dg_α) <= -c2 * dg0 && return α, f_α
            dg_α >= 0.0 &&
                return _wolfe_zoom!(θ_t, g_buf, f, grad!, θ, d, f0,
                                     α, α_prev, f_α, dg0,
                                     c1, c2, zoom_iter, zoom_eps)
            α_prev = α
            f_prev = f_α
            α      = min(2.0 * α, α_max)
        else
            # Simple bracket (GF / BM legacy behavior)
            if f_α > f0 + c1 * α * dg0
                return _wolfe_zoom!(θ_t, g_buf, f, grad!, θ, d, f0,
                                     0.0, α, f0, dg0,
                                     c1, c2, zoom_iter, zoom_eps)
            end
            grad!(g_buf, θ_t)
            dg_α = dot(d, g_buf)
            abs(dg_α) <= -c2 * dg0 && return α, f_α
            dg_α >= 0.0 && break
            α = min(2.0 * α, α_max)
        end
    end
    _set_trial!(α)
    return α, f(θ_t)
end

function _wolfe_zoom!(θ_t, g_buf, f, grad!, θ, d, f0,
                      α_lo::Float64, α_hi::Float64, f_lo::Float64,
                      dg0::Float64, c1::Float64, c2::Float64,
                      zoom_iter::Int, zoom_eps::Float64)
    @inline _set_trial!(a) = @. θ_t = θ + a * d
    for _ in 1:zoom_iter
        α  = (α_lo + α_hi) * 0.5
        _set_trial!(α); fa = f(θ_t)
        if fa > f0 + c1 * α * dg0 || fa >= f_lo
            α_hi = α
        else
            grad!(g_buf, θ_t)
            dga = dot(d, g_buf)
            abs(dga) <= -c2 * dg0 && return α, fa
            dga * (α_hi - α_lo) >= 0.0 && (α_hi = α_lo)
            α_lo = α
            f_lo = fa
        end
        abs(α_hi - α_lo) < zoom_eps && break
    end
    α_mid = (α_lo + α_hi) * 0.5
    _set_trial!(α_mid)
    return α_mid, f(θ_t)
end
