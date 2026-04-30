# ============================================================================
# Application/MR/_OrientationAggregation.jl
# Shared powder / orientation averaging helpers for MR domain optimizers.
#
# Used by:
#   - EPROptControl.jl  (g-tensor orientation rotation, band-selective or raw)
#   - DNPOptControl.jl  (electron-nuclear orientation rotation)
#
# Each caller supplies domain-specific `rotate_fn(α, β, γ)` and
# `gradient_fn(sys_ori)` / `fidelity_fn(sys_ori)` closures; the helper
# handles the reduction (threaded accumulate for gradients; serial sum
# for fidelities).
# ============================================================================

"""
    _aggregate_orientation_gradient(ctrl_shape, orientations, rotate_fn, gradient_fn)
    -> Matrix{Float64}

Thread-parallel weighted sum of per-orientation gradients.

Each entry of `orientations` is `(α, β, γ, w)`.  For each orientation,
`rotate_fn(α, β, γ)` returns an oriented system, and `gradient_fn(sys_ori)`
returns the per-orientation gradient matrix (of shape `ctrl_shape`).
The reduction `G += w * G_ori` is performed under a `ReentrantLock`.
"""
function _aggregate_orientation_gradient(ctrl_shape::Tuple{Int,Int},
                                          orientations,
                                          rotate_fn::Function,
                                          gradient_fn::Function)::Matrix{Float64}
    G  = zeros(ctrl_shape)
    lk = ReentrantLock()
    # Lesson 2: BLAS-thread guard via `@threadsif` for safe nested parallelism
    # when the per-orientation gradient calls into a multi-threaded LAPACK.
    @threadsif true for ori in orientations
        α, β, γ, w = ori
        sys_ori    = rotate_fn(α, β, γ)
        G_ori      = gradient_fn(sys_ori)
        lock(lk) do
            G .+= w .* G_ori
        end
    end
    return G
end

"""
    _aggregate_orientation_fidelity(orientations, rotate_fn, fidelity_fn) -> Float64

Serial weighted sum of per-orientation fidelities.  Matches legacy EPR/DNP
behavior which did not thread the fidelity loop (the per-orientation fidelity
call is typically cheap and called inside a gradient-function closure that
is itself outer-threaded).
"""
function _aggregate_orientation_fidelity(orientations,
                                          rotate_fn::Function,
                                          fidelity_fn::Function)::Float64
    F = 0.0
    for (α, β, γ, w) in orientations
        sys_ori = rotate_fn(α, β, γ)
        F      += w * fidelity_fn(sys_ori)
    end
    return F
end

"""
    _wrap_orient_gradient(base_grad, orientations, rotate_fn) -> Function

If `orientations === nothing`, returns `base_grad` unchanged. Otherwise returns a
`(s,c,t) ->` closure that evaluates `base_grad(sys_ori, c, t)` on every rotated
system produced by `rotate_fn(α,β,γ)` and sums the results with the grid weights.
"""
function _wrap_orient_gradient(base_grad    :: Function,
                                orientations :: Union{Nothing,Vector{NTuple{4,Float64}}},
                                rotate_fn    :: Function)::Function
    orientations === nothing && return base_grad
    return (s, c, t) -> _aggregate_orientation_gradient(
                           size(c.controls), orientations, rotate_fn,
                           sys_ori -> base_grad(sys_ori, c, t))
end

"""
    _wrap_orient_fidelity(base_fid, orientations, rotate_fn) -> Function

Fidelity counterpart to [`_wrap_orient_gradient`](@ref); identity when
`orientations === nothing`, orientation-averaged otherwise.
"""
function _wrap_orient_fidelity(base_fid     :: Function,
                                orientations :: Union{Nothing,Vector{NTuple{4,Float64}}},
                                rotate_fn    :: Function)::Function
    orientations === nothing && return base_fid
    return (s, c, t) -> _aggregate_orientation_fidelity(
                           orientations, rotate_fn,
                           sys_ori -> base_fid(sys_ori, c, t))
end
