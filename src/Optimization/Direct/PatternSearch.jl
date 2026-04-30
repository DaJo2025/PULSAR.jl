# ============================================================================
# Direct/PatternSearch.jl — Hooke-Jeeves, Compass, Powell direction-set
# ============================================================================

# ── Internal helpers ──────────────────────────────────────────────────────────

_ps_uvec(n::Int, i::Int) = (v = zeros(n); v[i] = 1.0; v)

# Golden-section line minimisation along ray x + α*d for α ≥ 0.
# Returns (x_new, f_new, n_evals).
function _ps_line_min(f, x::Vector{Float64}, d::Vector{Float64},
                      step::Float64, lb::Vector{Float64}, ub::Vector{Float64})
    clip(v) = clamp.(v, lb, ub)
    f0   = f(x); n_ev = 0
    φ    = 0.6180339887498949   # (√5 - 1)/2

    # Find upper bracket: double step until no improvement
    α_hi = step
    f_hi = f(clip(x .+ α_hi .* d)); n_ev += 1
    if f_hi >= f0
        # Halve to find any improvement
        for _ in 1:6
            α_hi /= 2
            f_hi  = f(clip(x .+ α_hi .* d)); n_ev += 1
            f_hi < f0 && break
        end
        f_hi >= f0 && return x, f0, n_ev
    end
    for _ in 1:20
        α_try = α_hi * 2
        f_try = f(clip(x .+ α_try .* d)); n_ev += 1
        f_try >= f_hi && break
        α_hi = α_try; f_hi = f_try
    end

    # Golden-section refinement in [0, α_hi*2]
    a, b   = 0.0, α_hi * 2.0
    c_α    = b - φ*(b - a); fc = f(clip(x .+ c_α .* d)); n_ev += 1
    d_α    = a + φ*(b - a); fd = f(clip(x .+ d_α .* d)); n_ev += 1
    for _ in 1:40
        b - a < 1e-9 * max(1.0, abs(a)) && break
        if fc < fd
            b = d_α; d_α = c_α; fd = fc
            c_α = b - φ*(b - a); fc = f(clip(x .+ c_α .* d)); n_ev += 1
        else
            a = c_α; c_α = d_α; fc = fd
            d_α = a + φ*(b - a); fd = f(clip(x .+ d_α .* d)); n_ev += 1
        end
    end
    α_best = (a + b) / 2
    x_new  = clip(x .+ α_best .* d)
    f_new  = f(x_new); n_ev += 1
    return f_new < f0 ? (x_new, f_new, n_ev) : (x, f0, n_ev)
end

# Hooke-Jeeves exploratory move: scan all axes, accept first improvement.
function _ps_hj_explore(f, x::Vector{Float64}, fx::Float64,
                         δ::Float64, lb::Vector{Float64}, ub::Vector{Float64})
    n    = length(x)
    clip(v) = clamp.(v, lb, ub)
    n_ev = 0
    for i in 1:n
        x_p  = clip(x .+ δ .* _ps_uvec(n, i))
        f_p  = f(x_p); n_ev += 1
        if f_p < fx; x = x_p; fx = f_p; continue; end
        x_m  = clip(x .- δ .* _ps_uvec(n, i))
        f_m  = f(x_m); n_ev += 1
        if f_m < fx; x = x_m; fx = f_m; end
    end
    return x, fx, n_ev
end

# ── Hooke-Jeeves ──────────────────────────────────────────────────────────────

"""
    hooke_jeeves_optimize(f, θ0; max_evals, max_iters, lower, upper, step,
                          step_min, step_shrink, tol) → (θ_best, f_best, stats)

Hooke-Jeeves pattern search: exploratory moves along coordinate axes, then
pattern extrapolation; step size shrinks on failure.
Minimises f; stats = (evals, iters, converged).
"""
function hooke_jeeves_optimize(
    f           :: Function,
    θ0          :: AbstractVector{<:Real};
    max_evals   :: Int     = 10_000,
    max_iters   :: Int     = 2_000,
    lower       :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper       :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step        :: Float64 = 0.1,
    step_min    :: Float64 = 1e-7,
    step_shrink :: Float64 = 0.5,
    tol         :: Float64 = 1e-6,
    callback = nothing,
)
    n    = length(θ0)
    lb   = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub   = upper === nothing ? fill( Inf, n) : Float64.(upper)
    clip(v) = clamp.(v, lb, ub)

    x_base = clip(Float64.(θ0))
    f_base = f(x_base); n_evals = 1
    x_prev = copy(x_base)
    δ      = step
    history = Float64[f_base]

    converged = false
    iter      = 0
    while iter < max_iters && n_evals < max_evals && δ > step_min
        iter += 1

        # Exploratory move from x_base
        x_exp, f_exp, nev = _ps_hj_explore(f, x_base, f_base, δ, lb, ub)
        n_evals += nev

        if f_exp < f_base
            # Pattern move: extrapolate beyond x_exp
            x_pat          = clip(x_exp .+ (x_exp .- x_prev))
            f_pat          = f(x_pat); n_evals += 1
            x_exp2, f_exp2, nev2 = _ps_hj_explore(f, x_pat, f_pat, δ, lb, ub)
            n_evals += nev2

            if f_exp2 < f_exp
                x_prev = copy(x_base)
                x_base = x_exp2; f_base = f_exp2
            else
                x_prev = copy(x_base)
                x_base = x_exp; f_base = f_exp
            end
        else
            δ *= step_shrink
        end

        push!(history, f_base)
        f_base < tol && (converged = true; break)
        isnothing(callback) || callback(iter, f_base; grad=nothing, evals=n_evals)
    end

    stats = (evals=n_evals, iters=iter, converged=converged)
    return x_base, f_base, stats
end

# ── Compass search ────────────────────────────────────────────────────────────

"""
    compass_search_optimize(f, θ0; max_evals, max_iters, lower, upper, step,
                             step_min, tol) → (θ_best, f_best, stats)

Coordinate (axial) search: poll all 2n neighbours; accept best improvement;
halve step on failed poll. Minimises f; stats = (evals, iters, converged).
"""
function compass_search_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 10_000,
    max_iters :: Int     = 5_000,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step      :: Float64 = 0.1,
    step_min  :: Float64 = 1e-8,
    tol       :: Float64 = 1e-7,
    callback = nothing,
)
    n    = length(θ0)
    lb   = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub   = upper === nothing ? fill( Inf, n) : Float64.(upper)
    clip(v) = clamp.(v, lb, ub)

    x       = clip(Float64.(θ0))
    fx      = f(x); n_evals = 1
    δ       = step
    history = Float64[fx]

    converged = false
    iter      = 0
    while iter < max_iters && n_evals < max_evals && δ > step_min
        iter += 1
        improved = false

        # Poll all 2n directions; keep best single improvement
        best_x  = x; best_f = fx
        for i in 1:n, sgn in (1.0, -1.0)
            x_try = clip(x .+ sgn * δ .* _ps_uvec(n, i))
            f_try = f(x_try); n_evals += 1
            if f_try < best_f
                best_f = f_try; best_x = x_try; improved = true
            end
            n_evals >= max_evals && break
        end

        if improved
            x = best_x; fx = best_f
        else
            δ *= 0.5
        end

        push!(history, fx)
        fx < tol && (converged = true; break)
        isnothing(callback) || callback(iter, fx; grad=nothing, evals=n_evals)
    end

    stats = (evals=n_evals, iters=iter, converged=converged)
    return x, fx, stats
end

# ── Powell direction-set ──────────────────────────────────────────────────────

"""
    powell_dirset_optimize(f, θ0; max_evals, max_iters, lower, upper, step,
                            tol) → (θ_best, f_best, stats)

Powell direction-set method with golden-section line searches and conjugate
direction update (direction of maximum decrease replaced each cycle).
Minimises f; stats = (evals, iters, converged).
"""
function powell_dirset_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 10_000,
    max_iters :: Int     = 500,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step      :: Float64 = 0.1,
    tol       :: Float64 = 1e-7,
    callback = nothing,
)
    n    = length(θ0)
    lb   = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub   = upper === nothing ? fill( Inf, n) : Float64.(upper)

    # Initial direction set: coordinate axes
    D    = Matrix{Float64}(I, n, n)   # columns are directions
    x    = clamp.(Float64.(θ0), lb, ub)
    fx   = f(x); n_evals = 1
    history = Float64[fx]

    converged = false
    iter      = 0
    while iter < max_iters && n_evals < max_evals
        iter += 1
        x_start = copy(x)
        f_start = fx

        max_decrease = -Inf
        k_max        = 1

        for i in 1:n
            f_before      = fx
            d             = D[:, i]
            x, fx, nev    = _ps_line_min(f, x, d, step, lb, ub)
            n_evals      += nev
            decrease_i    = f_before - fx
            if decrease_i > max_decrease
                max_decrease = decrease_i
                k_max        = i
            end
            n_evals >= max_evals && break
        end

        # New direction: overall displacement
        new_dir = x .- x_start
        nd_norm = norm(new_dir)

        # Convergence: displacement too small
        if nd_norm < tol
            converged = true; break
        end
        if f_start - fx < tol^2
            converged = true; break
        end

        # Replace direction of maximum decrease with displacement direction
        D[:, k_max] = new_dir ./ nd_norm

        push!(history, fx)
        isnothing(callback) || callback(iter, fx; grad=nothing, evals=n_evals)
    end

    stats = (evals=n_evals, iters=iter, converged=converged)
    return x, fx, stats
end
