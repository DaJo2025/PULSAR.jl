# ============================================================================
# Gradient/QOC/GRAPEFamily.jl
# GRAPE-family optimizers for quantum optimal control
#
# Generic interface (dispatches on first arg type Function):
#   grape_optimize(f, grad!, θ0; ...)
#   grape_cg_optimize(f, grad!, θ0; ...)
#   grape_lbfgsb_optimize(f, grad!, θ0; ...)
#
# Intended use with grape_state_kernel from MR/GRAPEState.jl:
#   fid, g = grape_state_kernel(reshape(θ, n_ctrl, N_ts), ctrl)
#   f(θ)   = -fid
#   grad!(g_out, θ) = begin
#       fid, grad_mat = grape_state_kernel(reshape(θ, n_ctrl, N_ts), ctrl)
#       g_out .= -vec(grad_mat)
#   end
#
# NOTE: the existing grape_optimize(system, target; ...) in GRAPE.jl operates
#       on AbstractQuantumSystem — different first argument type, no dispatch conflict.
# ============================================================================

using LinearAlgebra

# Strong-Wolfe line search lives in Gradient/_LineSearch.jl.
# GRAPE-family callers use the simple bracket (α_max=5.0, max_iter=40,
# zoom_iter=30, zoom_eps=1e-13).

@inline function _gf_ls!(θ_t, g_buf, f, grad!, θ, d, g0, f0)
    wolfe_line_search!(θ_t, g_buf, f, grad!, θ, d, g0, f0;
                       α_max=5.0, max_iter=40,
                       zoom_iter=30, zoom_eps=1e-13,
                       two_point_bracket=false)
end

# ---------------------------------------------------------------------------
# L-BFGS two-loop (same as QuasiNewton, self-contained copy)
# ---------------------------------------------------------------------------

# Delegate to the canonical L-BFGS two-loop in Generic/QuasiNewton.jl.
@inline _gf_lbfgs_dir!(d, g, S, Y, ρ_list) =
    _lbfgs_direction!(d, g, S, Y, ρ_list, length(S))

# ---------------------------------------------------------------------------
# GRAPE (gradient ascent — standard form)
# ---------------------------------------------------------------------------

"""
    grape_optimize(f, grad!, θ0; lower, upper, step, max_iter, tol,
                   adaptive, verbose) → (θ_opt, f_opt, stats)

Standard GRAPE gradient ascent with adaptive step size.
`f` is the (negated) fidelity to minimise; `grad!` fills ∇f in-place.
Step size is scaled by gradient norm to give unit-norm steps, then adapted
by tracking improvement (halved on no progress, grown on consistent progress).

Dispatch note: this method matches `grape_optimize(f::Function, ...)` and does
NOT conflict with the existing `grape_optimize(system::AbstractQuantumSystem, ...)`
defined in `Optimization/GRAPE.jl`.
"""
function grape_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step    :: Float64 = 0.1,
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    adaptive:: Bool    = true,
    verbose :: Bool    = true,
    print_interval :: Int = 50,
    callback = nothing,
)
    t_start = time()
    n       = length(θ0)
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    θ       = clamp.(float.(θ0), lb, ub)
    g       = zeros(n)

    grad!(g, θ);  n_fid = 0;  n_grd = 1
    f_cur   = f(θ);  n_fid += 1
    θ_best  = copy(θ);  f_best = f_cur
    α       = step
    no_prog = 0
    converged = false
    n_iter  = 0
    fid_hist  = Float64[-f_cur]
    grad_hist = Float64[norm(g)]

    for iter in 1:max_iter
        n_iter = iter
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        # Normalised gradient direction
        d    = -g ./ max(gnorm, 1e-30)
        θ_new = clamp.(θ .+ α .* d, lb, ub)
        f_new = f(θ_new);  n_fid += 1
        grad_new = zeros(n)
        grad!(grad_new, θ_new);  n_grd += 1

        if f_new < f_cur
            θ  .= θ_new
            g  .= grad_new
            f_cur = f_new
            no_prog = 0
            if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end
            adaptive && (α = min(α * 1.05, step * 10.0))
        else
            no_prog += 1
            adaptive && (α = max(α * 0.5, step * 1e-4))
            no_prog > 20 && break
        end

        push!(fid_hist, -f_cur)
        push!(grad_hist, gnorm)
        verbose && iter % print_interval == 0 &&
            @printf("  grape iter %4d  F=%.6f  α=%.3e  |g|=%.3e\n",
                    iter, -f_cur, α, gnorm)
        isnothing(callback) || callback(iter, -f_cur; grad=gnorm, evals=n_fid+n_grd)
    end

    verbose &&
        @printf("  grape done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_fid + n_grd, converged)

    reason = converged ? "gradient norm < tol ($tol)" : "maximum iterations reached"
    return OptimizationResult(
        reshape(copy(θ_best), 1, n),
        -f_best,
        fid_hist,
        grad_hist,
        n_iter,
        converged,
        reason,
        time() - t_start,
        n_fid,
        n_grd,
        Dict{String,Any}("algorithm" => "GRAPE"),
    )
end

# ---------------------------------------------------------------------------
# GRAPE-CG (nonlinear CG direction instead of raw gradient)
# ---------------------------------------------------------------------------

"""
    grape_cg_optimize(f, grad!, θ0; lower, upper, max_iter, tol, cg_method,
                      verbose) → (θ_opt, f_opt, stats)

GRAPE with nonlinear conjugate gradient update (PR+ by default).
Uses strong Wolfe line search.  Better convergence than basic GRAPE on
smooth fidelity landscapes.
"""
function grape_cg_optimize(
    f         :: Function,
    grad!     :: Function,
    θ0        :: AbstractVector{<:Real};
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    max_iter  :: Int     = 500,
    tol       :: Float64 = 1e-6,
    cg_method :: Symbol  = :PR,
    verbose   :: Bool    = true,
    print_interval :: Int = 50,
    callback = nothing,
)
    t_start = time()
    n       = length(θ0)
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    θ       = clamp.(float.(θ0), lb, ub)
    g       = zeros(n)
    g_old   = zeros(n)
    d       = zeros(n)
    θ_t     = similar(θ)                  # wolfe trial buffer (hoisted)
    g_ls    = zeros(n)                    # wolfe gradient scratch (hoisted)

    grad!(g, θ);  n_fid = 0;  n_grd = 1
    f_cur   = f(θ);  n_fid += 1
    @. d    = -g
    @. g_old= g
    θ_best  = copy(θ);  f_best = f_cur
    converged = false
    n_iter  = 0
    fid_hist  = Float64[-f_cur]
    grad_hist = Float64[norm(g)]

    for iter in 1:max_iter
        n_iter = iter
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        # Ensure descent
        dot(d, g) >= 0.0 && (@. d = -g)
        bounded && begin
            for i in 1:n
                if (θ[i] <= lb[i] + 1e-12 && d[i] < 0.0) ||
                   (θ[i] >= ub[i] - 1e-12 && d[i] > 0.0)
                    d[i] = 0.0
                end
            end
            norm(d) < 1e-14 && (@. d = -g)
        end

        α, f_new = _gf_ls!(θ_t, g_ls, f, grad!, θ, d, g, f_cur)
        n_fid += 2

        θ .+= α .* d
        bounded && @. θ = clamp(θ, lb, ub)
        f_cur = f_new

        grad!(g, θ);  n_grd += 1
        if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end

        # CG β coefficient (auto-restart every n iters)
        restart = iter % n == 0 || dot(g_old, g_old) < 1e-30
        β       = restart ? 0.0 : _cg_beta(g, g_old, d, cg_method)
        @. d    = -g + β * d
        @. g_old = g

        push!(fid_hist, -f_cur)
        push!(grad_hist, gnorm)
        verbose && iter % print_interval == 0 &&
            @printf("  grape_cg(%s) iter %4d  F=%.6f  |g|=%.3e\n",
                    cg_method, iter, -f_cur, gnorm)
        isnothing(callback) || callback(iter, -f_cur; grad=gnorm, evals=n_fid+n_grd)
    end

    verbose &&
        @printf("  grape_cg done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_fid + n_grd, converged)

    reason = converged ? "gradient norm < tol ($tol)" : "maximum iterations reached"
    return OptimizationResult(
        reshape(copy(θ_best), 1, n),
        -f_best,
        fid_hist,
        grad_hist,
        n_iter,
        converged,
        reason,
        time() - t_start,
        n_fid,
        n_grd,
        Dict{String,Any}("algorithm" => "GRAPE-CG ($cg_method)"),
    )
end

# ---------------------------------------------------------------------------
# GRAPE-L-BFGS-B (box-constrained L-BFGS update)
# ---------------------------------------------------------------------------

"""
    grape_lbfgsb_optimize(f, grad!, θ0; lower, upper, memory, max_iter, tol,
                          verbose) → (θ_opt, f_opt, stats)

GRAPE with L-BFGS-B quasi-Newton update.  Provides superlinear convergence
for smooth quantum control problems.  Box constraints enforced by projected
direction and gradient clipping.

This is the recommended method for high-fidelity NMR pulse optimisation.
"""
function grape_lbfgsb_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    memory  :: Int     = 10,
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    verbose :: Bool    = true,
    print_interval :: Int = 50,
    callback = nothing,
)
    t_start = time()
    n       = length(θ0)
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)

    θ       = clamp.(float.(θ0), lb, ub)
    g       = zeros(n)
    g_new   = zeros(n)
    d       = zeros(n)
    θ_t     = similar(θ)                   # wolfe trial buffer (hoisted)
    g_ls    = zeros(n)                     # wolfe gradient scratch (hoisted)
    S       = Vector{Vector{Float64}}()
    Y       = Vector{Vector{Float64}}()
    ρ_list  = Float64[]

    grad!(g, θ);  n_fid = 0;  n_grd = 1
    f_cur   = f(θ);  n_fid += 1
    θ_best  = copy(θ);  f_best = f_cur
    converged = false
    n_iter  = 0
    fid_hist  = Float64[-f_cur]
    grad_hist = Float64[norm(g)]

    for iter in 1:max_iter
        n_iter = iter
        # Projected gradient norm for convergence
        pg_norm = norm(θ .- clamp.(θ .- g, lb, ub))
        pg_norm < tol && (converged = true; break)

        # L-BFGS direction
        _gf_lbfgs_dir!(d, g, S, Y, ρ_list)

        # Project direction onto feasible cone
        for i in 1:n
            if (θ[i] <= lb[i] + 1e-12 && d[i] < 0.0) ||
               (θ[i] >= ub[i] - 1e-12 && d[i] > 0.0)
                d[i] = 0.0
            end
        end
        norm(d) < 1e-14 && break

        α, f_new = _gf_ls!(θ_t, g_ls, f, grad!, θ, d, g, f_cur)
        n_fid += 2

        s = α .* d              # freshly allocated; moved directly into S
        θ .+= s
        @. θ = clamp(θ, lb, ub)
        f_cur = f_new

        grad!(g_new, θ);  n_grd += 1
        y  = g_new .- g         # fresh; moved directly into Y below
        sy = dot(s, y)
        if sy > 1e-14 * dot(s, s)
            push!(S, s); push!(Y, y); push!(ρ_list, 1.0/sy)
            length(S) > memory && (popfirst!(S); popfirst!(Y); popfirst!(ρ_list))
        end
        @. g = g_new
        if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end

        push!(fid_hist, -f_cur)
        push!(grad_hist, pg_norm)
        verbose && iter % print_interval == 0 &&
            @printf("  grape_lbfgsb iter %4d  F=%.6f  |∇P|=%.3e  m=%d\n",
                    iter, -f_cur, pg_norm, length(S))
        isnothing(callback) || callback(iter, -f_cur; grad=pg_norm, evals=n_fid+n_grd)
    end

    verbose &&
        @printf("  grape_lbfgsb done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_fid + n_grd, converged)

    reason = converged ? "projected gradient norm < tol ($tol)" : "maximum iterations reached"
    return OptimizationResult(
        reshape(copy(θ_best), 1, n),
        -f_best,
        fid_hist,
        grad_hist,
        n_iter,
        converged,
        reason,
        time() - t_start,
        n_fid,
        n_grd,
        Dict{String,Any}("algorithm" => "GRAPE-L-BFGS-B", "lbfgs_memory" => memory),
    )
end
