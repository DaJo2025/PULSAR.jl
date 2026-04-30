# ============================================================================
# Gradient/Generic/ConjugateGradient.jl
# Nonlinear Conjugate Gradient optimizer
#
# Interface: f(θ) -> Real,  grad!(g, θ)  [in-place gradient]
# Returns:   (θ_opt, f_opt, stats)
# ============================================================================

using LinearAlgebra

# Strong Wolfe line search: sufficient decrease (Armijo) + curvature condition
function _cg_wolfe_ls(f, grad!, θ, d, g0, f0; c1=1e-4, c2=0.1,
                       α_max=10.0, max_iter=50)
    α_lo, α_hi = 0.0, α_max
    α  = min(1.0, α_max)
    f_lo = f0
    g_buf = similar(g0)
    dg0  = dot(d, g0)
    dg0 >= 0.0 && return 1e-6, f0   # not descent

    function phi(a)
        f(θ .+ a .* d)
    end
    function dphi(a)
        grad!(g_buf, θ .+ a .* d)
        dot(d, g_buf)
    end

    for _ in 1:max_iter
        fa = phi(α)
        if fa > f0 + c1 * α * dg0 || fa >= f_lo
            α_hi = α
        else
            dga = dphi(α)
            abs(dga) <= -c2 * dg0 && return α, fa
            dga >= 0.0 && (α_hi = α)
            α_lo = α
            f_lo = fa
        end
        α = (α_lo + α_hi) * 0.5
        abs(α_hi - α_lo) < 1e-12 * (1.0 + abs(α)) && break
    end
    return α, phi(α)
end

# ---------------------------------------------------------------------------
# Nonlinear Conjugate Gradient
# ---------------------------------------------------------------------------

"""
    cg_optimize(f, grad!, θ0; method, restart_iter, max_iter, tol,
                lower, upper, verbose) → (θ_opt, f_opt, stats)

Nonlinear conjugate gradient with strong Wolfe line search.

`method` options:
- `:FR`  — Fletcher-Reeves
- `:PR`  — Polak-Ribière (β = max(0, β_PR), i.e. PR+)
- `:HS`  — Hestenes-Stiefel
- `:DY`  — Dai-Yuan

Automatic restart when direction is not sufficiently descent or every
`restart_iter` iterations.
"""
function cg_optimize(
    f            :: Function,
    grad!        :: Function,
    θ0           :: AbstractVector{<:Real};
    method       :: Symbol  = :PR,
    restart_iter :: Int     = 0,      # 0 = auto (restart every n iters)
    max_iter     :: Int     = 2_000,
    tol          :: Float64 = 1e-6,
    lower        :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper        :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose      :: Bool    = false,
    convergence_mode :: Symbol = :gradient_norm,
    f_tol        :: Float64 = 1e-8,
    callback = nothing,
)
    n          = length(θ0)
    θ          = float.(copy(θ0))
    lb         = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub         = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded    = any(isfinite, lb) || any(isfinite, ub)
    restart_n  = restart_iter > 0 ? restart_iter : n

    g     = zeros(n)
    g_old = zeros(n)
    d     = zeros(n)

    grad!(g, θ);  n_evals = 1
    @. d = -g
    @. g_old = g
    f_cur  = f(θ);  n_evals += 1
    f_prev = f_cur

    θ_best    = copy(θ)
    f_best    = f_cur
    converged = false

    for iter in 1:max_iter
        gnorm = norm(g)
        g_ok  = gnorm < tol
        f_ok  = iter > 1 && abs(f_cur - f_prev) < f_tol
        _converged_mode(convergence_mode, g_ok, f_ok) && (converged = true; break)

        # Line search
        α, f_new = _cg_wolfe_ls(f, grad!, θ, d, g, f_cur)
        n_evals += 2

        θ .+= α .* d
        bounded && @. θ = clamp(θ, lb, ub)
        f_prev = f_cur
        f_cur  = f_new

        grad!(g, θ);  n_evals += 1
        if f_cur < f_best
            f_best = f_cur
            θ_best .= θ
        end

        # Compute β (restart forces β = 0 at the auto-restart boundary)
        restart = (iter % restart_n == 0) || dot(g_old, g_old) < 1e-30
        β       = restart ? 0.0 : _cg_beta(g, g_old, d, method)

        @. d = -g + β * d
        # Check descent
        dot(d, g) >= 0.0 && (@. d = -g)   # force restart

        @. g_old = g

        verbose && iter % 100 == 0 &&
            @printf("  cg(%s) iter %4d  f=%.6e  |g|=%.3e  β=%.3e\n",
                    method, iter, f_cur, gnorm, β)
        isnothing(callback) || callback(iter, f_best; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end
