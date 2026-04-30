# ============================================================================
# Gradient/Generic/FirstOrder.jl
# Generic first-order gradient-based optimizers
#
# Interface: f(θ::AbstractVector) -> Real
#            grad!(g::AbstractVector, θ::AbstractVector)  [fills g in-place]
# Returns:   (θ_opt, f_opt, stats) where stats = (evals, iters, converged)
# Convention: minimisation (negate for maximisation problems)
# ============================================================================

using LinearAlgebra

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@inline function _fo_clip!(θ, lb, ub)
    @inbounds for i in eachindex(θ)
        θ[i] = clamp(θ[i], lb[i], ub[i])
    end
end

# Shared convergence-mode predicate used by QN/CG/Newton/TR optimizers.
# mode ∈ (:gradient_norm, :fidelity_change, :both)
#   :gradient_norm   → ‖∇f‖ < tol     (current default — Nocedal & Wright Ch. 3)
#   :fidelity_change → |Δf| < f_tol   (matches Krotov.jl optimize.jl:331-335)
#   :both            → both conditions hold simultaneously
@inline function _converged_mode(mode::Symbol, g_ok::Bool, f_ok::Bool)
    mode === :gradient_norm   && return g_ok
    mode === :fidelity_change && return f_ok
    mode === :both            && return g_ok && f_ok
    throw(ArgumentError("convergence_mode must be :gradient_norm, :fidelity_change, or :both"))
end

@inline function _fo_bounds(n, lower, upper)
    lb = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub = upper === nothing ? fill( Inf, n) : Float64.(upper)
    return lb, ub
end

# Simple backtracking Armijo line search in descent direction d
function _fo_ls(f, θ, d, g, f0; c1=1e-4, β=0.5, max_ls=30)
    α   = 1.0
    dg  = dot(d, g)
    dg >= 0.0 && return 0.0, f0   # not a descent direction
    for _ in 1:max_ls
        f_new = f(θ .+ α .* d)
        f_new <= f0 + c1 * α * dg && return α, f_new
        α *= β
    end
    return α, f(θ .+ α .* d)
end

# ---------------------------------------------------------------------------
# Gradient Descent
# ---------------------------------------------------------------------------

"""
    gd_optimize(f, grad!, θ0; lr, max_iter, tol, lower, upper,
                line_search, verbose) → (θ_opt, f_opt, stats)

Gradient descent (fixed or line-search step size).
Minimises `f`; `grad!(g, θ)` must fill `g` with ∇f(θ) in-place.
"""
function gd_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lr      :: Float64 = 0.01,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    line_search :: Bool = false,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    g       = zeros(n)
    lb, ub  = _fo_bounds(n, lower, upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    f_cur   = f(θ);  n_evals = 1
    converged = false

    for iter in 1:max_iter
        grad!(g, θ);  n_evals += 1
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        if line_search
            α, f_new = _fo_ls(f, θ, -g, g, f_cur)
            n_evals += 1
            θ .+= α .* (-g)
        else
            θ .-= lr .* g
        end
        bounded && _fo_clip!(θ, lb, ub)
        f_cur = line_search ? f(θ) : f(θ)
        n_evals += 1

        verbose && iter % 100 == 0 &&
            @printf("  gd iter %4d  f=%.6e  |g|=%.3e\n", iter, f_cur, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ, f_cur, stats
end

# ---------------------------------------------------------------------------
# Stochastic Gradient Descent
# ---------------------------------------------------------------------------

"""
    sgd_optimize(f, grad!, θ0; lr0, lr_decay, momentum, max_iter, tol,
                 lower, upper, verbose) → (θ_opt, f_opt, stats)

SGD with optional Polyak-Ruppert decaying schedule lr(t) = lr0/(1 + decay*t)
and optional heavy-ball momentum.  For full-batch use, set `batch_grad!` to
the same function as `grad!`.
"""
function sgd_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lr0     :: Float64 = 0.1,
    lr_decay:: Float64 = 1e-3,
    momentum:: Float64 = 0.0,
    max_iter:: Int     = 2_000,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    g       = zeros(n)
    v       = zeros(n)   # momentum buffer
    lb, ub  = _fo_bounds(n, lower, upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    θ_best  = copy(θ)
    f_best  = f(θ);  n_evals = 1
    f_cur   = f_best
    converged = false

    for iter in 1:max_iter
        grad!(g, θ);  n_evals += 1
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        lr = lr0 / (1.0 + lr_decay * iter)
        v  .= momentum .* v .- lr .* g
        θ  .+= v
        bounded && _fo_clip!(θ, lb, ub)

        f_cur = f(θ);  n_evals += 1
        if f_cur < f_best
            f_best = f_cur
            θ_best .= θ
        end

        verbose && iter % 200 == 0 &&
            @printf("  sgd iter %4d  f=%.6e  lr=%.3e  |g|=%.3e\n",
                    iter, f_cur, lr, gnorm)
        isnothing(callback) || callback(iter, f_best; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end

# ---------------------------------------------------------------------------
# Heavy-ball Momentum
# ---------------------------------------------------------------------------

"""
    momentum_optimize(f, grad!, θ0; lr, beta, max_iter, tol,
                      lower, upper, verbose) → (θ_opt, f_opt, stats)

Heavy-ball momentum: v = β·v - α·∇f(θ);  θ += v.
"""
function momentum_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lr      :: Float64 = 0.01,
    beta    :: Float64 = 0.9,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    g       = zeros(n)
    v       = zeros(n)
    lb, ub  = _fo_bounds(n, lower, upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    f_cur   = f(θ);  n_evals = 1
    converged = false

    for iter in 1:max_iter
        grad!(g, θ);  n_evals += 1
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        v  .= beta .* v .- lr .* g
        θ  .+= v
        bounded && _fo_clip!(θ, lb, ub)
        f_cur = f(θ);  n_evals += 1

        verbose && iter % 100 == 0 &&
            @printf("  momentum iter %4d  f=%.6e  |g|=%.3e\n", iter, f_cur, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ, f_cur, stats
end

# ---------------------------------------------------------------------------
# Nesterov Accelerated Gradient
# ---------------------------------------------------------------------------

"""
    nag_optimize(f, grad!, θ0; lr, beta, max_iter, tol,
                 lower, upper, verbose) → (θ_opt, f_opt, stats)

Nesterov accelerated gradient: gradient evaluated at lookahead point.
"""
function nag_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lr      :: Float64 = 0.01,
    beta    :: Float64 = 0.9,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    callback = nothing,
)
    n         = length(θ0)
    θ         = float.(copy(θ0))
    g         = zeros(n)
    v         = zeros(n)
    θ_look    = zeros(n)
    lb, ub    = _fo_bounds(n, lower, upper)
    bounded   = any(isfinite, lb) || any(isfinite, ub)

    f_cur     = f(θ);  n_evals = 1
    converged = false

    for iter in 1:max_iter
        # Lookahead point
        @. θ_look = θ + beta * v
        grad!(g, θ_look);  n_evals += 1
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        v  .= beta .* v .- lr .* g
        θ  .+= v
        bounded && _fo_clip!(θ, lb, ub)
        f_cur = f(θ);  n_evals += 1

        verbose && iter % 100 == 0 &&
            @printf("  nag iter %4d  f=%.6e  |g|=%.3e\n", iter, f_cur, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ, f_cur, stats
end

# ---------------------------------------------------------------------------
# AdaGrad
# ---------------------------------------------------------------------------

"""
    adagrad_optimize(f, grad!, θ0; lr, eps, max_iter, tol,
                     lower, upper, verbose) → (θ_opt, f_opt, stats)

AdaGrad (Duchi et al. 2011): per-parameter adaptive learning rates.
G accumulates squared gradients; effective lr_i = lr / sqrt(G_i + ε).
"""
function adagrad_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lr      :: Float64 = 0.1,
    eps     :: Float64 = 1e-8,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    g       = zeros(n)
    G       = zeros(n)   # accumulated squared gradient
    lb, ub  = _fo_bounds(n, lower, upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    f_cur   = f(θ);  n_evals = 1
    converged = false

    for iter in 1:max_iter
        grad!(g, θ);  n_evals += 1
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        G  .+= g .^ 2
        θ  .-= lr ./ sqrt.(G .+ eps) .* g
        bounded && _fo_clip!(θ, lb, ub)
        f_cur = f(θ);  n_evals += 1

        verbose && iter % 100 == 0 &&
            @printf("  adagrad iter %4d  f=%.6e  |g|=%.3e\n", iter, f_cur, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ, f_cur, stats
end

# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

"""
    rmsprop_optimize(f, grad!, θ0; lr, rho, eps, max_iter, tol,
                     lower, upper, verbose) → (θ_opt, f_opt, stats)

RMSprop (Hinton 2012): exponentially weighted moving average of squared grads.
"""
function rmsprop_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lr      :: Float64 = 0.01,
    rho     :: Float64 = 0.9,
    eps     :: Float64 = 1e-8,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    g       = zeros(n)
    E_g2    = zeros(n)   # EMA of g²
    lb, ub  = _fo_bounds(n, lower, upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    f_cur   = f(θ);  n_evals = 1
    converged = false

    for iter in 1:max_iter
        grad!(g, θ);  n_evals += 1
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        E_g2 .= rho .* E_g2 .+ (1.0 - rho) .* g .^ 2
        θ    .-= lr ./ sqrt.(E_g2 .+ eps) .* g
        bounded && _fo_clip!(θ, lb, ub)
        f_cur = f(θ);  n_evals += 1

        verbose && iter % 100 == 0 &&
            @printf("  rmsprop iter %4d  f=%.6e  |g|=%.3e\n", iter, f_cur, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ, f_cur, stats
end

# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

"""
    adam_optimize(f, grad!, θ0; lr, beta1, beta2, eps, max_iter, tol,
                  lower, upper, amsgrad, verbose) → (θ_opt, f_opt, stats)

Adam (Kingma & Ba 2015): adaptive moment estimation.
Set `amsgrad=true` for the AMSGrad variant (Reddi et al. 2018).
"""
function adam_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lr      :: Float64 = 1e-3,
    beta1   :: Float64 = 0.9,
    beta2   :: Float64 = 0.999,
    eps     :: Float64 = 1e-8,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    amsgrad :: Bool    = false,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    g       = zeros(n)
    m       = zeros(n)    # first moment
    v       = zeros(n)    # second moment
    v_hat_max = zeros(n)  # AMSGrad: running max of v̂
    lb, ub  = _fo_bounds(n, lower, upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    θ_best  = copy(θ)
    f_best  = f(θ);  n_evals = 1
    converged = false

    for t in 1:max_iter
        grad!(g, θ);  n_evals += 1
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        m .= beta1 .* m .+ (1.0 - beta1) .* g
        v .= beta2 .* v .+ (1.0 - beta2) .* g .^ 2

        # Bias-corrected estimates
        m_hat = m ./ (1.0 - beta1^t)
        if amsgrad
            v_hat = v ./ (1.0 - beta2^t)
            v_hat_max .= max.(v_hat_max, v_hat)
            θ .-= lr ./ (sqrt.(v_hat_max) .+ eps) .* m_hat
        else
            v_hat = v ./ (1.0 - beta2^t)
            θ .-= lr ./ (sqrt.(v_hat) .+ eps) .* m_hat
        end
        bounded && _fo_clip!(θ, lb, ub)

        f_cur = f(θ);  n_evals += 1
        if f_cur < f_best
            f_best = f_cur
            θ_best .= θ
        end

        verbose && t % 100 == 0 &&
            @printf("  adam iter %4d  f=%.6e  |g|=%.3e\n", t, f_cur, gnorm)
        isnothing(callback) || callback(t, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end
