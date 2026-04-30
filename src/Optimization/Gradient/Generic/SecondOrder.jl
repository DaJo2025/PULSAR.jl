# ============================================================================
# Gradient/Generic/SecondOrder.jl
# Second-order and quasi-Newton trust-region gradient-based optimizers
#
# Functions:
#   newton_optimize              — Newton-CG (finite-diff or exact Hessian)
#   gauss_newton_optimize        — Gauss-Newton least-squares
#   lm_optimize                  — Levenberg-Marquardt
#   trust_region_newton_optimize — Trust-region Newton with CG-Steihaug
#   projected_gradient_optimize  — Projected gradient (box-constrained)
#
# Interface: f(θ) -> Real,  grad!(g, θ),  [optionally hess!(H, θ) or hvp!(Hv, θ, v)]
# Returns:   (θ_opt, f_opt, stats)
# ============================================================================

using LinearAlgebra

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Finite-difference Hessian-vector product: Hv ≈ (∇f(θ+ε*v) - ∇f(θ)) / ε
function _so_hvp_fd!(Hv, grad!, θ, g, v; ε=1e-5)
    n    = length(θ)
    g2   = zeros(n)
    θ_p  = θ .+ ε .* v
    grad!(g2, θ_p)
    @. Hv = (g2 - g) / ε
end

# CG-Steihaug: solve (H + shift*I) d = -g inside trust region of radius Δ
# H is applied via hvp!(Hv, v)  [in-place]
function _cg_steihaug(hvp!, g, Δ, shift; max_cg=50, cg_tol=0.1)
    n  = length(g)
    d  = zeros(n)
    r  = copy(g)      # r = g + H*d, initially g (d=0)
    p  = -copy(r)
    rs = dot(r, r)

    for _ in 1:max_cg
        Hp = zeros(n)
        hvp!(Hp, p)
        Hp .+= shift .* p
        κ  = dot(p, Hp)

        # Negative curvature: go to boundary along p
        if κ <= 0.0
            a = dot(p, p)
            b = 2.0 * dot(d, p)
            c = dot(d, d) - Δ^2
            disc = max(0.0, b^2 - 4*a*c)
            τ    = (-b + sqrt(disc)) / (2 * a)
            d  .+= τ .* p
            break
        end

        α   = rs / κ
        d_new = d .+ α .* p

        # Exceeded trust region
        if norm(d_new) >= Δ
            a = dot(p, p)
            b = 2.0 * dot(d, p)
            c = dot(d, d) - Δ^2
            disc = max(0.0, b^2 - 4*a*c)
            τ    = (-b + sqrt(disc)) / (2 * a)
            d  .+= τ .* p
            break
        end

        d  .= d_new
        r  .+= α .* Hp
        rs_new = dot(r, r)
        sqrt(rs_new) < cg_tol * sqrt(rs + 1e-30) && break
        β  = rs_new / rs
        p  .= -r .+ β .* p
        rs = rs_new
    end
    return d
end

# Simple backtracking for projected/Newton methods
function _so_backtrack(f, θ, d, g, f0; c1=1e-4, β=0.5, max_ls=30)
    α  = 1.0
    dg = dot(d, g)
    dg >= 0.0 && return 1e-8, f0
    for _ in 1:max_ls
        f_new = f(θ .+ α .* d)
        f_new <= f0 + c1 * α * dg && return α, f_new
        α *= β
    end
    return α, f(θ .+ α .* d)
end

# ---------------------------------------------------------------------------
# Hessian regularization (RFO / Goodwin)
# ---------------------------------------------------------------------------

"""
    _rfo_regularize(H, ε) -> Matrix{Float64}

Rational-function-optimization style Hessian regularization: eigendecompose
`Hsym = (H + H')/2`, clamp eigenvalues to `max(λ_i, ε)`, and rebuild.  Produces
a positive-definite matrix that preserves the eigenvectors of H — convergence
direction is unchanged but the step length is well-defined even on saddles or
in negative-curvature regions.

Reference: Goodwin & Kuprov, J. Chem. Phys. 144, 084107 (2016); Spinach's
`kernel/optimcon/hessreg.m`.
"""
function _rfo_regularize(H::AbstractMatrix, ε::Real=1e-6)
    n = size(H, 1)
    Hs = (H .+ H') ./ 2
    F  = eigen(Hermitian(Hs))
    λc = max.(F.values, Float64(ε))
    return F.vectors * Diagonal(λc) * F.vectors'
end

# ---------------------------------------------------------------------------
# Newton-CG
# ---------------------------------------------------------------------------

"""
    newton_optimize(f, grad!, θ0; hess!, hvp!, shift, max_iter, tol,
                    lower, upper, verbose) → (θ_opt, f_opt, stats)

Newton-CG (Hessian-free) optimizer.  The Newton direction is found by solving
(H + shift·I)·d = -g via conjugate gradients using finite-difference HVPs.

Optionally provide `hess!(H, θ)` for an exact n×n Hessian (used when n is small),
or `hvp!(Hv, θ, v)` for an analytic Hessian-vector product.
"""
function newton_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    hess!   :: Union{Nothing,Function} = nothing,
    hvp!    :: Union{Nothing,Function} = nothing,
    shift   :: Float64 = 1e-4,
    max_iter:: Int     = 200,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    convergence_mode :: Symbol = :gradient_norm,
    f_tol   :: Float64 = 1e-8,
    hessian_regularization :: Symbol = :shift,
    rfo_eps :: Float64 = 1e-6,
    callback = nothing,
)
    hessian_regularization in (:shift, :rfo, :none) ||
        throw(ArgumentError("hessian_regularization must be :shift, :rfo, or :none, got $hessian_regularization"))
    n       = length(θ0)
    θ       = float.(copy(θ0))
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    g       = zeros(n)
    grad!(g, θ);  n_evals = 1
    f_cur   = f(θ);  n_evals += 1
    f_prev  = f_cur
    θ_best  = copy(θ);  f_best = f_cur
    converged = false

    for iter in 1:max_iter
        gnorm = norm(g)
        g_ok  = gnorm < tol
        f_ok  = iter > 1 && abs(f_cur - f_prev) < f_tol
        _converged_mode(convergence_mode, g_ok, f_ok) && (converged = true; break)

        # Compute Newton direction
        if hess! !== nothing
            H = zeros(n, n)
            hess!(H, θ);  n_evals += 1
            H_reg = if hessian_regularization === :rfo
                _rfo_regularize(H, rfo_eps)
            elseif hessian_regularization === :none
                Hermitian((H .+ H') ./ 2)
            else
                H + shift * I
            end
            F_fac = factorize(Hermitian(H_reg))
            d = -(F_fac \ g)
        else
            # Hessian-free via HVP
            _hvp_fn!(Hv, v) = begin
                if hvp! !== nothing
                    hvp!(Hv, θ, v)
                else
                    _so_hvp_fd!(Hv, grad!, θ, g, v)
                    n_evals += 1
                end
            end
            d = _cg_steihaug(_hvp_fn!, g, Inf, shift; max_cg=min(50, n))
        end

        dot(d, g) >= 0.0 && (@. d = -g)  # fallback: steepest descent

        α, f_new = _so_backtrack(f, θ, d, g, f_cur)
        n_evals += 1

        θ .+= α .* d
        bounded && @. θ = clamp(θ, lb, ub)
        f_prev = f_cur
        f_cur  = f_new

        grad!(g, θ);  n_evals += 1
        if f_cur < f_best
            f_best = f_cur;  θ_best .= θ
        end

        verbose && iter % 20 == 0 &&
            @printf("  newton iter %3d  f=%.6e  |g|=%.3e\n", iter, f_cur, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end

# ---------------------------------------------------------------------------
# Gauss-Newton
# ---------------------------------------------------------------------------

"""
    gauss_newton_optimize(f_vec, jac!, θ0; max_iter, tol, lower, upper, verbose)
        → (θ_opt, f_opt, stats)

Gauss-Newton for nonlinear least squares:  min ½||r(θ)||².
`f_vec(θ)` returns the residual vector r; `jac!(J, θ)` fills the Jacobian J.
Solves the normal equations (J'J)·d = -J'r at each step.
"""
function gauss_newton_optimize(
    f_vec   :: Function,
    jac!    :: Function,
    θ0      :: AbstractVector{<:Real};
    max_iter:: Int     = 200,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    r       = f_vec(θ);  m = length(r)
    J       = zeros(m, n)
    n_evals = 1
    f_cur   = 0.5 * dot(r, r)
    θ_best  = copy(θ);  f_best = f_cur
    converged = false

    for iter in 1:max_iter
        jac!(J, θ);  n_evals += 1
        g   = J' * r
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        # Normal equations: (J'J + ε*I) d = -J'r
        JtJ = J' * J + 1e-10 * I
        d   = -(JtJ \ g)

        dot(d, g) >= 0.0 && (@. d = -g)

        # Backtrack
        α = 1.0; f_try = f_cur
        for _ in 1:20
            θ_try = clamp.(θ .+ α .* d, lb, ub)
            r_try = f_vec(θ_try);  n_evals += 1
            f_try = 0.5 * dot(r_try, r_try)
            f_try < f_cur - 1e-4 * α * dot(g, d) && break
            α *= 0.5
        end

        θ  .+= α .* d
        bounded && @. θ = clamp(θ, lb, ub)
        r   = f_vec(θ);  n_evals += 1
        f_cur = 0.5 * dot(r, r)

        if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end

        verbose && iter % 20 == 0 &&
            @printf("  gn iter %3d  f=%.6e  |g|=%.3e\n", iter, f_cur, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end

# ---------------------------------------------------------------------------
# Levenberg-Marquardt
# ---------------------------------------------------------------------------

"""
    lm_optimize(f_vec, jac!, θ0; λ0, λ_up, λ_dn, max_iter, tol,
                lower, upper, verbose) → (θ_opt, f_opt, stats)

Levenberg-Marquardt: Gauss-Newton with adaptive damping λ·I.
`f_vec(θ)` returns residual vector; `jac!(J, θ)` fills Jacobian.
"""
function lm_optimize(
    f_vec   :: Function,
    jac!    :: Function,
    θ0      :: AbstractVector{<:Real};
    λ0      :: Float64 = 1e-3,
    λ_up    :: Float64 = 10.0,
    λ_dn    :: Float64 = 0.1,
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)

    r       = f_vec(θ);  m = length(r)
    J       = zeros(m, n)
    n_evals = 1
    f_cur   = 0.5 * dot(r, r)
    θ_best  = copy(θ);  f_best = f_cur
    λ       = λ0
    converged = false

    for iter in 1:max_iter
        jac!(J, θ);  n_evals += 1
        g   = J' * r
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        JtJ = J' * J
        d   = -((JtJ + λ * I) \ g)
        θ_new = clamp.(θ .+ d, lb, ub)
        r_new = f_vec(θ_new);  n_evals += 1
        f_new = 0.5 * dot(r_new, r_new)

        # Actual vs predicted reduction
        pred = -dot(g, d) - 0.5 * dot(d, JtJ * d)
        ρ    = pred > 1e-30 ? (f_cur - f_new) / pred : 0.0

        if ρ > 0.25
            θ .= θ_new;  r .= r_new;  f_cur = f_new
            λ = max(λ * λ_dn, 1e-12)
            if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end
        else
            λ = min(λ * λ_up, 1e8)
        end

        verbose && iter % 50 == 0 &&
            @printf("  lm iter %4d  f=%.6e  λ=%.2e  |g|=%.3e\n",
                    iter, f_cur, λ, gnorm)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end

# ---------------------------------------------------------------------------
# Trust-Region Newton (CG-Steihaug)
# ---------------------------------------------------------------------------

"""
    trust_region_newton_optimize(f, grad!, θ0; hvp!, Δ0, Δ_max, η, max_iter, tol,
                                 lower, upper, verbose) → (θ_opt, f_opt, stats)

Trust-region Newton with CG-Steihaug subproblem solver.
Optionally supply `hvp!(Hv, θ, v)` for analytic Hessian-vector products;
defaults to finite differences.  Radius Δ adapted by actual/predicted ratio.
"""
function trust_region_newton_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    hvp!    :: Union{Nothing,Function} = nothing,
    Δ0      :: Float64 = 1.0,
    Δ_max   :: Float64 = 100.0,
    η       :: Float64 = 0.1,
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    verbose :: Bool    = false,
    check_invariants :: Bool = false,
    callback = nothing,
)
    n       = length(θ0)
    θ       = float.(copy(θ0))
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    g       = zeros(n)
    grad!(g, θ);  n_evals = 1
    f_cur   = f(θ);  n_evals += 1
    θ_best  = copy(θ);  f_best = f_cur
    Δ       = Δ0
    converged = false

    for iter in 1:max_iter
        gnorm = norm(g)
        gnorm < tol && (converged = true; break)

        _hvp_fn!(Hv, v) = begin
            if hvp! !== nothing
                hvp!(Hv, θ, v)
            else
                _so_hvp_fd!(Hv, grad!, θ, g, v)
                n_evals += 1
            end
        end

        d = _cg_steihaug(_hvp_fn!, g, Δ, 0.0; max_cg=min(n, 50))

        # Predicted reduction
        Hd  = zeros(n)
        _hvp_fn!(Hd, d)
        pred = -(dot(g, d) + 0.5 * dot(d, Hd))
        pred = max(pred, 1e-30)

        θ_new = θ .+ d
        bounded && @. θ_new = clamp(θ_new, lb, ub)
        f_new = f(θ_new);  n_evals += 1
        ρ     = (f_cur - f_new) / pred

        if check_invariants
            ok, msg = check_trust_region_ratio(ρ)
            _assert_invariant(ok, msg, :trust_region_ratio,
                              (; iter=iter, pred=pred, f_cur=f_cur, f_new=f_new))
        end

        if ρ >= η
            θ .= θ_new;  f_cur = f_new
            grad!(g, θ);  n_evals += 1
            if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end
        end

        Δ = _update_tr_radius(Δ, ρ, norm(d), Δ_max)

        verbose && iter % 50 == 0 &&
            @printf("  trn iter %4d  f=%.6e  Δ=%.3e  |g|=%.3e  ρ=%.2f\n",
                    iter, f_cur, Δ, gnorm, ρ)
        isnothing(callback) || callback(iter, f_cur; grad=gnorm, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end

# ---------------------------------------------------------------------------
# Projected Gradient (box-constrained)
# ---------------------------------------------------------------------------

"""
    projected_gradient_optimize(f, grad!, θ0; lower, upper, lr, max_iter, tol,
                                 line_search, verbose) → (θ_opt, f_opt, stats)

Projected gradient descent for box-constrained problems.
At each step: θ ← P[θ - α·∇f(θ)] where P projects onto [lb, ub].
Uses backtracking Armijo on the projected step.
"""
function projected_gradient_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    lr      :: Float64 = 0.01,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    line_search :: Bool = true,
    verbose :: Bool    = false,
    callback = nothing,
)
    n       = length(θ0)
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    θ       = clamp.(float.(θ0), lb, ub)
    g       = zeros(n)

    grad!(g, θ);  n_evals = 1
    f_cur   = f(θ);  n_evals += 1
    θ_best  = copy(θ);  f_best = f_cur
    converged = false

    for iter in 1:max_iter
        # Projected gradient for convergence: ||θ - P[θ-g]||
        g_proj = norm(θ .- clamp.(θ .- g, lb, ub))
        g_proj < tol && (converged = true; break)

        if line_search
            # Backtracking on projected step
            α = lr * 10.0
            for _ in 1:40
                θ_try = clamp.(θ .- α .* g, lb, ub)
                f_try = f(θ_try);  n_evals += 1
                f_try <= f_cur - 1e-4 * dot(g, θ .- θ_try) && (θ .= θ_try; f_cur = f_try; break)
                α *= 0.5
            end
        else
            θ .= clamp.(θ .- lr .* g, lb, ub)
            f_cur = f(θ);  n_evals += 1
        end

        grad!(g, θ);  n_evals += 1
        if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end

        verbose && iter % 100 == 0 &&
            @printf("  pg iter %4d  f=%.6e  |∇P|=%.3e\n", iter, f_cur, g_proj)
        isnothing(callback) || callback(iter, f_cur; grad=g_proj, evals=n_evals)
    end

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end
