# ============================================================================
# Gradient/QOC/HighOrderOC.jl
# High-order optimal control solvers
#
# oc_trust_region_newton_optimize — Trust-region Newton for QOC (HVP via 2nd-order adjoint)
# oc_semismooth_newton_optimize   — Semismooth Newton (projected Newton for box-constrained QOC)
#
# Both functions share the same (f, grad!, θ0) generic interface.
# Optionally supply hvp!(Hv, θ, v) for analytic Hessian-vector products.
# ============================================================================

using LinearAlgebra

# ---------------------------------------------------------------------------
# CG-Steihaug subproblem (self-contained copy)
# ---------------------------------------------------------------------------

function _hoc_cg_steihaug(hvp!, g, Δ, shift; max_cg=50)
    n  = length(g)
    d  = zeros(n)
    r  = copy(g)
    p  = -copy(r)
    Hp = zeros(n)                # hoisted: reused across CG iters
    rs = dot(r, r)
    rs < 1e-30 && return d

    for _ in 1:max_cg
        fill!(Hp, 0.0)
        hvp!(Hp, p)
        @. Hp += shift * p
        κ = dot(p, Hp)
        if κ <= 0.0
            a = dot(p, p); b = 2.0*dot(d, p); c = dot(d, d) - Δ^2
            disc = max(0.0, b^2 - 4*a*c)
            τ = (-b + sqrt(disc)) / (2*a)
            @. d += τ * p
            break
        end
        α = rs / κ
        # ||d + α p||² without allocating d_new
        dd_new = dot(d, d) + 2*α * dot(d, p) + α^2 * dot(p, p)
        if dd_new >= Δ^2
            a = dot(p, p); b = 2.0*dot(d, p); c = dot(d, d) - Δ^2
            disc = max(0.0, b^2 - 4*a*c)
            τ = (-b + sqrt(disc)) / (2*a)
            @. d += τ * p
            break
        end
        @. d += α * p
        @. r += α * Hp
        rs_new = dot(r, r)
        sqrt(rs_new) < 0.1 * sqrt(rs + 1e-30) && break
        β = rs_new / rs
        @. p = -r + β * p
        rs = rs_new
    end
    return d
end

# Finite-difference HVP
function _hoc_hvp_fd!(Hv, grad!, θ, g, v; ε=1e-5,
                       θ_p_buf::Union{Nothing,Vector{Float64}}=nothing,
                       g2_buf::Union{Nothing,Vector{Float64}}=nothing)
    n   = length(θ)
    g2  = g2_buf === nothing ? zeros(n) : g2_buf
    θ_p = θ_p_buf === nothing ? similar(θ) : θ_p_buf
    @. θ_p = θ + ε * v
    grad!(g2, θ_p)
    @. Hv = (g2 - g) / ε
end

# ---------------------------------------------------------------------------
# OC Trust-Region Newton
# ---------------------------------------------------------------------------

"""
    oc_trust_region_newton_optimize(f, grad!, θ0; hvp!, lower, upper,
                                     Δ0, Δ_max, η, shift, max_iter, tol, verbose)
        → (θ_opt, f_opt, stats)

Trust-region Newton for quantum optimal control.
Solves the trust-region subproblem:

    min  g'd + ½d'Hd  s.t. ||d|| ≤ Δ

using CG-Steihaug.  The Hessian-vector product is computed either via
the supplied analytic `hvp!(Hv, θ, v)` or finite differences.

This method achieves quadratic convergence near the optimum, making it
ideal for high-precision pulse optimisation.
"""
function oc_trust_region_newton_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    hvp!    :: Union{Nothing,Function} = nothing,
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    Δ0      :: Float64 = 0.5,
    Δ_max   :: Float64 = 50.0,
    η       :: Float64 = 0.1,
    shift   :: Float64 = 1e-6,
    max_iter:: Int     = 300,
    tol     :: Float64 = 1e-6,
    verbose :: Bool    = true,
    print_interval :: Int = 25,
    callback = nothing,
)
    n       = length(θ0)
    lb      = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub      = upper === nothing ? fill( Inf, n) : Float64.(upper)
    bounded = any(isfinite, lb) || any(isfinite, ub)

    θ       = clamp.(float.(θ0), lb, ub)
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
                _hoc_hvp_fd!(Hv, grad!, θ, g, v)
                n_evals += 1
            end
        end

        d = _hoc_cg_steihaug(_hvp_fn!, g, Δ, shift; max_cg=min(n, 80))

        # Project to feasible cone for bounded problems
        bounded && begin
            for i in 1:n
                if (θ[i] <= lb[i]+1e-12 && d[i] < 0.0) ||
                   (θ[i] >= ub[i]-1e-12 && d[i] > 0.0)
                    d[i] = 0.0
                end
            end
        end

        # Predicted reduction
        Hd   = zeros(n); _hvp_fn!(Hd, d)
        pred = -(dot(g, d) + 0.5 * dot(d, Hd))
        pred = max(pred, 1e-30)

        θ_new   = clamp.(θ .+ d, lb, ub)
        f_new   = f(θ_new);  n_evals += 1
        ρ       = (f_cur - f_new) / pred

        if ρ >= η
            θ .= θ_new;  f_cur = f_new
            grad!(g, θ);  n_evals += 1
            if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end
        end

        Δ = _update_tr_radius(Δ, ρ, norm(d), Δ_max)

        verbose && iter % print_interval == 0 &&
            @printf("  oc_trn iter %4d  F=%.6f  |g|=%.3e  Δ=%.3e  ρ=%.3f\n",
                    iter, -f_cur, gnorm, Δ, ρ)
        isnothing(callback) || callback(iter, -f_cur; grad=gnorm, evals=n_evals)
    end

    verbose &&
        @printf("  oc_trn done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_evals, converged)

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end

# ---------------------------------------------------------------------------
# OC Semismooth Newton
# ---------------------------------------------------------------------------

"""
    oc_semismooth_newton_optimize(f, grad!, θ0; lower, upper, step_ss,
                                   max_iter, tol, verbose) → (θ_opt, f_opt, stats)

Semismooth Newton method for box-constrained optimal control.

Finds a stationary point of the projected gradient optimality condition:

    F(θ) = θ - P[θ - τ·∇f(θ)] = 0

where P is projection onto [lb, ub] and τ is a step parameter.

A generalised Newton step is applied to F:  θ ← θ - (∂F)⁻¹ F(θ)

The generalized Jacobian ∂F is built from the active/inactive set:
  - Inactive index i (lb_i < θ_i < ub_i and projected stays there):
    (∂F)_{ii} = 1 - τ·H_{ii}  (diagonal approximation)
  - Active index i (projection is at a bound):
    (∂F)_{ii} = 1
    right-hand side entry: F_i = θ_i - clamp(θ_i, lb_i, ub_i)

The diagonal Hessian approximation uses a finite-difference diagonal:
  H_{ii} ≈ (g_i(θ + ε·eᵢ) - g_i(θ)) / ε
"""
function oc_semismooth_newton_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step_ss :: Float64 = 0.1,
    max_iter:: Int     = 300,
    tol     :: Float64 = 1e-6,
    verbose :: Bool    = true,
    print_interval :: Int = 25,
    callback = nothing,
)
    n      = length(θ0)
    lb     = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub     = upper === nothing ? fill( Inf, n) : Float64.(upper)
    θ      = clamp.(float.(θ0), lb, ub)
    g      = zeros(n)
    g_p    = zeros(n)   # perturbed gradient for diagonal Hessian
    H_diag = zeros(n)   # diagonal Hessian workspace (reused every outer iter)
    θ_p    = similar(θ) # perturbed-point buffer (reused every outer iter)
    ε_fd   = 1e-5

    grad!(g, θ);  n_evals = 1
    f_cur   = f(θ);  n_evals += 1
    θ_best  = copy(θ);  f_best = f_cur
    converged = false

    for iter in 1:max_iter
        # Projected gradient norm (optimality condition)
        pg_norm = norm(θ .- clamp.(θ .- g, lb, ub))
        pg_norm < tol && (converged = true; break)

        # Diagonal Hessian approximation via FD (restore-in-place, no per-i copy)
        copyto!(θ_p, θ)
        for i in 1:n
            θ_p[i] += ε_fd
            grad!(g_p, θ_p);  n_evals += 1
            H_diag[i] = (g_p[i] - g[i]) / ε_fd
            θ_p[i]   -= ε_fd
        end

        # Semismooth Newton direction
        θ_proj = clamp.(θ .- step_ss .* g, lb, ub)
        F_val  = θ .- θ_proj   # optimality residual

        d = zeros(n)
        for i in 1:n
            # Active set: θ_i at a bound after projection
            if θ_proj[i] <= lb[i] + 1e-10 || θ_proj[i] >= ub[i] - 1e-10
                # Active: Newton step in active-set sense
                d[i] = -F_val[i]
            else
                # Inactive: apply generalised Newton
                jac_ii = 1.0 - step_ss * H_diag[i]
                abs(jac_ii) > 1e-12 ? (d[i] = -F_val[i] / jac_ii) : (d[i] = -F_val[i])
            end
        end

        # Backtracking Armijo on the semismooth step
        α  = 1.0
        F0 = dot(F_val, F_val)
        for _ in 1:30
            θ_try = clamp.(θ .+ α .* d, lb, ub)
            grad!(g_p, θ_try);  n_evals += 1
            F_try = θ_try .- clamp.(θ_try .- step_ss .* g_p, lb, ub)
            dot(F_try, F_try) < (1.0 - 1e-4 * α) * F0 && break
            α *= 0.5
        end

        θ .+= α .* d
        @. θ = clamp(θ, lb, ub)
        f_cur = f(θ);  n_evals += 1
        grad!(g, θ);  n_evals += 1
        if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end

        verbose && iter % print_interval == 0 &&
            @printf("  oc_ssn iter %4d  F=%.6f  |∇P|=%.3e\n", iter, -f_cur, pg_norm)
        isnothing(callback) || callback(iter, -f_cur; grad=pg_norm, evals=n_evals)
    end

    verbose &&
        @printf("  oc_ssn done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_evals, converged)

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return θ_best, f_best, stats
end
