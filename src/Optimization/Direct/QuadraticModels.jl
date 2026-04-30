# ============================================================================
# Direct/QuadraticModels.jl — Trust-region DFO: UOBYQA, NEWUOA, BOBYQA
# ============================================================================
# Shared quadratic-model trust-region framework (Powell-style).
# UOBYQA: (n+1)(n+2)/2 interpolation points (exact quadratic fit).
# NEWUOA: 2n+1 points (min-Frobenius-norm quadratic fit).
# BOBYQA: 2n+1 points + box projection.
# ============================================================================

# ── Internal helpers ──────────────────────────────────────────────────────────

_qm_uvec(n::Int, i::Int) = (v = zeros(n); v[i] = 1.0; v)

# Quadratic-model feature vector for displacement y.
# Q(y) = g'y + ½ y'Hy  ←→  φ(y)' θ   where θ = [g; diag(H); upper-tri(H)]
function _qm_feats(y::AbstractVector{Float64})
    n = length(y)
    k = n*(n+3)÷2
    v = Vector{Float64}(undef, k)
    v[1:n] .= y
    col = n + 1
    for i in 1:n
        v[col] = 0.5 * y[i]^2; col += 1
        for j in i+1:n
            v[col] = y[i] * y[j]; col += 1
        end
    end
    return v
end

# Extract (g, H) from coefficient vector (same ordering as _qm_feats).
function _qm_extract(θ::Vector{Float64}, n::Int)
    g = copy(θ[1:n])
    H = zeros(n, n)
    col = n + 1
    for i in 1:n
        H[i, i] = θ[col]; col += 1
        for j in i+1:n
            H[i, j] = H[j, i] = θ[col]; col += 1
        end
    end
    return g, H
end

# Fit quadratic model Q(y) = g'y + ½ y'Hy from (m × n) displacements Y
# (columns) and corresponding function differences rhs.
function _qm_fit(Y::Matrix{Float64}, rhs::Vector{Float64})
    n, m   = size(Y)
    np     = n*(n+3)÷2
    A      = Matrix{Float64}(undef, m, np)
    for k in 1:m
        A[k, :] = _qm_feats(Y[:, k])
    end
    coeffs = m >= np ? (A \ rhs) : (pinv(A) * rhs)
    return _qm_extract(coeffs, n)
end

# Trust-region subproblem: minimise g'p + ½ p'Hp  s.t. ‖p‖ ≤ Δ.
function _qm_trs(g::Vector{Float64}, H::Matrix{Float64}, Δ::Float64)
    n   = length(g)
    Δ < 1e-14 && return zeros(n)
    eig = eigen(Symmetric(H))
    d   = eig.values
    Q   = eig.vectors
    Qtg = Q' * g

    # Unconstrained Newton step (only valid when H ≻ 0)
    d_safe = max.(d, 1e-10)
    p_free = -Q * (Qtg ./ d_safe)
    all(d .> 1e-10) && norm(p_free) <= Δ && return p_free

    # Find λ* via bisection: ‖(H+λI)⁻¹g‖ = Δ
    λ_lo = max(0.0, -minimum(d)) + 1e-10
    λ_hi = max(norm(g) / Δ, λ_lo) + 1e-6
    for _ in 1:80                        # expand until step fits in ball
        p = -Q * (Qtg ./ (d .+ λ_hi))
        norm(p) <= Δ && break
        λ_hi *= 2.0
    end
    for _ in 1:60
        λ = 0.5 * (λ_lo + λ_hi)
        norm(-Q * (Qtg ./ (d .+ λ))) > Δ ? (λ_lo = λ) : (λ_hi = λ)
        λ_hi - λ_lo < 1e-13 * (1.0 + λ_lo) && break
    end
    return -Q * (Qtg ./ (d .+ 0.5*(λ_lo + λ_hi)))
end

# ── Shared DFO loop ───────────────────────────────────────────────────────────

function _qm_dfo(
    f          :: Function,
    x0         :: AbstractVector{<:Real},
    m_target   :: Int;                    # number of interpolation points
    lower      :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper      :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    box_proj   :: Bool    = false,
    step       :: Float64 = 0.1,
    tol        :: Float64 = 1e-8,
    max_evals  :: Int     = 10_000,
    max_iters  :: Int     = 2_000,
    callback               = nothing,
)
    n    = length(x0)
    lb   = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub   = upper === nothing ? fill( Inf, n) : Float64.(upper)
    clip = box_proj ? (x -> clamp.(x, lb, ub)) : identity

    m    = m_target
    pts  = Vector{Vector{Float64}}(undef, m)
    fv   = Vector{Float64}(undef, m)

    # ── Initialise interpolation points ──────────────────────────────────────
    pts[1] = clip(Float64.(x0)); fv[1] = f(pts[1]); n_ev = 1
    k = 2
    for i in 1:n                                    # x0 ± step·eᵢ
        k > m && break
        pts[k] = clip(x0 .+ step .* _qm_uvec(n, i)); fv[k] = f(pts[k]); n_ev += 1; k += 1
        k > m && break
        pts[k] = clip(x0 .- step .* _qm_uvec(n, i)); fv[k] = f(pts[k]); n_ev += 1; k += 1
    end
    for i in 1:n, j in i+1:n                       # cross terms for UOBYQA
        k > m && break
        pts[k] = clip(x0 .+ step .* (_qm_uvec(n,i) .+ _qm_uvec(n,j)))
        fv[k]  = f(pts[k]); n_ev += 1; k += 1
    end
    while k <= m                                    # fallback (should not occur)
        pts[k] = clip(x0 .+ step .* randn(n)); fv[k] = f(pts[k]); n_ev += 1; k += 1
    end

    base  = argmin(fv)
    x_best = copy(pts[base]); f_best = fv[base]
    Δ = step; Δ_max = step * 100.0
    history   = Float64[f_best]
    converged = false
    iter      = 0

    while iter < max_iters && n_ev < max_evals && Δ > tol
        iter += 1

        nb  = [k for k in 1:m if k != base]        # non-base indices
        Y   = hcat([pts[k] .- pts[base] for k in nb]...)   # n × (m-1)
        rhs = [fv[k] - fv[base] for k in nb]

        # Fit model and compute trust-region step
        g, H  = _qm_fit(Y, rhs)
        p     = _qm_trs(g, H, Δ)
        x_new = clip(pts[base] .+ p)
        p_act = x_new .- pts[base]

        norm(p_act) < 1e-14 && (Δ /= 4; continue)

        f_new  = f(x_new); n_ev += 1
        pred   = -(g' * p_act + 0.5 * p_act' * (H * p_act))   # predicted decrease
        actual = fv[base] - f_new                               # actual decrease
        ρ_rat  = pred > 1e-14 ? actual / pred : 0.0

        # Update trust-region radius
        ρ_rat > 0.7 && (Δ = min(Δ * 2.0, Δ_max))
        ρ_rat < 0.1 && (Δ = max(Δ / 4.0, tol))

        # Update interpolation set (replace worst non-base point)
        if ρ_rat > 0.0 || f_new < fv[base]
            worst        = nb[argmax([fv[k] for k in nb])]
            pts[worst]   = x_new
            fv[worst]    = f_new
            f_new < f_best && (f_best = f_new; x_best = copy(x_new))
            base = argmin(fv)
        end

        push!(history, f_best)
        isnothing(callback) || callback(iter, f_best; grad=nothing, evals=n_ev)
    end

    converged = Δ <= tol * 2
    stats = (evals=n_ev, iters=iter, converged=converged)
    return x_best, f_best, stats
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    uobyqa_optimize(f, θ0; max_evals, max_iters, lower, upper, step, tol)
        → (θ_best, f_best, stats)

Unconstrained quadratic trust-region DFO using (n+1)(n+2)/2 interpolation
points (exact quadratic fit). Powell-style UOBYQA formulation.
Minimises f; stats = (evals, iters, converged).
"""
function uobyqa_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 10_000,
    max_iters :: Int     = 2_000,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step      :: Float64 = 0.1,
    tol       :: Float64 = 1e-8,
    callback = nothing,
)
    n = length(θ0)
    m = (n+1)*(n+2)÷2
    return _qm_dfo(f, θ0, m; lower=lower, upper=upper, box_proj=false,
                   step=step, tol=tol, max_evals=max_evals, max_iters=max_iters,
                   callback=callback)
end

"""
    newuoa_optimize(f, θ0; max_evals, max_iters, lower, upper, step, tol)
        → (θ_best, f_best, stats)

Unconstrained quadratic trust-region DFO using 2n+1 interpolation points
(min-Frobenius-norm quadratic fit). Powell-style NEWUOA formulation.
Minimises f; stats = (evals, iters, converged).
"""
function newuoa_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 10_000,
    max_iters :: Int     = 2_000,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step      :: Float64 = 0.1,
    tol       :: Float64 = 1e-8,
    callback = nothing,
)
    n = length(θ0)
    m = 2*n + 1
    return _qm_dfo(f, θ0, m; lower=lower, upper=upper, box_proj=false,
                   step=step, tol=tol, max_evals=max_evals, max_iters=max_iters,
                   callback=callback)
end

"""
    bobyqa_optimize(f, θ0; max_evals, max_iters, lower, upper, step, tol)
        → (θ_best, f_best, stats)

Box-constrained quadratic trust-region DFO using 2n+1 interpolation points.
Powell-style BOBYQA formulation; steps projected to satisfy lower/upper bounds.
Minimises f; stats = (evals, iters, converged).
"""
function bobyqa_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 10_000,
    max_iters :: Int     = 2_000,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    step      :: Float64 = 0.1,
    tol       :: Float64 = 1e-8,
    callback = nothing,
)
    n = length(θ0)
    m = 2*n + 1
    return _qm_dfo(f, θ0, m; lower=lower, upper=upper, box_proj=true,
                   step=step, tol=tol, max_evals=max_evals, max_iters=max_iters,
                   callback=callback)
end
