# ============================================================================
# Direct/ConstrainedDirect.jl — COBYLA and LINCOA (linearly constrained DFO)
# ============================================================================
# Constraints expressed as A*θ ≤ b (m_c × n matrix A, m_c vector b).
# Box bounds lower/upper are converted to linear rows automatically.
# ============================================================================

# ── Internal helpers ──────────────────────────────────────────────────────────

_cd_uvec(n::Int, i::Int) = (v = zeros(n); v[i] = 1.0; v)

# Build augmented constraint matrix from (A,b) + box bounds.
function _cd_augment(A, b, lb, ub, n)
    A_aug = A === nothing ? zeros(0, n) : Matrix{Float64}(A)
    b_aug = b === nothing ? Float64[]   : Float64.(b)
    for i in 1:n
        if isfinite(lb[i])
            A_aug = vcat(A_aug, -_cd_uvec(n, i)')
            push!(b_aug, -lb[i])
        end
        if isfinite(ub[i])
            A_aug = vcat(A_aug, _cd_uvec(n, i)')
            push!(b_aug, ub[i])
        end
    end
    return A_aug, b_aug
end

# Max constraint violation: max(0, max_j(A_aug*x - b_aug)).
_cd_viol(x, A, b) = size(A, 1) == 0 ? 0.0 : max(0.0, maximum(A * x .- b))

# Find a feasible step: start from -ρ*normalize(c_f), then scale back
# along each violated constraint direction until feasibility is restored.
function _cd_step(c_f::Vector{Float64}, x_B::Vector{Float64},
                  A::Matrix{Float64}, b::Vector{Float64}, ρ::Float64)
    n  = length(c_f)
    cn = norm(c_f)
    d  = cn > 1e-14 ? -ρ .* (c_f ./ cn) : randn(n) .* (ρ / sqrt(n))

    # Scale down to respect each constraint slack
    m_c = size(A, 1)
    for j in 1:m_c
        aj  = A[j, :]
        ajd = dot(aj, d)
        slack_j = b[j] - dot(aj, x_B)
        if ajd > slack_j + 1e-12
            scale = slack_j > 0 ? slack_j / ajd : 0.0
            d .*= max(0.0, scale)
        end
    end

    # Hard-clip to [-ρ, ρ] box
    d = clamp.(d, -ρ, ρ)
    return d
end

# ── COBYLA ────────────────────────────────────────────────────────────────────

"""
    cobyla_optimize(f, θ0; A, b, lower, upper, max_evals, max_iters, step,
                    tol, penalty) → (θ_best, f_best, stats)

COBYLA-style DFO: linear models on n+1 simplex + penalised merit function for
A*θ ≤ b constraints. Box bounds added as linear rows.
Minimises f; stats = (evals, iters, converged).
"""
function cobyla_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    A         :: Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    b         :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    max_evals :: Int     = 10_000,
    max_iters :: Int     = 2_000,
    step      :: Float64 = 0.1,
    tol       :: Float64 = 1e-6,
    penalty   :: Float64 = 100.0,
    callback = nothing,
)
    n  = length(θ0)
    lb = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub = upper === nothing ? fill( Inf, n) : Float64.(upper)

    A_aug, b_aug = _cd_augment(A, b, lb, ub, n)
    m_c = size(A_aug, 1)

    merit(x, fv, pen) = fv + pen * _cd_viol(x, A_aug, b_aug)

    # ── Initialise simplex ────────────────────────────────────────────────────
    simp  = Vector{Vector{Float64}}(undef, n+1)
    fvals = Vector{Float64}(undef, n+1)
    mvals = Vector{Float64}(undef, n+1)

    simp[1]  = clamp.(Float64.(θ0), lb, ub)
    fvals[1] = f(simp[1]); n_ev = 1
    mvals[1] = merit(simp[1], fvals[1], penalty)

    for i in 1:n
        v = copy(simp[1]); v[i] = clamp(v[i] + step, lb[i], ub[i])
        simp[i+1]  = v
        fvals[i+1] = f(v); n_ev += 1
        mvals[i+1] = merit(v, fvals[i+1], penalty)
    end

    best   = argmin(mvals)
    x_best = copy(simp[best]); f_best = fvals[best]; m_best = mvals[best]
    ρ      = step
    history   = Float64[f_best]
    converged = false
    iter      = 0

    while iter < max_iters && n_ev < max_evals && ρ > tol
        iter += 1
        base = argmin(mvals)
        x_B  = simp[base]

        # Build linear gradient model of f from simplex differences
        nb   = [k for k in 1:n+1 if k != base]
        D    = hcat([simp[k] .- x_B for k in nb]...)   # n × n
        Δf   = [fvals[k] - fvals[base] for k in nb]
        c_f  = try; D' \ Δf; catch; pinv(D') * Δf; end

        # Trial step
        d       = _cd_step(c_f, x_B, A_aug, b_aug, ρ)
        x_trial = clamp.(x_B .+ d, lb, ub)
        f_trial = f(x_trial); n_ev += 1
        m_trial = merit(x_trial, f_trial, penalty)

        if m_trial < m_best
            m_best = m_trial; f_best = f_trial; x_best = copy(x_trial)
            worst  = argmax(mvals)
            simp[worst]  = x_trial
            fvals[worst] = f_trial
            mvals[worst] = m_trial
        else
            ρ *= 0.5
        end

        # Grow penalty each outer iteration to drive feasibility (standard COBYLA)
        penalty = min(penalty * 2.0, 1e8)
        # Recompute merit values with updated penalty
        for k in 1:n+1
            mvals[k] = merit(simp[k], fvals[k], penalty)
        end
        m_best = merit(x_best, f_best, penalty)

        push!(history, f_best)
        isnothing(callback) || callback(iter, f_best; grad=nothing, evals=n_ev)
    end

    converged = ρ <= tol * 2
    stats = (evals=n_ev, iters=iter, converged=converged)
    return x_best, f_best, stats
end

# ── LINCOA ────────────────────────────────────────────────────────────────────

"""
    lincoa_optimize(f, θ0; A, b, lower, upper, max_evals, max_iters, step, tol)
        → (θ_best, f_best, stats)

LINCOA-style DFO: quadratic trust-region model (NEWUOA framework) with linear
constraints A*θ ≤ b enforced by step truncation at constraint boundaries.
Minimises f; stats = (evals, iters, converged).
"""
function lincoa_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    A         :: Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    b         :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    max_evals :: Int     = 10_000,
    max_iters :: Int     = 2_000,
    step      :: Float64 = 0.1,
    tol       :: Float64 = 1e-8,
    callback = nothing,
)
    n  = length(θ0)
    lb = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub = upper === nothing ? fill( Inf, n) : Float64.(upper)

    A_aug, b_aug = _cd_augment(A, b, lb, ub, n)
    m_c = size(A_aug, 1)

    # Project a candidate point back to feasibility by scaling the step α ∈ [0,1]
    function feasible_step(x_base, p)
        α = 1.0
        for j in 1:m_c
            aj  = A_aug[j, :]
            ajp = dot(aj, p)
            if ajp > 0
                slack_j = b_aug[j] - dot(aj, x_base)
                if slack_j < ajp * α
                    α = max(0.0, slack_j / ajp)
                end
            end
        end
        return x_base .+ α .* p
    end

    # Initialise interpolation set (NEWUOA: 2n+1 points)
    m    = 2*n + 1
    pts  = Vector{Vector{Float64}}(undef, m)
    fv   = Vector{Float64}(undef, m)

    project(x) = feasible_step(clamp.(x, lb, ub), zeros(n))   # project to feas.
    pts[1] = clamp.(feasible_step(Float64.(θ0), zeros(n)), lb, ub)
    fv[1]  = f(pts[1]); n_ev = 1
    k = 2
    for i in 1:n
        k > m && break
        xp = feasible_step(clamp.(Float64.(θ0) .+ step .* _cd_uvec(n,i), lb, ub), zeros(n))
        pts[k] = xp; fv[k] = f(xp); n_ev += 1; k += 1
        k > m && break
        xm = feasible_step(clamp.(Float64.(θ0) .- step .* _cd_uvec(n,i), lb, ub), zeros(n))
        pts[k] = xm; fv[k] = f(xm); n_ev += 1; k += 1
    end

    base   = argmin(fv)
    x_best = copy(pts[base]); f_best = fv[base]
    Δ = step; Δ_max = step * 100.0
    history   = Float64[f_best]
    converged = false
    iter      = 0

    while iter < max_iters && n_ev < max_evals && Δ > tol
        iter += 1
        nb  = [k for k in 1:m if k != base]
        Y   = hcat([pts[k] .- pts[base] for k in nb]...)
        rhs = [fv[k] - fv[base] for k in nb]

        # Reuse quadratic model helpers from QuadraticModels.jl
        g, H  = _qm_fit(Y, rhs)
        p_unc = _qm_trs(g, H, Δ)

        # Project step to satisfy linear constraints
        x_new = feasible_step(pts[base], p_unc)
        x_new = clamp.(x_new, lb, ub)
        p_act = x_new .- pts[base]
        norm(p_act) < 1e-14 && (Δ /= 4; continue)

        f_new  = f(x_new); n_ev += 1
        pred   = -(g' * p_act + 0.5 * p_act' * (H * p_act))
        actual = fv[base] - f_new
        ρ_rat  = pred > 1e-14 ? actual / pred : 0.0

        ρ_rat > 0.7 && (Δ = min(Δ * 2.0, Δ_max))
        ρ_rat < 0.1 && (Δ = max(Δ / 4.0, tol))

        if ρ_rat > 0.0 || f_new < fv[base]
            worst      = nb[argmax([fv[k] for k in nb])]
            pts[worst] = x_new; fv[worst] = f_new
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
