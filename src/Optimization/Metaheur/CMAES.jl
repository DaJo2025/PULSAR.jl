# ============================================================================
# Metaheur/CMAES.jl — Generic CMA-ES and PS-CMA-ES
# ============================================================================

"""
    cmaes_optimize(f, θ0; max_evals, max_iters, lower, upper, popsize, seed,
                   sigma_init, tol_fun, tol_x, maximize) → (θ_best, f_best, stats)

Full-covariance CMA-ES. Minimises `f`; set `maximize=true` to maximise.
"""
function cmaes_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 100_000,
    max_iters :: Int     = 5_000,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    popsize   :: Int     = 0,
    seed      :: Union{Nothing,Int} = nothing,
    sigma_init:: Float64 = 0.3,
    tol_fun   :: Float64 = 1e-11,
    tol_x     :: Float64 = 1e-12,
    maximize  :: Bool    = false,
    check_invariants :: Bool = false,
    callback = nothing,
)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    N   = length(θ0)
    obj = maximize ? (x -> -f(x)) : f
    lb  = lower === nothing ? nothing : Float64.(lower)
    ub  = upper === nothing ? nothing : Float64.(upper)

    _clip(x) = lb === nothing ? x : clamp.(x, lb, ub)

    λ = popsize > 0 ? popsize : 4 + floor(Int, 3 * log(N))
    μ = max(1, λ ÷ 2)

    raw_w = [log(μ + 0.5) - log(i) for i in 1:μ]
    w     = raw_w ./ sum(raw_w)
    μ_eff = 1.0 / sum(w .^ 2)

    c_σ = (μ_eff + 2.0) / (N + μ_eff + 5.0)
    d_σ = 1.0 + 2.0 * max(0.0, sqrt((μ_eff - 1.0)/(N + 1.0)) - 1.0) + c_σ
    c_c = (4.0 + μ_eff/N) / (N + 4.0 + 2.0*μ_eff/N)
    c_1 = 2.0 / ((N + 1.3)^2 + μ_eff)
    c_μ = min(1.0 - c_1, 2.0*(μ_eff - 2.0 + 1.0/μ_eff)/((N + 2.0)^2 + μ_eff))
    χ_N = sqrt(N) * (1.0 - 1.0/(4N) + 1.0/(21*N^2))

    m   = _clip(Float64.(θ0))
    σ   = sigma_init
    C   = Matrix{Float64}(I, N, N)
    p_σ = zeros(N)
    p_c = zeros(N)

    F0      = obj(m)
    θ_best  = copy(m)
    f_best  = F0
    n_evals = 1
    history = Float64[]

    iter = 0
    while iter < max_iters && n_evals < max_evals
        iter += 1

        # Eigendecomposition of C
        eig  = eigen(Symmetric(C))
        D    = sqrt.(max.(eig.values, 0.0))
        B    = eig.vectors
        sqC  = B * Diagonal(D) * B'
        invD = Diagonal(1.0 ./ max.(D, 1e-14))
        invsqC = B * invD * B'

        zs = [randn(rng, N) for _ in 1:λ]
        xs = [_clip(m .+ σ .* (sqC * z)) for z in zs]
        # Serial fitness evaluation. Nested @threads :static (used inside
        # GRAPE kernels with per-thread buffers keyed on threadid()) cannot
        # run under an outer @threads, so parallelism stays in the kernel.
        fs = Vector{Float64}(undef, λ)
        for k in 1:λ
            fs[k] = obj(xs[k])
        end
        n_evals += λ

        order  = sortperm(fs)
        xs_sel = xs[order[1:μ]]
        zs_sel = zs[order[1:μ]]

        if fs[order[1]] < f_best
            f_best = fs[order[1]]
            θ_best = copy(xs_sel[1])
        end
        push!(history, f_best)
        isnothing(callback) || callback(iter, maximize ? -f_best : f_best; grad=nothing, evals=n_evals)

        # Convergence
        fs[order[end]] - fs[order[1]] < tol_fun && break
        σ * maximum(D) < tol_x && break

        m_old = copy(m)
        m     = sum(w[i] .* xs_sel[i] for i in 1:μ)
        step  = (m .- m_old) ./ σ

        p_σ = (1 - c_σ) .* p_σ .+ sqrt(c_σ*(2-c_σ)*μ_eff) .* (invsqC * step)
        h_σ = norm(p_σ)/sqrt(1-(1-c_σ)^(2iter)) < (1.4+2/(N+1))*χ_N ? 1.0 : 0.0
        p_c = (1 - c_c) .* p_c .+ h_σ * sqrt(c_c*(2-c_c)*μ_eff) .* step

        δ_h  = (1 - h_σ) * c_c * (2 - c_c)
        Cmu  = sum(w[i] .* (zs_sel[i] * zs_sel[i]') for i in 1:μ)
        C    = (1-c_1-c_μ) .* C .+ c_1 .* (p_c*p_c' .+ δ_h .* C) .+ c_μ .* Cmu
        C    = (C + C') ./ 2

        if check_invariants
            ok, msg = check_cma_covariance(C)
            _assert_invariant(ok, msg, :cma_covariance,
                              (; iter=iter, N=N))
        end

        σ = max(σ * exp((c_σ/d_σ) * (norm(p_σ)/χ_N - 1.0)), 1e-14)
    end

    f_out = maximize ? -f_best : f_best
    converged = iter < max_iters && n_evals < max_evals
    stats = (evals=n_evals, iters=iter, converged=converged, history=history)
    return θ_best, f_out, stats
end

# ── PS-CMA-ES ─────────────────────────────────────────────────────────────────

"""
    pscmaes_optimize(f, θ0; max_evals, max_iters, lower, upper, popsize, seed,
                     n_islands, exchange_interval, sigma_init, maximize) → (θ_best, f_best, stats)

PS-CMA-ES: `n_islands` parallel CMA-ES instances sharing elite solutions.
Each island runs `exchange_interval` generations, then broadcasts its best to all others.
Minimises `f`; set `maximize=true` to maximise.
"""
function pscmaes_optimize(
    f                 :: Function,
    θ0                :: AbstractVector{<:Real};
    max_evals         :: Int     = 100_000,
    max_iters         :: Int     = 5_000,
    lower             :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper             :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    popsize           :: Int     = 0,
    seed              :: Union{Nothing,Int} = nothing,
    n_islands         :: Int     = 4,
    exchange_interval :: Int     = 10,
    sigma_init        :: Float64 = 0.3,
    maximize          :: Bool    = false,
    callback = nothing,
)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    N   = length(θ0)
    obj = maximize ? (x -> -f(x)) : f
    lb  = lower === nothing ? nothing : Float64.(lower)
    ub  = upper === nothing ? nothing : Float64.(upper)

    _clip(x) = lb === nothing ? x : clamp.(x, lb, ub)

    λ = popsize > 0 ? popsize : 4 + floor(Int, 3 * log(N))
    μ = max(1, λ ÷ 2)
    raw_w = [log(μ + 0.5) - log(i) for i in 1:μ]
    w     = raw_w ./ sum(raw_w)
    μ_eff = 1.0 / sum(w .^ 2)

    c_σ = (μ_eff + 2.0)/(N + μ_eff + 5.0)
    d_σ = 1.0 + 2.0*max(0.0, sqrt((μ_eff-1.0)/(N+1.0))-1.0) + c_σ
    c_c = (4.0 + μ_eff/N)/(N + 4.0 + 2.0*μ_eff/N)
    c_1 = 2.0/((N+1.3)^2 + μ_eff)
    c_μ = min(1.0-c_1, 2.0*(μ_eff-2.0+1.0/μ_eff)/((N+2.0)^2+μ_eff))
    χ_N = sqrt(N)*(1.0-1.0/(4N)+1.0/(21N^2))

    # Island state
    ms   = [_clip(Float64.(θ0) .+ (sigma_init/2) .* randn(rng, N)) for _ in 1:n_islands]
    ms[1] = _clip(Float64.(θ0))
    σs    = fill(sigma_init, n_islands)
    Cs    = [Matrix{Float64}(I, N, N) for _ in 1:n_islands]
    pσs   = [zeros(N) for _ in 1:n_islands]
    pcs   = [zeros(N) for _ in 1:n_islands]
    ibest_f = [obj(ms[k]) for k in 1:n_islands]
    ibest_x = copy.(ms)
    n_evals  = n_islands

    θ_best = copy(ms[argmin(ibest_f)])
    f_best = minimum(ibest_f)
    history = Float64[]

    n_evals_per_island = (max_evals - n_islands) ÷ n_islands
    iters_per_island   = max_iters ÷ n_islands
    total_evals = n_islands
    total_iters = 0

    # Per-island RNGs for thread safety
    base_seed_ps = seed === nothing ? rand(UInt32) : UInt32(seed) + UInt32(1000)
    island_rngs  = [MersenneTwister(base_seed_ps + UInt32(k)) for k in 1:n_islands]

    epoch = 0
    while total_iters < max_iters && total_evals < max_evals
        epoch += 1

        # Run all islands in parallel for exchange_interval generations each
        island_evals = zeros(Int, n_islands)
        Threads.@threads for island in 1:n_islands
            rng_i = island_rngs[island]
            m   = ms[island];  σ = σs[island];  C = Cs[island]
            p_σ = pσs[island]; p_c = pcs[island]
            ev  = 0

            for gen in 1:exchange_interval
                eig  = eigen(Symmetric(C))
                D_v  = sqrt.(max.(eig.values, 0.0))
                B    = eig.vectors
                sqC  = B * Diagonal(D_v) * B'
                invD = Diagonal(1.0 ./ max.(D_v, 1e-14))
                invsqC = B * invD * B'

                zs = [randn(rng_i, N) for _ in 1:λ]
                xs = [_clip(m .+ σ .* (sqC * z)) for z in zs]
                # Evaluate samples (nested @threads would over-subscribe; use serial here)
                fs = [obj(x) for x in xs]
                ev += λ

                ord    = sortperm(fs)
                xs_sel = xs[ord[1:μ]]
                zs_sel = zs[ord[1:μ]]

                if fs[ord[1]] < ibest_f[island]
                    ibest_f[island] = fs[ord[1]]
                    ibest_x[island] = copy(xs_sel[1])
                end

                m_old = copy(m)
                m     = sum(w[i] .* xs_sel[i] for i in 1:μ)
                step  = (m .- m_old) ./ σ

                p_σ = (1-c_σ) .* p_σ .+ sqrt(c_σ*(2-c_σ)*μ_eff) .* (invsqC * step)
                h_σ = norm(p_σ)/sqrt(1-(1-c_σ)^(2*(total_iters+gen))) < (1.4+2/(N+1))*χ_N ? 1.0 : 0.0
                p_c = (1-c_c) .* p_c .+ h_σ*sqrt(c_c*(2-c_c)*μ_eff) .* step

                δ_h = (1-h_σ)*c_c*(2-c_c)
                Cmu = sum(w[i] .* (zs_sel[i]*zs_sel[i]') for i in 1:μ)
                C   = (1-c_1-c_μ).*C .+ c_1.*(p_c*p_c' .+ δ_h.*C) .+ c_μ.*Cmu
                C   = (C+C')./2
                σ   = max(σ*exp((c_σ/d_σ)*(norm(p_σ)/χ_N-1.0)), 1e-14)
            end

            ms[island] = m; σs[island] = σ; Cs[island] = C
            pσs[island] = p_σ; pcs[island] = p_c
            island_evals[island] = ev
        end

        total_iters  += n_islands * exchange_interval
        total_evals  += sum(island_evals)

        # Update global best (serial after parallel block)
        for island in 1:n_islands
            if ibest_f[island] < f_best
                f_best = ibest_f[island]
                θ_best = copy(ibest_x[island])
            end
        end

        # ── Elite exchange: inject global best into each island's mean ────────
        best_isl = argmin(ibest_f)
        for island in 1:n_islands
            if island != best_isl
                ms[island] = _clip(ibest_x[best_isl] .+ σs[island] .* randn(rng, N))
            end
        end

        push!(history, f_best)
        isnothing(callback) || callback(total_iters, maximize ? -f_best : f_best; grad=nothing, evals=total_evals)
    end

    f_out = maximize ? -f_best : f_best
    stats = (evals=total_evals, iters=total_iters, converged=false, history=history)
    return θ_best, f_out, stats
end

# ---------------------------------------------------------------------------
# Sep-CMA-ES (diagonal covariance, O(λN) per generation)
# Folded from Optimization/DirectSearchMethods.jl
# Reference: Ros & Hansen, PPSN X (2008)
# ---------------------------------------------------------------------------

"""
    sep_cmaes_optimize(f, x0; max_iter, population_size, sigma_init, tol_fun, tol_x,
                       maximize, verbose, print_interval) → (x_best, f_best)

Separable CMA-ES with diagonal covariance matrix.  Suitable for large parameter
spaces (N ≥ 500) where full-covariance CMA-ES is prohibitive.

By default **maximises** `f`; set `maximize=false` to minimise.

# Arguments
- `f`      — objective `f(x::Vector{Float64}) -> Float64`
- `x0`     — initial parameter vector

# Keyword arguments
- `max_iter`        — maximum generations (default 2000)
- `population_size` — population λ; 0 → auto (default 0)
- `sigma_init`      — initial step size (default 0.3)
- `tol_fun`         — function-value convergence tolerance (default 1e-11)
- `tol_x`           — step-size convergence tolerance (default 1e-12)
- `maximize`        — `true` (default) to maximise; `false` to minimise
- `verbose`         — print progress (default false)
- `print_interval`  — logging frequency in generations (default 100)

# Returns
`(x_best, f_best)` — best parameters and their objective value.

# Example
```julia
f    = x -> -sum((x .- 1).^2)
x, v = sep_cmaes_optimize(f, zeros(500); sigma_init=0.5, maximize=true)
```
"""
function sep_cmaes_optimize(
    f              :: Function,
    x0             :: AbstractVector{<:Real};
    max_iter       :: Int     = 2000,
    population_size:: Int     = 0,
    sigma_init     :: Float64 = 0.3,
    tol_fun        :: Float64 = 1e-11,
    tol_x          :: Float64 = 1e-12,
    maximize       :: Bool    = true,
    verbose        :: Bool    = false,
    print_interval :: Int     = 100,
)
    N  = length(x0)
    λ  = population_size > 0 ? population_size : 4 + floor(Int, 3 * log(N))
    μ  = max(1, λ ÷ 2)

    w_raw = [log(μ + 0.5) - log(i) for i in 1:μ]
    w_rec = w_raw ./ sum(w_raw)
    μeff  = 1.0 / sum(w_rec .^ 2)

    c_σ = (μeff + 2.0) / (N + μeff + 5.0)
    d_σ = 1.0 + c_σ + 2.0 * max(0.0, sqrt((μeff - 1.0) / (N + 1.0)) - 1.0)
    c_c = (4.0 + μeff / N) / (N + 4.0 + 2.0 * μeff / N)
    c_1 = 2.0 / (N + 1.3)
    c_μ = min(1.0 - c_1,
              2.0 * (μeff - 2.0 + 1.0/μeff) / (N + 2.0 + μeff))
    χ_N = sqrt(N) * (1.0 - 1.0/(4N) + 1.0/(21N*N))

    m   = Float64.(copy(x0))
    σ   = sigma_init
    D   = ones(N)
    p_σ = zeros(N)
    p_c = zeros(N)

    obj    = maximize ? f : x -> -f(x)
    F_best = obj(m)
    x_best = copy(m)

    for gen in 1:max_iter
        Z    = [randn(N) for _ in 1:λ]
        X    = [m .+ σ .* D .* Z[k] for k in 1:λ]
        # Serial fitness evaluation (see comment above).
        fval = Vector{Float64}(undef, λ)
        for k in 1:λ
            fval[k] = obj(X[k])
        end
        idx  = sortperm(fval; rev=true)

        if fval[idx[1]] > F_best
            F_best = fval[idx[1]];  x_best = copy(X[idx[1]])
        end

        m_new = sum(w_rec[i] .* X[idx[i]] for i in 1:μ)
        δm    = (m_new - m) / σ

        p_σ = (1 - c_σ) .* p_σ .+ sqrt(c_σ*(2-c_σ)*μeff) .* (δm ./ D)
        h_σ = (norm(p_σ)/sqrt(1-(1-c_σ)^(2gen)) < (1.4+2.0/(N+1))*χ_N) ? 1.0 : 0.0
        p_c = (1 - c_c) .* p_c .+ h_σ*sqrt(c_c*(2-c_c)*μeff) .* δm

        wz2 = sum(w_rec[i] .* Z[idx[i]].^2 for i in 1:μ)
        D2  = (1-c_1-c_μ) .* D.^2 .+ c_1 .* p_c.^2 .+ c_μ .* wz2
        D   = sqrt.(max.(D2, 1e-20))

        σ = clamp(σ * exp(c_σ/d_σ * (norm(p_σ)/χ_N - 1.0)), 1e-14, Inf)
        m = m_new

        verbose && gen % print_interval == 0 &&
            @printf("[sep-CMA-ES] gen=%5d  F_best=%.8f  σ=%.3e\n", gen, F_best, σ)

        maximum(fval) - minimum(fval) < tol_fun && break
        σ * maximum(D) < tol_x && break
    end

    f_out = maximize ? F_best : -F_best
    return x_best, f_out
end
