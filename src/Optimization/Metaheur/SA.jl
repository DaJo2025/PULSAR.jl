# ============================================================================
# Metaheur/SA.jl — Simulated Annealing variants
# ============================================================================

# ── Shared helper ─────────────────────────────────────────────────────────────
function _sa_clamp(x, lb, ub)
    if lb === nothing
        return x
    end
    return clamp.(x, lb, ub)
end

# ── SA ────────────────────────────────────────────────────────────────────────

"""
    sa_optimize(f, θ0; max_evals, max_iters, lower, upper, seed,
                T_init, T_min, cooling_rate, step_size, maximize) → (θ_best, f_best, stats)

Simulated Annealing with geometric cooling schedule.
Minimises `f`; set `maximize=true` to maximise.
"""
function sa_optimize(
    f           :: Function,
    θ0          :: AbstractVector{<:Real};
    max_evals   :: Int     = 50_000,
    max_iters   :: Int     = 50_000,
    lower       :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper       :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    seed        :: Union{Nothing,Int} = nothing,
    T_init      :: Float64 = 1.0,
    T_min       :: Float64 = 1e-8,
    cooling_rate:: Float64 = 0.999,
    step_size   :: Float64 = 0.1,
    maximize    :: Bool    = false,
    callback = nothing,
)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    n   = length(θ0)
    lb  = lower
    ub  = upper
    obj = maximize ? (x -> -f(x)) : f

    θ_cur  = _sa_clamp(Float64.(θ0), lb, ub)
    f_cur  = obj(θ_cur)
    θ_best = copy(θ_cur)
    f_best = f_cur
    T      = T_init
    n_evals = 1
    history = Float64[]

    iter = 0
    while iter < max_iters && n_evals < max_evals && T > T_min
        iter  += 1
        θ_new  = _sa_clamp(θ_cur .+ step_size .* randn(rng, n), lb, ub)
        f_new  = obj(θ_new)
        n_evals += 1

        ΔE = f_new - f_cur
        if ΔE < 0 || rand(rng) < exp(-ΔE / T)
            θ_cur = θ_new
            f_cur = f_new
            if f_cur < f_best
                f_best = f_cur
                θ_best = copy(θ_cur)
            end
        end

        T = T * cooling_rate
        push!(history, f_best)
        isnothing(callback) || callback(iter, maximize ? -f_best : f_best; grad=nothing, evals=n_evals)
    end

    f_out = maximize ? -f_best : f_best
    stats = (evals=n_evals, iters=iter, converged=T <= T_min, history=history)
    return θ_best, f_out, stats
end

# ── MCSA ──────────────────────────────────────────────────────────────────────

"""
    mcsa_optimize(f, θ0; max_evals, max_iters, lower, upper, seed,
                  n_chains, T_init, T_min, cooling_rate, step_size, maximize) → (θ_best, f_best, stats)

Monte-Carlo Simulated Annealing: runs `n_chains` independent SA chains; returns best.
Minimises `f`; set `maximize=true` to maximise.
"""
function mcsa_optimize(
    f           :: Function,
    θ0          :: AbstractVector{<:Real};
    max_evals   :: Int     = 50_000,
    max_iters   :: Int     = 10_000,
    lower       :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper       :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    seed        :: Union{Nothing,Int} = nothing,
    n_chains    :: Int     = 8,
    T_init      :: Float64 = 1.0,
    T_min       :: Float64 = 1e-8,
    cooling_rate:: Float64 = 0.999,
    step_size   :: Float64 = 0.1,
    maximize    :: Bool    = false,
    callback = nothing,
)
    base_seed = seed === nothing ? rand(UInt32) : UInt32(seed)
    # Seed the main rng for initial perturbations
    rng    = MersenneTwister(base_seed)
    n      = length(θ0)
    lb     = lower
    ub     = upper
    obj    = maximize ? (x -> -f(x)) : f
    budget = max_evals ÷ n_chains

    # Pre-generate per-chain starting points (serial, reproducible)
    starts = Vector{Vector{Float64}}(undef, n_chains)
    starts[1] = Float64.(θ0)
    for c in 2:n_chains
        starts[c] = _sa_clamp(Float64.(θ0) .+ step_size .* randn(rng, n), lb, ub)
    end

    # Per-chain results (pre-allocated for thread safety)
    chain_best_θ = [copy(starts[c]) for c in 1:n_chains]
    chain_best_f = fill(Inf, n_chains)
    chain_evals  = zeros(Int, n_chains)
    chain_iters  = zeros(Int, n_chains)

    # Run chains in parallel; each chain gets its own independent RNG
    Threads.@threads for c in 1:n_chains
        rng_c = MersenneTwister(base_seed + UInt32(c))
        θ_c = copy(starts[c])
        f_c = obj(θ_c)
        θb  = copy(θ_c)
        fb  = f_c
        T   = T_init
        ev  = 1
        it  = 0

        while it < max_iters && ev < budget && T > T_min
            it += 1
            θ_new = _sa_clamp(θ_c .+ step_size .* randn(rng_c, n), lb, ub)
            fn    = obj(θ_new)
            ev   += 1

            ΔE = fn - f_c
            if ΔE < 0 || rand(rng_c) < exp(-ΔE / T)
                θ_c = θ_new
                f_c = fn
                if f_c < fb; fb = f_c; θb = copy(θ_c); end
            end
            T *= cooling_rate
        end

        chain_best_θ[c] = θb
        chain_best_f[c] = fb
        chain_evals[c]  = ev
        chain_iters[c]  = it
    end

    # Merge results across chains
    best_c  = argmin(chain_best_f)
    θ_best  = chain_best_θ[best_c]
    f_best  = chain_best_f[best_c]
    total_evals = sum(chain_evals)
    total_iters = sum(chain_iters)

    f_out = maximize ? -f_best : f_best
    # history is the per-chain best at final iteration (parallel chains don't share a timeline)
    all_hist = chain_best_f
    stats = (evals=total_evals, iters=total_iters, converged=false,
             history=maximize ? [-x for x in all_hist] : all_hist)
    return θ_best, f_out, stats
end

# ── SSMC ──────────────────────────────────────────────────────────────────────

"""
    ssmc_optimize(f, θ0; max_evals, max_iters, lower, upper, seed,
                  n_walkers, beta_init, beta_max, beta_growth, step_size,
                  resample_thresh, maximize) → (θ_best, f_best, stats)

Substochastic Monte Carlo: walker ensemble with importance weights and
resampling. Inspired by Barzegar et al. (2021) SSMC algorithm.
Minimises `f`; set `maximize=true` to maximise.
"""
function ssmc_optimize(
    f               :: Function,
    θ0              :: AbstractVector{<:Real};
    max_evals       :: Int     = 50_000,
    max_iters       :: Int     = 1_000,
    lower           :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper           :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    seed            :: Union{Nothing,Int} = nothing,
    n_walkers       :: Int     = 50,
    beta_init       :: Float64 = 0.01,
    beta_max        :: Float64 = 10.0,
    beta_growth     :: Float64 = 1.05,
    step_size       :: Float64 = 0.1,
    resample_thresh :: Float64 = 0.5,   # ESS / n_walkers threshold
    maximize        :: Bool    = false,
    callback = nothing,
)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    n   = length(θ0)
    lb  = lower
    ub  = upper
    obj = maximize ? (x -> -f(x)) : f

    # Initialise walkers
    walkers = Vector{Vector{Float64}}(undef, n_walkers)
    walkers[1] = _sa_clamp(Float64.(θ0), lb, ub)
    for k in 2:n_walkers
        walkers[k] = _sa_clamp(Float64.(θ0) .+ step_size .* randn(rng, n), lb, ub)
    end
    fvals   = [obj(w) for w in walkers]
    weights = ones(Float64, n_walkers)

    n_evals = n_walkers
    history = Float64[]
    θ_best  = copy(walkers[argmin(fvals)])
    f_best  = minimum(fvals)
    β       = beta_init

    iter = 0
    while iter < max_iters && n_evals < max_evals
        iter += 1

        # ── Propose moves for each walker ────────────────────────────────────
        for k in 1:n_walkers
            θ_prop = _sa_clamp(walkers[k] .+ step_size .* randn(rng, n), lb, ub)
            f_prop = obj(θ_prop)
            n_evals += 1

            ΔE = f_prop - fvals[k]
            # Substochastic acceptance: accept if better; else accept with e^{-β·ΔE}
            if ΔE < 0 || rand(rng) < exp(-β * ΔE)
                walkers[k] = θ_prop
                fvals[k]   = f_prop
            end

            if fvals[k] < f_best
                f_best = fvals[k]
                θ_best = copy(walkers[k])
            end
            n_evals >= max_evals && break
        end

        # ── Importance weights: w_k ∝ exp(-β * f_k) ─────────────────────────
        f_min   = minimum(fvals)
        weights .= exp.(-β .* (fvals .- f_min))   # stabilise by shifting
        w_sum   = sum(weights)
        weights ./= w_sum

        # ── Resample when effective sample size drops ────────────────────────
        ess = 1.0 / sum(weights .^ 2)
        if ess / n_walkers < resample_thresh
            counts  = _systematic_resample(rng, weights, n_walkers)
            new_w   = Vector{Vector{Float64}}(undef, n_walkers)
            new_f   = Vector{Float64}(undef, n_walkers)
            idx = 1
            for (ki, cnt) in enumerate(counts)
                for _ in 1:cnt
                    new_w[idx] = copy(walkers[ki])
                    new_f[idx] = fvals[ki]
                    idx += 1
                end
            end
            walkers = new_w
            fvals   = new_f
            weights .= 1.0 / n_walkers
        end

        # ── Increase inverse temperature ─────────────────────────────────────
        β = min(β * beta_growth, beta_max)
        push!(history, f_best)
        isnothing(callback) || callback(iter, maximize ? -f_best : f_best; grad=nothing, evals=n_evals)
    end

    f_out = maximize ? -f_best : f_best
    stats = (evals=n_evals, iters=iter, converged=β >= beta_max, history=history)
    return θ_best, f_out, stats
end

# Systematic resampling — returns integer count vector summing to n
function _systematic_resample(rng, weights::Vector{Float64}, n::Int)::Vector{Int}
    counts = zeros(Int, length(weights))
    cumw   = cumsum(weights)
    u0     = rand(rng) / n
    j      = 1
    for i in 1:n
        u = u0 + (i - 1) / n
        while j < length(cumw) && cumw[j] < u
            j += 1
        end
        counts[j] += 1
    end
    return counts
end
