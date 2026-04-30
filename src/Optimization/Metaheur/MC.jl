# ============================================================================
# Metaheur/MC.jl — Monte Carlo Random Search and Grid Search
# ============================================================================

"""
    mc_random_search(f, θ0; max_evals, lower, upper, seed, maximize) → (θ_best, f_best, stats)

Uniform random search over the box [lower, upper]. Falls back to Gaussian
perturbation around θ0 if bounds are not provided.
Minimises `f`; set `maximize=true` to maximise.
"""
function mc_random_search(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 10_000,
    max_iters :: Int     = max_evals,   # alias; evals is the binding limit
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    seed      :: Union{Nothing,Int} = nothing,
    maximize  :: Bool    = false,
    callback = nothing,
)
    base_seed = seed === nothing ? rand(UInt32) : UInt32(seed)
    n   = length(θ0)
    lb  = lower === nothing ? nothing : Float64.(lower)
    ub  = upper === nothing ? nothing : Float64.(upper)
    obj = maximize ? (x -> -f(x)) : f

    bounded = lb !== nothing && ub !== nothing && all(isfinite, lb) && all(isfinite, ub)
    σ_fb    = 1.0   # fallback spread

    # Evaluate initial point
    θ_best  = Float64.(θ0)
    f_best  = obj(θ_best)
    n_evals = 1

    # Batch size for parallel evaluation (amortise thread overhead)
    batch = max(Threads.nthreads(), min(256, max_evals ÷ 4))
    history = Float64[f_best]

    while n_evals < max_evals
        nb = min(batch, max_evals - n_evals)
        # Per-thread RNGs for reproducible parallel sampling
        rngs = [MersenneTwister(base_seed + UInt32(n_evals + t)) for t in 0:nb-1]
        cands = Vector{Vector{Float64}}(undef, nb)
        for t in 1:nb
            cands[t] = if bounded
                lb .+ rand(rngs[t], n) .* (ub .- lb)
            else
                θ = Float64.(θ0) .+ σ_fb .* randn(rngs[t], n)
                lb === nothing ? θ : clamp.(θ, lb, ub)
            end
        end
        # Parallel fitness evaluation of this batch
        fcs = Vector{Float64}(undef, nb)
        Threads.@threads for t in 1:nb
            fcs[t] = obj(cands[t])
        end
        n_evals += nb
        # Serial reduction to find batch best
        for t in 1:nb
            if fcs[t] < f_best
                f_best = fcs[t]
                θ_best = copy(cands[t])
            end
            push!(history, f_best)
        end
        isnothing(callback) || callback(n_evals, maximize ? -f_best : f_best; grad=nothing, evals=n_evals)
    end

    f_out = maximize ? -f_best : f_best
    stats = (evals=n_evals, iters=n_evals, converged=false, history=history)
    return θ_best, f_out, stats
end

# ── Grid Search ───────────────────────────────────────────────────────────────

"""
    grid_search(f, ranges; seed, maximize) → (θ_best, f_best, stats)

Exhaustive grid search. `ranges` is a vector of AbstractRange (one per dimension).
For large grids use `max_evals` to cap evaluations (random subset is drawn).
Minimises `f`; set `maximize=true` to maximise.
"""
function grid_search(
    f        :: Function,
    ranges   :: AbstractVector;
    max_evals:: Int  = typemax(Int),
    seed     :: Union{Nothing,Int} = nothing,
    maximize :: Bool = false,
    callback = nothing,
)
    obj   = maximize ? (x -> -f(x)) : f
    grids = [collect(r) for r in ranges]
    ndim  = length(grids)

    # Build grid as iterator using Cartesian product
    sizes  = length.(grids)
    ntotal = prod(sizes)

    rng   = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    order = ntotal <= max_evals ? (1:ntotal) : randperm(rng, ntotal)[1:max_evals]

    n_pts   = length(order)
    θ_best  = [grids[d][1] for d in 1:ndim]
    f_best  = Inf
    n_evals = n_pts

    # Convert flat linear index → per-dimension indices
    function linear_to_idx(k::Int)
        idx = Vector{Int}(undef, ndim)
        rem = k - 1
        for d in ndim:-1:1
            idx[d] = rem % sizes[d] + 1
            rem    = rem ÷ sizes[d]
        end
        return idx
    end

    # Build all grid points and evaluate in parallel (embarrassingly parallel)
    pts   = [begin idx = linear_to_idx(order[i]); [Float64(grids[d][idx[d]]) for d in 1:ndim]; end
             for i in 1:n_pts]
    fvals = Vector{Float64}(undef, n_pts)
    Threads.@threads for i in 1:n_pts
        fvals[i] = obj(pts[i])
    end

    # Serial reduction to collect best
    history = Float64[]
    for i in 1:n_pts
        if fvals[i] < f_best
            f_best = fvals[i]
            θ_best = copy(pts[i])
        end
        push!(history, f_best)
        isnothing(callback) || callback(i, maximize ? -f_best : f_best; grad=nothing, evals=i)
    end

    f_out = maximize ? -f_best : f_best
    stats = (evals=n_evals, iters=n_evals, converged=ntotal <= max_evals, history=history)
    return θ_best, f_out, stats
end
