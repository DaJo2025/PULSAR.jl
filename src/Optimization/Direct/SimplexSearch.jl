# ============================================================================
# Direct/SimplexSearch.jl — Nelder-Mead simplex (generic function interface)
# ============================================================================

"""
    nelder_mead_optimize(f, θ0; max_evals, max_iters, lower, upper, tol, step,
                         α, γ, ρ_c, σ_s) → (θ_best, f_best, stats)

Classical Nelder-Mead: reflect, expand, contract, shrink on n+1 simplex vertices.
Minimises f; stats = (evals, iters, converged).
"""
function nelder_mead_optimize(
    f         :: Function,
    θ0        :: AbstractVector{<:Real};
    max_evals :: Int     = 10_000,
    max_iters :: Int     = 5_000,
    lower     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper     :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    tol       :: Float64 = 1e-6,
    step      :: Float64 = 0.05,
    α         :: Float64 = 1.0,   # reflection
    γ         :: Float64 = 2.0,   # expansion
    ρ_c       :: Float64 = 0.5,   # contraction
    σ_s       :: Float64 = 0.5,   # shrinkage
    check_invariants :: Bool = false,
    callback = nothing,
)
    n   = length(θ0)
    lb  = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub  = upper === nothing ? fill( Inf, n) : Float64.(upper)
    clip(x) = clamp.(x, lb, ub)

    # ── Build initial simplex (n+1 vertices) ─────────────────────────────────
    simplex = Vector{Vector{Float64}}(undef, n + 1)
    simplex[1] = clip(Float64.(θ0))
    for i in 1:n
        v = copy(simplex[1])
        v[i] += abs(v[i]) > 1e-10 ? step * abs(v[i]) : step
        simplex[i+1] = clip(v)
    end

    fvals   = [f(x) for x in simplex]
    n_evals = n + 1
    history = Float64[minimum(fvals)]

    converged = false
    iter      = 0
    while iter < max_iters && n_evals < max_evals
        iter += 1

        # Sort: index 1 = best, n+1 = worst
        ord     = sortperm(fvals)
        simplex = simplex[ord]
        fvals   = fvals[ord]

        push!(history, fvals[1])

        diam = maximum(norm(simplex[i] .- simplex[1]) for i in 2:n+1)
        if diam < tol
            converged = true; break
        end

        x_bar = sum(simplex[1:n]) ./ n   # centroid of n best

        # Reflect
        x_r = x_bar .+ α .* (x_bar .- simplex[n+1])
        f_r = f(x_r); n_evals += 1

        if f_r < fvals[1]
            # Expand
            x_e = x_bar .+ γ .* (x_r .- x_bar)
            f_e = f(x_e); n_evals += 1
            simplex[n+1] = f_e < f_r ? x_e : x_r
            fvals[n+1]   = f_e < f_r ? f_e : f_r
        elseif f_r < fvals[n]
            simplex[n+1] = x_r; fvals[n+1] = f_r
        else
            # Contract
            x_c = x_bar .+ ρ_c .* (simplex[n+1] .- x_bar)
            f_c = f(x_c); n_evals += 1
            if f_c < fvals[n+1]
                simplex[n+1] = x_c; fvals[n+1] = f_c
            else
                # Shrink
                for i in 2:n+1
                    simplex[i] = simplex[1] .+ σ_s .* (simplex[i] .- simplex[1])
                    fvals[i]   = f(simplex[i]); n_evals += 1
                end
                if check_invariants
                    ok, msg = check_simplex_shape(simplex)
                    _assert_invariant(ok, msg, :simplex_shape,
                                      (; iter=iter, n_vertices=length(simplex)))
                end
            end
        end
        isnothing(callback) || callback(iter, fvals[1]; grad=nothing, evals=n_evals)
    end

    bi = argmin(fvals)
    stats = (evals=n_evals, iters=iter, converged=converged)
    return simplex[bi], fvals[bi], stats
end
