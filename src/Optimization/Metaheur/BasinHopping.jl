# ============================================================================
# Metaheur/BasinHopping.jl — Basin Hopping with Metropolis acceptance
#
# Wales, D. J. & Doye, J. P. K. (1997) J. Phys. Chem. A 101, 5111–5116.
# ============================================================================

"""
    basin_hopping_optimize(f, grad!, θ0; kwargs...) → (θ_best, f_best, stats)
    basin_hopping_optimize(f, θ0;         kwargs...) → (θ_best, f_best, stats)

Basin Hopping metaheuristic for global optimisation.

Each hop consists of three steps:
  1. **Perturb** the current point with Gaussian noise (σ scales with √T).
  2. **Local optimise** the perturbed point to the nearest basin minimum.
     Uses L-BFGS-B when `grad!` is supplied, Nelder-Mead otherwise.
  3. **Accept or reject** via the Metropolis criterion:
     always accept if F_new ≥ F_current; accept with probability
     `exp((F_new − F_current) / T)` if F_new < F_current.

The global best across *all* trials is tracked independently of the
acceptance chain and is returned as `θ_best`.

Temperature is cooled geometrically: `T ← T × cool_rate` after each hop.
`cool_rate` is computed automatically from `T_init`, `T_final`, and `n_hops`
unless provided explicitly.

# Arguments
- `f`          : objective function `f(θ) → scalar` (minimised by default).
- `grad!`      : in-place gradient `grad!(g, θ)`.  Optional; if omitted,
                 Nelder-Mead is used as the local solver.
- `θ0`         : initial parameter vector.

# Keyword arguments
| Keyword          | Default   | Description                                      |
|------------------|-----------|--------------------------------------------------|
| `n_hops`         | `50`      | Number of basin-hopping steps.                   |
| `T_init`         | `0.1`     | Initial Metropolis temperature (fidelity units). |
| `T_final`        | `1e-4`    | Final temperature (sets `cool_rate` if not set). |
| `cool_rate`      | `nothing` | Override geometric cooling rate per hop.         |
| `perturb_sigma`  | `0.2`     | Perturbation σ at T = T_init; scales as √(T/T_init). |
| `local_iters`    | `200`     | Max iterations for each local optimisation.      |
| `local_tol`      | `1e-7`    | Convergence tolerance for local optimiser.       |
| `local_memory`   | `15`      | L-BFGS-B history length (ignored for Nelder-Mead). |
| `lower`          | `nothing` | Lower box bounds (length n).                     |
| `upper`          | `nothing` | Upper box bounds (length n).                     |
| `seed`           | `nothing` | RNG seed for reproducibility.                    |
| `maximize`       | `false`   | Set `true` to maximise `f`.                      |
| `verbose`        | `false`   | Print hop-by-hop progress.                       |
| `callback`       | `nothing` | `callback(hop, f_best; ...)` called each hop.   |

# Returns
- `θ_best`  : best parameter vector found.
- `f_best`  : corresponding objective value (in original sense, i.e. maximised if `maximize=true`).
- `stats`   : NamedTuple `(evals, iters, converged, history, n_accepted, n_metro)`.
  - `history`    : `f_best` (global best) after each hop.
  - `n_accepted` : hops accepted because F_new ≥ F_current.
  - `n_metro`    : hops accepted via Metropolis (F_new < F_current).

# Example
```julia
# With gradient (L-BFGS-B local solver):
θ_opt, f_opt, stats = basin_hopping_optimize(
    f, grad!, θ0;
    n_hops=40, T_init=0.05, T_final=5e-4,
    perturb_sigma=0.25, local_iters=150,
    lower=lb, upper=ub, seed=42, maximize=true,
)

# Without gradient (Nelder-Mead local solver):
θ_opt, f_opt, stats = basin_hopping_optimize(
    f, θ0;
    n_hops=30, T_init=0.1, maximize=true,
)
```
"""
function basin_hopping_optimize(
    f               :: Function,
    grad!           :: Union{Function, Nothing},
    θ0              :: AbstractVector{<:Real};
    n_hops          :: Int     = 50,
    T_init          :: Float64 = 0.1,
    T_final         :: Float64 = 1e-4,
    cool_rate       :: Union{Float64, Nothing} = nothing,
    perturb_sigma   :: Float64 = 0.2,
    local_iters     :: Int     = 200,
    local_tol       :: Float64 = 1e-7,
    local_memory    :: Int     = 15,
    lower           :: Union{Nothing, AbstractVector{<:Real}} = nothing,
    upper           :: Union{Nothing, AbstractVector{<:Real}} = nothing,
    seed            :: Union{Nothing, Int} = nothing,
    maximize          :: Bool    = false,
    verbose           :: Bool    = false,
    skip_initial_opt  :: Bool    = false,
    callback          = nothing,
)
    base_seed = seed === nothing ? rand(UInt32) : UInt32(seed)
    rng  = MersenneTwister(base_seed)
    n    = length(θ0)
    lb   = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub   = upper === nothing ? fill( Inf, n) : Float64.(upper)

    # Internally always minimise
    obj      = maximize ? (x -> -f(x)) : f
    obj_grad = if grad! === nothing
        nothing
    else
        maximize ? ((g, x) -> (grad!(g, x); g .*= -1; g)) : grad!
    end

    # Geometric cooling rate
    α = if cool_rate !== nothing
        Float64(cool_rate)
    elseif n_hops > 1
        (T_final / T_init)^(1.0 / (n_hops - 1))
    else
        1.0
    end

    # ── Local optimiser wrapper ───────────────────────────────────────────────
    function _local_opt(θ_start::Vector{Float64})
        if obj_grad !== nothing
            θ_loc, f_loc, _ = lbfgsb_optimize(
                obj, obj_grad, θ_start;
                lower   = lower,
                upper   = upper,
                memory  = local_memory,
                max_iter= local_iters,
                tol     = local_tol,
                verbose = false,
            )
            return θ_loc, f_loc
        else
            θ_loc, f_loc, _ = nelder_mead_optimize(
                obj, θ_start;
                max_iters = local_iters,
                lower     = lower,
                upper     = upper,
            )
            return θ_loc, f_loc
        end
    end

    # ── Single-chain hopping function (used both serially and in parallel) ───
    function _run_chain(chain_seed::UInt32, θ_start::Vector{Float64})
        rng_c    = MersenneTwister(chain_seed)
        θ_c      = clamp.(θ_start, lb, ub)
        if skip_initial_opt
            f_c   = obj(θ_c)
            n_ev  = 1
        else
            θ_c, f_c = _local_opt(θ_c)
            n_ev  = local_iters
        end
        θb   = copy(θ_c);  fb = f_c
        T_c  = T_init
        nacc = 0;  nmet = 0

        for hop in 1:n_hops
            σ_hop   = perturb_sigma * sqrt(T_c / T_init)
            θ_trial = clamp.(θ_c .+ σ_hop .* randn(rng_c, n), lb, ub)
            θ_trial, f_trial = _local_opt(θ_trial)
            n_ev   += local_iters

            ΔE = f_trial - f_c
            if ΔE <= 0.0
                θ_c = θ_trial;  f_c = f_trial;  nacc += 1
            elseif rand(rng_c) < exp(-ΔE / T_c)
                θ_c = θ_trial;  f_c = f_trial;  nmet += 1
            end
            if f_trial < fb;  fb = f_trial;  θb = copy(θ_trial);  end
            T_c *= α
        end
        return θb, fb, n_ev, nacc, nmet
    end

    # ── Determine number of parallel chains ──────────────────────────────────
    nth      = Threads.nthreads()
    n_chains = nth > 1 ? nth : 1
    # Perturb starting points for chains 2..n_chains
    starts = [clamp.(Float64.(θ0), lb, ub)]
    for c in 2:n_chains
        rng_init = MersenneTwister(base_seed + UInt32(c))
        push!(starts, clamp.(Float64.(θ0) .+ perturb_sigma .* randn(rng_init, n), lb, ub))
    end

    # Pre-allocate results arrays for thread safety
    chain_θ    = [copy(starts[c]) for c in 1:n_chains]
    chain_f    = fill(Inf, n_chains)
    chain_ev   = zeros(Int, n_chains)
    chain_acc  = zeros(Int, n_chains)
    chain_met  = zeros(Int, n_chains)

    if verbose
        @printf("  Basin Hopping: n=%d  n_hops=%d  T_init=%.4f  T_final=%.5f  α=%.5f  chains=%d\n",
                n, n_hops, T_init, T_final, α, n_chains)
    end

    # Run chains in parallel
    Threads.@threads for c in 1:n_chains
        θb, fb, nev, nacc, nmet = _run_chain(base_seed + UInt32(c - 1), starts[c])
        chain_θ[c]   = θb
        chain_f[c]   = fb
        chain_ev[c]  = nev
        chain_acc[c] = nacc
        chain_met[c] = nmet
    end

    best_c   = argmin(chain_f)
    θ_best   = chain_θ[best_c]
    f_best   = chain_f[best_c]
    n_evals  = sum(chain_ev)
    n_accept = sum(chain_acc)
    n_metro  = sum(chain_met)
    history  = chain_f   # final best per chain

    f_out = maximize ? -f_best : f_best
    stats = (
        evals      = n_evals,
        iters      = n_hops * n_chains,
        converged  = false,
        history    = maximize ? [-x for x in history] : history,
        n_accepted = n_accept,
        n_metro    = n_metro,
    )
    return θ_best, f_out, stats
end

# ── Convenience method without grad! ─────────────────────────────────────────

"""
    basin_hopping_optimize(f, θ0; kwargs...) → (θ_best, f_best, stats)

Basin Hopping with Nelder-Mead as the local solver (no gradient required).
See the main method for full documentation.
"""
basin_hopping_optimize(
    f  :: Function,
    θ0 :: AbstractVector{<:Real};
    kwargs...,
) = basin_hopping_optimize(f, nothing, θ0; kwargs...)
