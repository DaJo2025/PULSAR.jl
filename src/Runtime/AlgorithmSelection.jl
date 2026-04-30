# ============================================================
# Pulsar.jl — Algorithm Selection and Auto-Optimizer
# Pulse Design Library for Spin Control Algorithms and Rollout
# ============================================================
#
# Provides an automatic algorithm recommendation system based on
# problem characteristics (system size, n_controls, constraints,
# robustness requirements) and an auto_optimize wrapper that picks
# and runs the best algorithm without user intervention.
# ============================================================

using LinearAlgebra, Printf

# ──────────────────────────────────────────────────────────────
# Recommendation type
# ──────────────────────────────────────────────────────────────

"""
    AlgorithmRecommendation

Stores the result of recommend_optimizer.

# Fields
- `method`: Symbol identifying the recommended algorithm
  (:grape, :lbfgs, :bfgs, :newton, :cmaes, :nelder_mead, :trust_region,
   :constrained_grape, :robust_grape)
- `config`: Pre-configured algorithm config object
- `reasoning`: Human-readable explanation of the choice
- `expected_convergence_rate`: "fast", "moderate", or "slow"
- `expected_iterations`: Rough estimate of iterations needed
- `warnings`: Any caveats about the recommendation
"""
struct AlgorithmRecommendation
    method::Symbol
    config::Any
    reasoning::String
    expected_convergence_rate::String
    expected_iterations::Int
    warnings::Vector{String}
end

# ──────────────────────────────────────────────────────────────
# Main recommendation logic
# ──────────────────────────────────────────────────────────────

"""
    recommend_optimizer(system, target, n_timesteps; kwargs...) -> AlgorithmRecommendation

Recommend the best optimization algorithm for a given quantum control problem.

# Decision Logic

| Condition                          | Recommended Method        |
|------------------------------------|---------------------------|
| has_constraints                    | constrained (GRAPE-based) |
| needs_robustness                   | robust_grape              |
| !gradient_reliable, n_params < 30  | nelder_mead               |
| !gradient_reliable, n_params ≤ 200 | cmaes                     |
| n_params ≤ 200                     | lbfgs                     |
| n_params ≤ 1000                    | lbfgs (memory=20)         |
| n_params > 1000                    | grape (parallelises well) |
| time_budget very tight             | grape                     |

# Arguments
- `system`: Quantum system (AbstractQuantumSystem)
- `target`: Optimization target (QuantumTarget)
- `n_timesteps`: Number of time steps in control sequence

# Keyword Arguments
- `has_constraints`: Problem has amplitude/energy/power constraints (default: false)
- `needs_robustness`: Must be robust to parameter uncertainty (default: false)
- `time_budget_seconds`: Wall-clock budget; Inf means no restriction
- `gradient_reliable`: False for non-smooth or noisy problems (default: true)
- `gpu_available`: GPU device detected (default: false)
"""
function recommend_optimizer(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    n_timesteps::Int;
    has_constraints::Bool          = false,
    needs_robustness::Bool         = false,
    time_budget_seconds::Float64   = Inf,
    gradient_reliable::Bool        = true,
    gpu_available::Bool            = false,
)::AlgorithmRecommendation

    n_params  = system.n_controls * n_timesteps
    warnings  = String[]

    # ── Constrained problems ───────────────────────────────────
    if has_constraints
        cfg = ConstrainedConfig(
            base_method        = "grape",
            constraint_method  = "augmented_lagrangian",
            max_iter           = 1000,
            verbose            = false,
        )
        return AlgorithmRecommendation(
            :constrained_grape, cfg,
            "Problem has constraints: using GRAPE with augmented-Lagrangian constraint handling.",
            "moderate", 800, warnings,
        )
    end

    # ── Robust problems ────────────────────────────────────────
    if needs_robustness
        cfg = RobustConfig(
            uncertainty_type      = "parametric",
            uncertainty_magnitude = 0.05,
            robustness_measure    = "mean",
            n_samples             = 20,
            base_method           = "grape",
            verbose               = false,
        )
        return AlgorithmRecommendation(
            :robust_grape, cfg,
            "Robustness requested: using sample-averaged robust GRAPE.",
            "moderate", 500, warnings,
        )
    end

    # ── Derivative-free methods ────────────────────────────────
    if !gradient_reliable
        if n_params <= 30
            cfg = NelderMeadConfig(max_iter=5000, verbose=false)
            return AlgorithmRecommendation(
                :nelder_mead, cfg,
                "Gradient unreliable and n_params=$(n_params) ≤ 30: Nelder-Mead simplex.",
                "slow", 3000, warnings,
            )
        else
            pop = max(20, 4 + floor(Int, 3*log(n_params)))
            cfg = CMAESConfig(max_iter=1000, population_size=pop, verbose=false)
            push!(warnings, "Derivative-free CMA-ES is slower than gradient methods; gradient-based preferred if gradient becomes reliable.")
            return AlgorithmRecommendation(
                :cmaes, cfg,
                "Gradient unreliable and n_params=$(n_params): CMA-ES covariance adaptation.",
                "slow", 800, warnings,
            )
        end
    end

    # ── Very tight time budget ─────────────────────────────────
    if isfinite(time_budget_seconds) && time_budget_seconds < 5.0
        cfg = GRAPEConfig(max_iter=200, adapt_step_size=true, verbose=false)
        push!(warnings, "Tight time budget: using GRAPE with limited iterations.")
        return AlgorithmRecommendation(
            :grape, cfg,
            "Tight time budget ($(time_budget_seconds)s): GRAPE with 200 iterations for predictable cost.",
            "moderate", 200, warnings,
        )
    end

    # ── Gradient-based selection by problem size ───────────────
    if n_params <= 200
        cfg = LBFGSConfig(max_iter=300, memory_size=10, verbose=false)
        return AlgorithmRecommendation(
            :lbfgs, cfg,
            "n_params=$(n_params) ≤ 200: L-BFGS with superlinear convergence.",
            "fast", 100, warnings,
        )
    elseif n_params <= 1000
        cfg = LBFGSConfig(max_iter=500, memory_size=20, verbose=false)
        return AlgorithmRecommendation(
            :lbfgs, cfg,
            "n_params=$(n_params) ≤ 1000: L-BFGS(m=20) balances memory and convergence speed.",
            "fast", 200, warnings,
        )
    else
        # Large problems — GRAPE parallelises best
        if gpu_available
            push!(warnings, "GPU available: consider setting backend=:cuda for large problems.")
        end
        cfg = GRAPEConfig(max_iter=1000, adapt_step_size=true, verbose=false)
        return AlgorithmRecommendation(
            :grape, cfg,
            "n_params=$(n_params) > 1000: GRAPE scales well and parallelises over timesteps.",
            "moderate", 800, warnings,
        )
    end
end

# ──────────────────────────────────────────────────────────────
# Human-readable description
# ──────────────────────────────────────────────────────────────

"""
    describe_recommendation(rec) -> String

Return a formatted, human-readable description of an AlgorithmRecommendation.
"""
function describe_recommendation(rec::AlgorithmRecommendation)::String
    io = IOBuffer()
    println(io, "━"^60)
    println(io, "Pulsar Algorithm Recommendation")
    println(io, "━"^60)
    println(io, "  Method  : $(rec.method)")
    println(io, "  Speed   : $(rec.expected_convergence_rate)")
    println(io, "  Est. iter: $(rec.expected_iterations)")
    println(io, "  Reason  : $(rec.reasoning)")
    if !isempty(rec.warnings)
        println(io, "\n  ⚠  Warnings:")
        for w in rec.warnings
            println(io, "     • $w")
        end
    end
    println(io, "━"^60)
    return String(take!(io))
end

# ──────────────────────────────────────────────────────────────
# Auto-optimizer
# ──────────────────────────────────────────────────────────────

"""
    auto_optimize(system, target, controls_init; verbose, kwargs...) -> OptimizationResult

Automatically select and run the best optimization algorithm.

This is the highest-level entry point in Pulsar.  It calls
`recommend_optimizer`, optionally prints the recommendation, then
dispatches to the appropriate optimizer and returns its result.

# Arguments
- `system`: Quantum system
- `target`: Optimization target
- `controls_init`: Initial control sequence

# Keyword Arguments
- `verbose`: Print recommendation before running (default: true)
- All keyword arguments of `recommend_optimizer` are forwarded.

# Example
```julia
result = auto_optimize(system, target, controls_init; verbose=true)
println("Achieved fidelity: ", result.fidelity)
```
"""
function auto_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    verbose::Bool = true,
    kwargs...
)::OptimizationResult

    rec = recommend_optimizer(system, target, controls_init.n_timesteps; kwargs...)

    if verbose
        print(describe_recommendation(rec))
    end

    # Dispatch to the recommended method
    if rec.method == :grape
        cfg = rec.config isa GRAPEConfig ? rec.config : GRAPEConfig()
        return grape_optimize(system, target, controls_init; config=cfg)

    elseif rec.method == :lbfgs
        cfg = rec.config isa LBFGSConfig ? rec.config : LBFGSConfig()
        return lbfgs_optimize(system, target, controls_init; config=cfg)

    elseif rec.method == :bfgs
        cfg = rec.config isa BFGSConfig ? rec.config : BFGSConfig()
        return bfgs_optimize(system, target, controls_init; config=cfg)

    elseif rec.method == :cmaes
        cfg = rec.config isa CMAESConfig ? rec.config : CMAESConfig()
        return cmaes_optimize(system, target, controls_init; config=cfg)

    elseif rec.method == :nelder_mead
        cfg = rec.config isa NelderMeadConfig ? rec.config : NelderMeadConfig()
        return nelder_mead_optimize(system, target, controls_init; config=cfg)

    elseif rec.method == :trust_region
        cfg = rec.config isa TrustRegionConfig ? rec.config : TrustRegionConfig()
        return trust_region_optimize(system, target, controls_init; config=cfg)

    elseif rec.method == :constrained_grape
        cfg = rec.config isa ConstrainedConfig ? rec.config : ConstrainedConfig()
        return constrained_optimize(system, target, controls_init, AbstractConstraint[]; config=cfg)

    elseif rec.method == :robust_grape
        cfg = rec.config isa RobustConfig ? rec.config : RobustConfig()
        return robust_optimize(system, target, controls_init; config=cfg)

    else
        @warn "Unknown method $(rec.method), falling back to GRAPE"
        return grape_optimize(system, target, controls_init)
    end
end

# ──────────────────────────────────────────────────────────────
# Per-iteration time estimator
# ──────────────────────────────────────────────────────────────

"""
    estimate_iteration_time(system, controls, method) -> Float64

Estimate the wall-clock time per iteration for a given method on the
current hardware by running a short 5-iteration benchmark.

Returns time in seconds.
"""
function estimate_iteration_time(
    system::AbstractQuantumSystem,
    controls::ControlSequence,
    method::Symbol,
)::Float64
    target_dummy = state_target(zeros(ComplexF64, system.dim) |> v -> begin v[1]=1; v end)

    n_bench = 5
    t_start = time()
    try
        if method == :grape
            grape_optimize(system, target_dummy, controls;
                           config=GRAPEConfig(max_iter=n_bench, verbose=false))
        elseif method == :lbfgs
            lbfgs_optimize(system, target_dummy, controls;
                           config=LBFGSConfig(max_iter=n_bench, verbose=false))
        end
    catch
        return NaN
    end
    return (time() - t_start) / n_bench
end
