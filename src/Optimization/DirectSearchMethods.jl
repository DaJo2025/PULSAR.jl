"""
    DirectSearchMethods.jl

Derivative-free (direct-search) optimization methods for Pulsar.

Implements three derivative-free methods that do not require gradient information:

1. **Nelder-Mead** — Simplex-based method.  Constructs a (n+1)-vertex simplex
   in parameter space and iteratively deforms it toward the optimum via
   reflection, expansion, contraction, and shrinkage operations.  Suitable for
   smooth problems with n ≤ 50.

2. **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy.  A stochastic,
   population-based method that adapts both step size and the full covariance
   structure of a Gaussian search distribution.  State-of-the-art for
   derivative-free optimization on n ≤ 1000.

3. **PSO** — Particle Swarm Optimization.  Maintains a swarm of candidate
   solutions, each with a velocity; particles are attracted to their own best
   known position and to the global best.

All methods wrap their controls in flat `Vector{Float64}` internally and
return the standard `OptimizationResult` type defined in GRAPE.jl.

References:
  Nelder & Mead, Comput. J. 7 (1965) 308–313.
  Hansen & Ostermeier, Evol. Comput. 9 (2001) 159–195.
  Kennedy & Eberhart, Proc. ICNN (1995) 1942–1948.
"""

# ============================================================================
# Configuration types
# ============================================================================

"""
    NelderMeadConfig

Configuration for the Nelder-Mead simplex optimizer.

# Fields
- `max_iter::Int` — maximum simplex iterations (default 5000)
- `tol::Float64` — simplex diameter convergence threshold (default 1e-6)
- `alpha::Float64` — reflection coefficient α (default 1.0)
- `beta::Float64` — contraction coefficient β (default 0.5)
- `gamma::Float64` — expansion coefficient γ (default 2.0)
- `delta::Float64` — shrinkage coefficient δ (default 0.5)
- `verbose::Bool` — print progress (default false)
- `print_interval::Int` — logging frequency in iterations (default 500)

# Example
```julia
cfg = NelderMeadConfig(max_iter=10000, tol=1e-7)
```
"""
struct NelderMeadConfig
    max_iter::Int
    tol::Float64
    alpha::Float64
    beta::Float64
    gamma::Float64
    delta::Float64
    verbose::Bool
    print_interval::Int
end

"""
    NelderMeadConfig(; kwargs...) -> NelderMeadConfig

Construct a `NelderMeadConfig` with keyword arguments and default values.
"""
function NelderMeadConfig(;
    max_iter::Int                            = 5000,
    tol::Float64                             = 1e-6,
    convergence_tol::Union{Nothing,Float64}  = nothing,   # legacy alias for `tol`
    alpha::Float64      = 1.0,
    beta::Float64       = 0.5,
    gamma::Float64      = 2.0,
    delta::Float64      = 0.5,
    verbose::Bool       = false,
    print_interval::Int = 500
)::NelderMeadConfig
    if convergence_tol !== nothing
        tol = convergence_tol
    end
    max_iter > 0         || throw(ArgumentError("max_iter must be positive"))
    tol > 0              || throw(ArgumentError("tol must be positive"))
    alpha > 0            || throw(ArgumentError("alpha must be positive"))
    0 < beta  < 1        || throw(ArgumentError("beta must be in (0,1)"))
    gamma > 1            || throw(ArgumentError("gamma must be > 1"))
    0 < delta < 1        || throw(ArgumentError("delta must be in (0,1)"))

    return NelderMeadConfig(max_iter, tol, alpha, beta, gamma, delta,
                             verbose, print_interval)
end

# ----------------------------------------------------------------------------

"""
    CMAESConfig

Configuration for the CMA-ES optimizer.

# Fields
- `max_iter::Int` — maximum generations (default 1000)
- `population_size::Int` — number of offspring λ per generation; 0 → use default
  λ = 4 + floor(3 * log(n)) (default 0)
- `sigma_init::Float64` — initial step size σ₀ (default 0.3)
- `tol_fun::Float64` — convergence tolerance on function value range (default 1e-10)
- `tol_x::Float64` — convergence tolerance on parameter range (default 1e-12)
- `verbose::Bool` — print progress (default false)
- `print_interval::Int` — logging frequency in generations (default 100)

# Example
```julia
cfg = CMAESConfig(max_iter=2000, sigma_init=0.5, population_size=20)
```
"""
struct CMAESConfig
    max_iter::Int
    population_size::Int
    sigma_init::Float64
    tol_fun::Float64
    tol_x::Float64
    verbose::Bool
    print_interval::Int
end

"""
    CMAESConfig(; kwargs...) -> CMAESConfig

Construct a `CMAESConfig` with keyword arguments and default values.
"""
function CMAESConfig(;
    max_iter::Int                              = 1000,
    population_size::Int                       = 0,
    sigma_init::Float64                        = 0.3,
    initial_sigma::Union{Nothing,Float64}      = nothing,   # legacy alias for sigma_init
    tol_fun::Float64                           = 1e-10,
    tol_x::Float64                             = 1e-12,
    verbose::Bool                              = false,
    print_interval::Int                        = 100,
    noise_handling::Bool                       = false      # accepted, currently ignored
)::CMAESConfig
    if initial_sigma !== nothing
        sigma_init = initial_sigma
    end
    max_iter > 0       || throw(ArgumentError("max_iter must be positive"))
    population_size >= 0 || throw(ArgumentError("population_size must be ≥ 0"))
    sigma_init > 0     || throw(ArgumentError("sigma_init must be positive"))

    return CMAESConfig(max_iter, population_size, sigma_init,
                       tol_fun, tol_x, verbose, print_interval)
end

# ----------------------------------------------------------------------------

"""
    PSOConfig

Configuration for the Particle Swarm Optimizer.

# Fields
- `max_iter::Int` — maximum iterations (default 1000)
- `n_particles::Int` — swarm size; 0 → use default = max(10, 2 + floor(n/2)) (default 0)
- `w::Float64` — inertia weight (default 0.7)
- `c1::Float64` — cognitive (personal best) acceleration coefficient (default 1.5)
- `c2::Float64` — social (global best) acceleration coefficient (default 1.5)
- `v_max::Float64` — maximum velocity magnitude (default 2.0)
- `verbose::Bool` — print progress (default false)
- `print_interval::Int` — logging frequency (default 100)

# Example
```julia
cfg = PSOConfig(max_iter=500, n_particles=30, w=0.729, c1=1.494, c2=1.494)
```
"""
struct PSOConfig
    max_iter::Int
    n_particles::Int
    w::Float64
    c1::Float64
    c2::Float64
    v_max::Float64
    verbose::Bool
    print_interval::Int
    seed::Union{Nothing,Int}
end

"""
    PSOConfig(; kwargs...) -> PSOConfig

Construct a `PSOConfig` with keyword arguments and default values.
"""
function PSOConfig(;
    max_iter::Int                       = 1000,
    n_particles::Int                    = 0,
    w::Float64                          = 0.7,
    c1::Float64                         = 1.5,
    c2::Float64                         = 1.5,
    v_max::Float64                      = 2.0,
    verbose::Bool                       = false,
    print_interval::Int                 = 100,
    seed::Union{Nothing,Integer}        = nothing
)::PSOConfig
    max_iter > 0     || throw(ArgumentError("max_iter must be positive"))
    n_particles >= 0  || throw(ArgumentError("n_particles must be ≥ 0"))
    0 <= w <= 1       || throw(ArgumentError("inertia w must be in [0, 1]"))
    c1 >= 0           || throw(ArgumentError("c1 must be ≥ 0"))
    c2 >= 0           || throw(ArgumentError("c2 must be ≥ 0"))
    v_max > 0         || throw(ArgumentError("v_max must be positive"))

    seed_int = seed === nothing ? nothing : Int(seed)
    return PSOConfig(max_iter, n_particles, w, c1, c2, v_max, verbose, print_interval, seed_int)
end

# ============================================================================
# Internal helpers shared by all three methods
# ============================================================================

"""
    _ds_fidelity(system, target, u_vec, n_c, n_t, dt) -> Float64

Evaluate fidelity for a flattened control vector. Shared by all direct-search methods.
"""
function _ds_fidelity(system::AbstractQuantumSystem,
                       target::QuantumTarget,
                       u_vec::Vector{Float64},
                       n_c::Int, n_t::Int, dt::Float64)::Float64
    controls = reshape(u_vec, n_c, n_t)
    seq      = ControlSequence(controls, dt, dt * n_t, n_t)
    H_total  = build_total_hamiltonian(system, seq)
    U_steps  = compute_propagators(H_total, dt)
    U_total  = compute_total_propagator(U_steps)
    return compute_fidelity(U_total, target)
end

"""
    _ds_result(u_best, n_c, n_t, F_best, fid_hist, converged, reason,
               t_start, n_fid, algorithm, metadata) -> OptimizationResult

Build an `OptimizationResult` from direct-search outputs (no gradient history).
"""
function _ds_result(u_best::Vector{Float64},
                     n_c::Int, n_t::Int,
                     F_best::Float64,
                     fid_hist::Vector{Float64},
                     converged::Bool,
                     reason::String,
                     t_start::Float64,
                     n_fid::Int,
                     algorithm::String,
                     metadata::Dict{String,Any})::OptimizationResult
    return OptimizationResult(
        reshape(copy(u_best), n_c, n_t),
        F_best,
        fid_hist,
        Float64[],          # no gradient norm history for derivative-free
        length(fid_hist),
        converged,
        reason,
        time() - t_start,
        n_fid,
        0,                  # zero gradient evaluations
        merge(metadata, Dict{String,Any}("algorithm" => algorithm))
    )
end

# ============================================================================
# QOC-domain wrappers — delegate to generic implementations in
# Direct/SimplexSearch.jl and Metaheur/CMAES.jl.
# These have different type signatures (system, target, ctrl) vs the
# generic (f, θ0) versions, so they coexist via multiple dispatch.
# ============================================================================

# ============================================================================
# Nelder-Mead
# ============================================================================

"""
    nelder_mead_optimize(system::AbstractQuantumSystem,
                          target::QuantumTarget,
                          controls_init::ControlSequence;
                          config::NelderMeadConfig = NelderMeadConfig()) -> OptimizationResult

QOC wrapper for Nelder-Mead. Delegates to `nelder_mead_optimize(f, θ0)` in
`Direct/SimplexSearch.jl`. See that module for algorithm details.
Suitable for low-dimensional problems (n_controls × n_timesteps ≤ 50).

# Arguments
- `system`        — quantum system
- `target`        — optimization target
- `controls_init` — initial control sequence (starting point)
- `config`        — `NelderMeadConfig` (default `NelderMeadConfig()`)

# Returns
`OptimizationResult`.

# Example
```julia
result = nelder_mead_optimize(sys, target, seq;
             config = NelderMeadConfig(max_iter=5000, tol=1e-7))
```
"""
function nelder_mead_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::NelderMeadConfig = NelderMeadConfig()
)::OptimizationResult

    t_start = time()
    dt      = controls_init.dt
    n_c     = system.n_controls
    n_t     = controls_init.n_timesteps

    # Negate fidelity: generic optimizer minimises, we want to maximise fidelity
    f = u_v -> -_ds_fidelity(system, target, u_v, n_c, n_t, dt)
    u0 = vec(copy(controls_init.controls))

    θ_best, neg_F_best, stats = nelder_mead_optimize(f, u0;
        max_iters = config.max_iter,
        max_evals = config.max_iter * (n_c * n_t + 2),
        tol       = config.tol,
        α         = config.alpha,
        γ         = config.gamma,
        ρ_c       = config.beta,
        σ_s       = config.delta,
    )

    F_best   = -neg_F_best
    fid_hist = hasproperty(stats, :history) && !isempty(stats.history) ?
               [-x for x in stats.history] : [F_best]
    converged = stats.converged
    reason    = converged ? "simplex diameter < tol $(config.tol)" :
                            "maximum iterations/evaluations reached"

    return _ds_result(θ_best, n_c, n_t, F_best, fid_hist,
                       converged, reason, t_start, stats.evals,
                       "Nelder-Mead", Dict{String,Any}())
end

# ============================================================================
# CMA-ES
# ============================================================================

"""
    cmaes_optimize(system::AbstractQuantumSystem,
                   target::QuantumTarget,
                   controls_init::ControlSequence;
                   config::CMAESConfig = CMAESConfig()) -> OptimizationResult

QOC wrapper for CMA-ES. Delegates to `cmaes_optimize(f, θ0)` in
`Metaheur/CMAES.jl`. See that module for algorithm details.

# Arguments
- `system`        — quantum system
- `target`        — optimization target
- `controls_init` — initial control sequence (provides starting mean m = u0)
- `config`        — `CMAESConfig` (default `CMAESConfig()`)

# Returns
`OptimizationResult`.

# Example
```julia
result = cmaes_optimize(sys, target, seq;
             config = CMAESConfig(max_iter=500, sigma_init=0.3))
```
"""
function cmaes_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::CMAESConfig = CMAESConfig()
)::OptimizationResult

    t_start = time()
    dt      = controls_init.dt
    n_c     = system.n_controls
    n_t     = controls_init.n_timesteps

    f = u_v -> -_ds_fidelity(system, target, u_v, n_c, n_t, dt)
    u0 = vec(copy(controls_init.controls))

    θ_best, neg_F_best, stats = cmaes_optimize(f, u0;
        max_iters  = config.max_iter,
        max_evals  = config.max_iter * (config.population_size > 0 ?
                         config.population_size :
                         4 + floor(Int, 3 * log(n_c * n_t))) * 2,
        popsize    = config.population_size,
        sigma_init = config.sigma_init,
        tol_fun    = config.tol_fun,
        tol_x      = config.tol_x,
    )

    F_best   = -neg_F_best
    fid_hist = hasproperty(stats, :history) && !isempty(stats.history) ?
               [-x for x in stats.history] : [F_best]
    converged = stats.converged
    reason    = converged ? "CMA-ES convergence tolerance reached" :
                            "maximum iterations/evaluations reached"

    return _ds_result(θ_best, n_c, n_t, F_best, fid_hist,
                       converged, reason, t_start, stats.evals,
                       "CMA-ES", Dict{String,Any}())
end

# ============================================================================
# PSO
# ============================================================================

"""
    pso_optimize(system::AbstractQuantumSystem,
                 target::QuantumTarget,
                 controls_init::ControlSequence;
                 config::PSOConfig = PSOConfig()) -> OptimizationResult

Particle Swarm Optimization (PSO) for quantum control.

Maintains a swarm of candidate solutions, each with a velocity.  At every
iteration, each particle moves according to its inertia plus attraction toward
its personal best and the global best.

# Update Rule
    v_{i} ← w * v_{i}  +  c1 * r1 * (p_i - x_i)  +  c2 * r2 * (g - x_i)
    x_{i} ← x_{i} + v_{i}

where r1, r2 ~ Uniform(0, 1) are fresh each iteration, `p_i` is particle i's
personal best, and `g` is the global best.  Velocity is clamped to `[-v_max, v_max]`.

# Initialization
Particles are initialized by adding Gaussian noise (std = 0.1 * ‖u0‖ + 0.1)
to the initial control vector.  Velocities are initialized as zero.

# Arguments
- `system`        — quantum system
- `target`        — optimization target
- `controls_init` — initial control sequence (determines swarm center)
- `config`        — `PSOConfig` (default `PSOConfig()`)

# Returns
`OptimizationResult`.

# Example
```julia
result = pso_optimize(sys, target, seq;
             config = PSOConfig(max_iter=500, n_particles=30))
```
"""
function pso_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::PSOConfig = PSOConfig()
)::OptimizationResult

    t_start = time()
    dt      = controls_init.dt
    n_c     = system.n_controls
    n_t     = controls_init.n_timesteps

    f = u_v -> -_ds_fidelity(system, target, u_v, n_c, n_t, dt)
    u0 = vec(copy(controls_init.controls))

    θ_best, neg_F_best, stats = pso_optimize(f, u0;
        max_iters = config.max_iter,
        max_evals = config.max_iter * max(10, 2 + n_c * n_t ÷ 2) * 2,
        popsize   = config.n_particles,
        w         = config.w,
        c1        = config.c1,
        c2        = config.c2,
        v_max     = config.v_max,
        seed      = config.seed,
    )

    F_best   = -neg_F_best
    fid_hist = hasproperty(stats, :history) && !isempty(stats.history) ?
               [-x for x in stats.history] : [F_best]
    converged = stats.converged
    reason    = converged ? "PSO swarm converged" :
                            "maximum iterations/evaluations reached"

    return _ds_result(θ_best, n_c, n_t, F_best, fid_hist,
                       converged, reason, t_start, stats.evals,
                       "PSO", Dict{String,Any}("n_particles" => config.n_particles,
                                               "inertia"     => config.w))
end

# ============================================================================
# Backward-compatible Matrix+dt overloads
# ============================================================================
for fn in (:nelder_mead_optimize, :cmaes_optimize, :pso_optimize)
    @eval function $fn(system::AbstractQuantumSystem,
                        target::QuantumTarget,
                        u_init::AbstractMatrix{<:Real},
                        dt::Real;
                        kwargs...)::OptimizationResult
        n_c, n_t = size(u_init)
        seq = ControlSequence(Matrix{Float64}(u_init), Float64(dt),
                              Float64(dt) * n_t, n_t)
        return $fn(system, target, seq; kwargs...)
    end
end
