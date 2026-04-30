"""
    SecondOrderMethods.jl

Quasi-Newton and Newton second-order optimization methods for Pulsar.

Implements three methods:

1. **BFGS** — Broyden–Fletcher–Goldfarb–Shanno with inverse Hessian approximation.
   Superlinear convergence near the optimum.  Full n×n Hessian stored.

2. **L-BFGS** — Limited-memory BFGS for large problems.  Only the last `m`
   (s, y) correction pairs are stored, giving O(m·n) memory instead of O(n²).
   Uses Nocedal's two-loop recursion for the direction computation.

3. **Newton-CG** — Full Newton step with Hessian approximated by finite
   differences.  Uses Tikhonov regularization to handle indefinite Hessians.
   Direction is computed by solving (H + λI) d = -g.

All three methods use a line search (backtracking Armijo or strong Wolfe
conditions) to ensure sufficient decrease at each step.

References:
  Nocedal & Wright, "Numerical Optimization", 2nd ed., Springer (2006).
  Liu & Nocedal, "On the limited memory BFGS method", Math. Prog. 45 (1989).
"""

# ============================================================================
# Configuration types
# ============================================================================

"""
    BFGSConfig

Configuration for the BFGS optimizer.

# Fields
- `max_iter::Int` — maximum iterations (default 500)
- `convergence_tol::Float64` — fidelity change tolerance (default 1e-8)
- `gradient_tol::Float64` — gradient norm tolerance (default 1e-6)
- `line_search_method::String` — `"backtracking"` (default) or `"wolfe"`
- `initial_hessian_scale::Float64` — scale for initial inverse Hessian H₀ = scale*I (default 1.0)
- `verbose::Bool` — print progress (default false)
- `print_interval::Int` — logging frequency (default 50)

# Example
```julia
cfg = BFGSConfig(max_iter=200, gradient_tol=1e-7, line_search_method="wolfe")
```
"""
struct BFGSConfig
    max_iter::Int
    convergence_tol::Float64
    gradient_tol::Float64
    line_search_method::String
    initial_hessian_scale::Float64
    verbose::Bool
    print_interval::Int
    convergence_mode::Symbol
    wolfe_c1::Float64
    wolfe_c2::Float64
end

"""
    BFGSConfig(; kwargs...) -> BFGSConfig

Construct a `BFGSConfig` with keyword arguments. See struct docstring for fields.

The `convergence_mode` keyword selects the stopping rule:
  - `:gradient_norm`   — stop when ‖∇F‖ < `gradient_tol` (default; Nocedal & Wright)
  - `:fidelity_change` — stop when |ΔF| < `convergence_tol` (Krotov.jl-style; useful
    when the gradient plateaus on ill-conditioned problems but progress continues)
  - `:both`            — both conditions must hold
"""
function BFGSConfig(;
    max_iter::Int                                    = 500,
    convergence_tol::Float64                         = 1e-8,
    gradient_tol::Float64                            = 1e-6,
    line_search_method::String                       = "backtracking",
    line_search::Union{Nothing,Symbol,String}        = nothing,   # legacy alias
    initial_hessian_scale::Float64                   = 1.0,
    verbose::Bool                                    = false,
    print_interval::Int                              = 50,
    convergence_mode::Symbol                         = :gradient_norm,
    wolfe_c1::Float64                                = 1e-4,
    wolfe_c2::Float64                                = 0.9,
    record_line_search::Bool                         = false,    # accepted (recording stub)
    record_hessian::Bool                             = false     # accepted (recording stub)
)::BFGSConfig
    if line_search !== nothing
        line_search_method = String(line_search)
    end
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    line_search_method in ("backtracking", "wolfe") ||
        throw(ArgumentError("line_search_method must be \"backtracking\" or \"wolfe\""))
    initial_hessian_scale > 0 ||
        throw(ArgumentError("initial_hessian_scale must be positive"))
    convergence_mode in (:gradient_norm, :fidelity_change, :both) ||
        throw(ArgumentError("convergence_mode must be :gradient_norm, :fidelity_change, or :both"))
    0.0 < wolfe_c1 < wolfe_c2 < 1.0 ||
        throw(ArgumentError("require 0 < wolfe_c1 < wolfe_c2 < 1 (got c1=$wolfe_c1, c2=$wolfe_c2)"))

    return BFGSConfig(max_iter, convergence_tol, gradient_tol,
                      line_search_method, initial_hessian_scale,
                      verbose, print_interval, convergence_mode,
                      wolfe_c1, wolfe_c2)
end

# ----------------------------------------------------------------------------

"""
    LBFGSConfig

Configuration for the L-BFGS optimizer.

# Fields
- `max_iter::Int` — maximum iterations (default 500)
- `convergence_tol::Float64` — fidelity change tolerance (default 1e-8)
- `gradient_tol::Float64` — gradient norm tolerance (default 1e-6)
- `memory_size::Int` — number of (s, y) pairs to store (default 10)
- `line_search_method::String` — `"backtracking"` (default) or `"wolfe"`
- `verbose::Bool` — print progress (default false)
- `print_interval::Int` — logging frequency (default 50)

# Example
```julia
cfg = LBFGSConfig(max_iter=1000, memory_size=20)
```
"""
struct LBFGSConfig
    max_iter::Int
    convergence_tol::Float64
    gradient_tol::Float64
    memory_size::Int
    line_search_method::String
    verbose::Bool
    print_interval::Int
    convergence_mode::Symbol
    wolfe_c1::Float64
    wolfe_c2::Float64
end

"""
    LBFGSConfig(; kwargs...) -> LBFGSConfig

Construct a `LBFGSConfig` with keyword arguments. See struct docstring for fields.

`convergence_mode` ∈ (`:gradient_norm`, `:fidelity_change`, `:both`); see
`BFGSConfig` for semantics.
"""
function LBFGSConfig(;
    max_iter::Int              = 500,
    convergence_tol::Float64   = 1e-8,
    gradient_tol::Float64      = 1e-6,
    memory_size::Int           = 10,
    line_search_method::String = "backtracking",
    verbose::Bool              = false,
    print_interval::Int        = 50,
    convergence_mode::Symbol   = :gradient_norm,
    wolfe_c1::Float64          = 1e-4,
    wolfe_c2::Float64          = 0.9,
)::LBFGSConfig
    max_iter > 0    || throw(ArgumentError("max_iter must be positive"))
    memory_size >= 1 || throw(ArgumentError("memory_size must be ≥ 1"))
    line_search_method in ("backtracking", "wolfe") ||
        throw(ArgumentError("line_search_method must be \"backtracking\" or \"wolfe\""))
    convergence_mode in (:gradient_norm, :fidelity_change, :both) ||
        throw(ArgumentError("convergence_mode must be :gradient_norm, :fidelity_change, or :both"))
    0.0 < wolfe_c1 < wolfe_c2 < 1.0 ||
        throw(ArgumentError("require 0 < wolfe_c1 < wolfe_c2 < 1 (got c1=$wolfe_c1, c2=$wolfe_c2)"))

    return LBFGSConfig(max_iter, convergence_tol, gradient_tol,
                       memory_size, line_search_method, verbose, print_interval,
                       convergence_mode, wolfe_c1, wolfe_c2)
end

# ----------------------------------------------------------------------------

"""
    NewtonConfig

Configuration for the Newton-CG optimizer with finite-difference Hessian.

# Fields
- `max_iter::Int` — maximum iterations (default 200)
- `convergence_tol::Float64` — fidelity change tolerance (default 1e-8)
- `gradient_tol::Float64` — gradient norm tolerance (default 1e-6)
- `hessian_method::String` — `"finite_diff"` (only supported method, default)
- `regularization::Float64` — Tikhonov regularization λ: solves (H+λI)d=-g (default 1e-4)
- `finite_diff_eps::Float64` — finite difference step for Hessian (default 1e-5)
- `verbose::Bool` — print progress (default false)
- `print_interval::Int` — logging frequency (default 20)

# Example
```julia
cfg = NewtonConfig(max_iter=100, regularization=1e-3)
```
"""
struct NewtonConfig
    max_iter::Int
    convergence_tol::Float64
    gradient_tol::Float64
    hessian_method::String
    regularization::Float64
    finite_diff_eps::Float64
    verbose::Bool
    print_interval::Int
    convergence_mode::Symbol
end

"""
    NewtonConfig(; kwargs...) -> NewtonConfig

Construct a `NewtonConfig` with keyword arguments. See struct docstring for fields.

`convergence_mode` ∈ (`:gradient_norm`, `:fidelity_change`, `:both`); see
`BFGSConfig` for semantics.
"""
function NewtonConfig(;
    max_iter::Int            = 200,
    convergence_tol::Float64 = 1e-8,
    gradient_tol::Float64    = 1e-6,
    hessian_method::String   = "finite_diff",
    regularization::Float64  = 1e-4,
    finite_diff_eps::Float64 = 1e-5,
    verbose::Bool            = false,
    print_interval::Int      = 20,
    convergence_mode::Symbol = :gradient_norm,
)::NewtonConfig
    max_iter > 0      || throw(ArgumentError("max_iter must be positive"))
    regularization >= 0 || throw(ArgumentError("regularization must be ≥ 0"))
    finite_diff_eps > 0 || throw(ArgumentError("finite_diff_eps must be positive"))
    hessian_method == "finite_diff" ||
        throw(ArgumentError("hessian_method must be \"finite_diff\""))
    convergence_mode in (:gradient_norm, :fidelity_change, :both) ||
        throw(ArgumentError("convergence_mode must be :gradient_norm, :fidelity_change, or :both"))

    return NewtonConfig(max_iter, convergence_tol, gradient_tol,
                        hessian_method, regularization, finite_diff_eps,
                        verbose, print_interval, convergence_mode)
end

# ============================================================================
# Internal helpers
# ============================================================================

"""
    _so_fidelity(system, target, u_vec, n_c, n_t, dt) -> Float64

Evaluate fidelity for a flattened control vector `u_vec` of length `n_c * n_t`.
"""
function _so_fidelity(system::AbstractQuantumSystem,
                       target::QuantumTarget,
                       u_vec::Vector{Float64},
                       n_c::Int, n_t::Int, dt::Float64)::Float64
    controls = reshape(u_vec, n_c, n_t)
    seq = ControlSequence(controls, dt, dt * n_t, n_t)
    H_total = build_total_hamiltonian(system, seq)
    U_steps = compute_propagators(H_total, dt)
    U_total = compute_total_propagator(U_steps)
    return compute_fidelity(U_total, target)
end

"""
    _so_gradient(system, target, u_vec, n_c, n_t, dt) -> Vector{Float64}

Evaluate the flattened GRAPE gradient for a control vector `u_vec`.
"""
function _so_gradient(system::AbstractQuantumSystem,
                       target::QuantumTarget,
                       u_vec::Vector{Float64},
                       n_c::Int, n_t::Int, dt::Float64)::Vector{Float64}
    controls = reshape(u_vec, n_c, n_t)
    seq = ControlSequence(controls, dt, dt * n_t, n_t)
    G = compute_grape_gradient(system, seq, target)
    return vec(G)
end

"""
    _make_result(u_best, F_best, fid_hist, gnorm_hist, converged, reason,
                 t_start, n_fid, n_grad, algorithm, metadata) -> OptimizationResult

Internal constructor for `OptimizationResult` from flattened controls.
"""
function _make_result(u_best::Vector{Float64},
                       n_c::Int, n_t::Int,
                       F_best::Float64,
                       fid_hist::Vector{Float64},
                       gnorm_hist::Vector{Float64},
                       converged::Bool,
                       reason::String,
                       t_start::Float64,
                       n_fid::Int,
                       n_grad::Int,
                       algorithm::String,
                       metadata::Dict{String,Any})::OptimizationResult
    return OptimizationResult(
        reshape(copy(u_best), n_c, n_t),
        F_best,
        fid_hist,
        gnorm_hist,
        length(fid_hist),
        converged,
        reason,
        time() - t_start,
        n_fid,
        n_grad,
        merge(metadata, Dict{String,Any}("algorithm" => algorithm))
    )
end

# ============================================================================
# QOC-domain wrappers — delegate to generic implementations in
# Gradient/Generic/QuasiNewton.jl and Gradient/Generic/SecondOrder.jl.
# These have different type signatures (system, target, ctrl) vs the
# generic (f, grad!, u0) versions, so they coexist via multiple dispatch.
# ============================================================================

# ============================================================================
# BFGS
# ============================================================================

"""
    bfgs_optimize(system::AbstractQuantumSystem,
                  target::QuantumTarget,
                  controls_init::ControlSequence;
                  config::BFGSConfig = BFGSConfig()) -> OptimizationResult

QOC wrapper for BFGS. Delegates to `bfgs_optimize(f, grad!, u0)` in
`Gradient/Generic/QuasiNewton.jl`. See that module for algorithm details.

# Arguments
- `system`        — quantum system
- `target`        — optimization target
- `controls_init` — initial control sequence
- `config`        — `BFGSConfig` (default `BFGSConfig()`)

# Returns
`OptimizationResult`.

# Example
```julia
result = bfgs_optimize(sys, target, seq;
             config = BFGSConfig(max_iter=300, line_search_method="wolfe"))
```
"""
function bfgs_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::BFGSConfig = BFGSConfig()
)::OptimizationResult

    t_start = time()
    dt  = controls_init.dt
    n_c = system.n_controls
    n_t = controls_init.n_timesteps

    f_neg(u)      = -_so_fidelity(system, target, u, n_c, n_t, dt)
    grad_neg!(g, u) = (g .= -_so_gradient(system, target, u, n_c, n_t, dt))
    u0 = vec(copy(controls_init.controls))

    θ_best, neg_F_best, stats = bfgs_optimize(f_neg, grad_neg!, u0;
        max_iter         = config.max_iter,
        tol              = config.gradient_tol,
        verbose          = config.verbose,
        convergence_mode = config.convergence_mode,
        f_tol            = config.convergence_tol,
        wolfe_c1         = config.wolfe_c1,
        wolfe_c2         = config.wolfe_c2,
    )

    F_best   = -neg_F_best
    fid_hist = hasproperty(stats, :history) && !isempty(stats.history) ?
               [-x for x in stats.history] : [F_best]
    converged = stats.converged
    reason    = converged ? "gradient norm < $(config.gradient_tol)" :
                            "maximum iterations reached"

    return _make_result(θ_best, n_c, n_t, F_best, fid_hist, Float64[],
                        converged, reason, t_start, stats.evals, 0,
                        "BFGS", Dict{String,Any}())
end

# ============================================================================
# L-BFGS optimizer
# ============================================================================

"""
    lbfgs_optimize(system::AbstractQuantumSystem,
                   target::QuantumTarget,
                   controls_init::ControlSequence;
                   config::LBFGSConfig = LBFGSConfig()) -> OptimizationResult

QOC wrapper for L-BFGS. Delegates to `lbfgs_optimize(f, grad!, u0)` in
`Gradient/Generic/QuasiNewton.jl`. See that module for algorithm details.

# Arguments
- `system`        — quantum system
- `target`        — optimization target
- `controls_init` — initial control sequence
- `config`        — `LBFGSConfig` (default `LBFGSConfig()`)

# Returns
`OptimizationResult`.

# Example
```julia
result = lbfgs_optimize(sys, target, seq;
             config = LBFGSConfig(max_iter=500, memory_size=20))
```
"""
function lbfgs_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::LBFGSConfig = LBFGSConfig()
)::OptimizationResult

    t_start = time()
    dt  = controls_init.dt
    n_c = system.n_controls
    n_t = controls_init.n_timesteps

    f_neg(u)        = -_so_fidelity(system, target, u, n_c, n_t, dt)
    grad_neg!(g, u) = (g .= -_so_gradient(system, target, u, n_c, n_t, dt))
    u0 = vec(copy(controls_init.controls))

    θ_best, neg_F_best, stats = lbfgs_optimize(f_neg, grad_neg!, u0;
        memory           = config.memory_size,
        max_iter         = config.max_iter,
        tol              = config.gradient_tol,
        verbose          = config.verbose,
        convergence_mode = config.convergence_mode,
        f_tol            = config.convergence_tol,
        wolfe_c1         = config.wolfe_c1,
        wolfe_c2         = config.wolfe_c2,
    )

    F_best   = -neg_F_best
    fid_hist = hasproperty(stats, :history) && !isempty(stats.history) ?
               [-x for x in stats.history] : [F_best]
    converged = stats.converged
    reason    = converged ? "gradient norm < $(config.gradient_tol)" :
                            "maximum iterations reached"

    return _make_result(θ_best, n_c, n_t, F_best, fid_hist, Float64[],
                        converged, reason, t_start, stats.evals, 0,
                        "L-BFGS",
                        Dict{String,Any}("memory_size" => config.memory_size))
end

# ============================================================================
# Newton optimizer
# ============================================================================

"""
    newton_optimize(system::AbstractQuantumSystem,
                    target::QuantumTarget,
                    controls_init::ControlSequence;
                    config::NewtonConfig = NewtonConfig()) -> OptimizationResult

QOC wrapper for Newton-CG. Delegates to `newton_optimize(f, grad!, u0)` in
`Gradient/Generic/SecondOrder.jl`. See that module for algorithm details.
Practical only for small problems (n_controls × n_timesteps ≤ 50).

# Arguments
- `system`        — quantum system
- `target`        — optimization target
- `controls_init` — initial control sequence
- `config`        — `NewtonConfig` (default `NewtonConfig()`)

# Returns
`OptimizationResult`.

# Example
```julia
result = newton_optimize(sys, target, seq;
             config = NewtonConfig(max_iter=50, regularization=1e-3))
```
"""
function newton_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::NewtonConfig = NewtonConfig()
)::OptimizationResult

    t_start = time()
    dt  = controls_init.dt
    n_c = system.n_controls
    n_t = controls_init.n_timesteps

    f_neg(u)        = -_so_fidelity(system, target, u, n_c, n_t, dt)
    grad_neg!(g, u) = (g .= -_so_gradient(system, target, u, n_c, n_t, dt))
    u0 = vec(copy(controls_init.controls))

    # Use generic Newton-CG with finite-difference HVP
    θ_best, neg_F_best, stats = newton_optimize(f_neg, grad_neg!, u0;
        shift            = config.regularization,
        max_iter         = config.max_iter,
        tol              = config.gradient_tol,
        verbose          = config.verbose,
        convergence_mode = config.convergence_mode,
        f_tol            = config.convergence_tol,
    )

    F_best   = -neg_F_best
    fid_hist = hasproperty(stats, :history) && !isempty(stats.history) ?
               [-x for x in stats.history] : [F_best]
    converged = stats.converged
    reason    = converged ? "gradient norm < $(config.gradient_tol)" :
                            "maximum iterations reached"

    return _make_result(θ_best, n_c, n_t, F_best, fid_hist, Float64[],
                        converged, reason, t_start, stats.evals, 0,
                        "Newton",
                        Dict{String,Any}("regularization" => config.regularization))
end

# ============================================================================
# Backward-compatible Matrix + dt overloads
# ============================================================================
# Build a `ControlSequence` from `(u_init, dt)` and forward to the canonical
# method.  Used by the legacy unit-test calling convention.

for fn in (:bfgs_optimize, :lbfgs_optimize, :newton_optimize)
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

# ============================================================================
# Trust-Region methods (folded from TrustRegionMethods.jl)
# Reference: Nocedal & Wright "Numerical Optimization", Ch. 4–5.
# ============================================================================

"""
    TrustRegionConfig(; kwargs...) -> TrustRegionConfig

Configuration for trust-region optimization.

# Fields
- `initial_radius`: Starting trust-region radius (default: 1.0)
- `max_radius`: Maximum allowed radius (default: 100.0)
- `min_radius`: Minimum radius; below this convergence declared (default: 1e-8)
- `eta`: Acceptance threshold — accept step if ρ ≥ η (default: 0.1)
- `eta_very_successful`: Expand radius if ρ > this threshold (default: 0.75)
- `max_iter`: Maximum iterations (default: 500)
- `subproblem_solver`: Subproblem solver — only "cg_steihaug" is currently active;
  "cauchy" and "dogleg" are accepted but fall back to "cg_steihaug" with a warning
- `hessian_method`: "finite_diff" (Hessian-vector products via FD) or
  "bfgs_update" (L-BFGS Hessian-vector approximation from gradient history)
- `gradient_tol`: Stop if ‖∇F‖ < gradient_tol (default: 1e-6)
- `verbose`: Print iteration log (default: false)
- `print_interval`: Print every N iterations (default: 50)
"""
struct TrustRegionConfig
    initial_radius::Float64
    max_radius::Float64
    min_radius::Float64
    eta::Float64
    eta_very_successful::Float64
    max_iter::Int
    subproblem_solver::String
    hessian_method::String
    gradient_tol::Float64
    verbose::Bool
    print_interval::Int
end

function TrustRegionConfig(;
    initial_radius::Float64      = 1.0,
    max_radius::Float64          = 100.0,
    min_radius::Float64          = 1e-8,
    eta::Float64                 = 0.1,
    eta_very_successful::Float64 = 0.75,
    max_iter::Int                = 500,
    subproblem_solver::String    = "dogleg",
    hessian_method::String       = "bfgs_update",
    gradient_tol::Float64        = 1e-6,
    verbose::Bool                = false,
    print_interval::Int          = 50,
)
    return TrustRegionConfig(initial_radius, max_radius, min_radius, eta,
                              eta_very_successful, max_iter, subproblem_solver,
                              hessian_method, gradient_tol, verbose, print_interval)
end

"""
    trust_region_optimize(system, target, controls_init; config) -> OptimizationResult

Optimize quantum control using trust-region methods.

At each iteration k:
1. Compute gradient gₖ = ∇F(uₖ) via GRAPE
2. Compute / update approximate Hessian Bₖ (BFGS update or finite diff)
3. Solve trust-region subproblem: min_{‖p‖≤Δ}  gₖ'p + ½p'Bₖp
4. Evaluate ratio ρ = actual_reduction / predicted_reduction
5. Update radius and accept/reject step based on ρ

# Arguments
- `system`        — quantum system (AbstractQuantumSystem)
- `target`        — optimization target (QuantumTarget)
- `controls_init` — initial control sequence (ControlSequence)
- `config`        — TrustRegionConfig with algorithm parameters
"""
function trust_region_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::TrustRegionConfig = TrustRegionConfig(),
)::OptimizationResult

    t_start = time()
    n_c = system.n_controls
    n_t = controls_init.n_timesteps
    dt  = controls_init.dt

    # Validate subproblem_solver — only cg_steihaug is implemented in the
    # generic trust_region_newton_optimize; warn for other choices.
    if config.subproblem_solver ∉ ("cg_steihaug", "cauchy", "dogleg")
        throw(ArgumentError("subproblem_solver must be \"cg_steihaug\", \"cauchy\", or \"dogleg\""))
    end
    if config.subproblem_solver != "cg_steihaug"
        @warn "subproblem_solver=\"$(config.subproblem_solver)\" is not yet implemented; " *
              "falling back to \"cg_steihaug\"."
    end

    f_neg(u)        = -_so_fidelity(system, target, u, n_c, n_t, dt)
    grad_neg!(g, u) = (g .= -_so_gradient(system, target, u, n_c, n_t, dt))
    u0 = vec(copy(controls_init.controls))

    # Wire up hessian_method: build an L-BFGS HVP approximation when requested
    hvp_fn = nothing
    if config.hessian_method == "bfgs_update"
        # L-BFGS Hessian-vector product using a small history (m=5)
        lbfgs_s = Vector{Vector{Float64}}()
        lbfgs_y = Vector{Vector{Float64}}()
        g_prev  = zeros(length(u0))
        u_prev  = copy(u0)
        grad_neg!(g_prev, u_prev)
        function hvp_fn!(Hv, u, v)
            # Update L-BFGS history from gradient change
            g_cur = zeros(length(u))
            grad_neg!(g_cur, u)
            sk = u .- u_prev
            yk = g_cur .- g_prev
            sy = dot(sk, yk)
            if sy > 1e-14 * dot(sk, sk)
                push!(lbfgs_s, copy(sk))
                push!(lbfgs_y, copy(yk))
                if length(lbfgs_s) > 5
                    popfirst!(lbfgs_s); popfirst!(lbfgs_y)
                end
            end
            @. u_prev = u
            @. g_prev = g_cur
            # Two-loop L-BFGS HVP: Hv ≈ H_k * v
            q = copy(v)
            m = length(lbfgs_s)
            α_vec = zeros(m)
            for i in m:-1:1
                ρi = 1.0 / dot(lbfgs_y[i], lbfgs_s[i])
                α_vec[i] = ρi * dot(lbfgs_s[i], q)
                q .-= α_vec[i] .* lbfgs_y[i]
            end
            # Scale by initial Hessian γ = sᵀy / yᵀy
            if m > 0
                γ = dot(lbfgs_s[m], lbfgs_y[m]) / dot(lbfgs_y[m], lbfgs_y[m])
                q .*= γ
            end
            for i in 1:m
                ρi = 1.0 / dot(lbfgs_y[i], lbfgs_s[i])
                β  = ρi * dot(lbfgs_y[i], q)
                q .+= (α_vec[i] - β) .* lbfgs_s[i]
            end
            @. Hv = q
        end
    elseif config.hessian_method != "finite_diff"
        throw(ArgumentError("hessian_method must be \"finite_diff\" or \"bfgs_update\""))
    end

    θ_best, neg_F_best, stats = trust_region_newton_optimize(f_neg, grad_neg!, u0;
        hvp!     = hvp_fn,
        Δ0       = config.initial_radius,
        Δ_max    = config.max_radius,
        η        = config.eta,
        max_iter = config.max_iter,
        tol      = config.gradient_tol,
        verbose  = config.verbose,
    )

    F_best   = -neg_F_best
    fid_hist = hasproperty(stats, :history) && !isempty(stats.history) ?
               [-x for x in stats.history] : [F_best]
    converged = stats.converged
    reason    = converged ? "gradient norm < $(config.gradient_tol)" :
                            "maximum iterations reached"

    return OptimizationResult(
        reshape(θ_best, n_c, n_t), F_best,
        fid_hist, Float64[],
        stats.iters, converged, reason, time() - t_start,
        stats.evals, 0,
        Dict{String,Any}("algorithm"       => "Trust-Region",
                         "subproblem"      => "cg_steihaug",
                         "hessian_method"  => config.hessian_method),
    )
end
