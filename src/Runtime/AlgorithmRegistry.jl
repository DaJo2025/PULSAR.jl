# ============================================================
# PULSAR.jl — Algorithm Routing Registry (Theme 8)
# ============================================================
#
# Central registry mapping optimizer names → callables and capability
# metadata.  Replaces the if-elseif chains scattered across:
#
#   • Application/MR/OptControl.jl :: _optimcon_generic / _MR_GENERIC_METHODS
#   • Application/QuantumComputing/OptControl.jl :: optimcon dispatch
#   • Runtime/AlgorithmSelection.jl :: auto_optimize dispatch
#
# So that:
#   1. New optimizers register once and become reachable from every
#      Application wrapper that knows how to call the registry.
#   2. `list_optimizers(; gradient, bounds, noise)` answers "what can I
#      run on my current problem?" — the natural query that
#      `recommend_optimizer` should consume.
#
# The registry is intentionally lightweight (a Dict + a struct) so it can
# be loaded early in the module without pulling in any optimizer code.
# Optimizer files register their entries via small `register_optimizer!`
# calls at top-level after their callables are defined.
# ============================================================

"""
    OptimizerSupports

NamedTuple-backed capability descriptor for a registered optimizer.

# Fields
- `gradient    :: Bool` — uses analytical gradient information
- `hessian     :: Bool` — uses (or approximates) Hessian information
- `bounds      :: Bool` — supports box-bounds on optimization variables
- `noise       :: Bool` — supports stochastic / noisy objective evaluations
- `parallel    :: Bool` — exploits parallelism over ensemble members
- `qoc         :: Bool` — accepts the GRAPE-style waveform interface natively
- `generic     :: Bool` — accepts the generic `f(θ), grad!(g, θ)` interface
- `open_system :: Bool` — runs end-to-end on `LindbladMRControl` /
  `LindbladQCControl` via the application-layer kernel dispatch
  (`_mr_kernel` / `_qc_kernel`).  Algorithms that genuinely require
  Hilbert-space structure (e.g. analytic Tr-based gate fidelity) are
  marked `false`.
"""
const OptimizerSupports = @NamedTuple{
    gradient    :: Bool,
    hessian     :: Bool,
    bounds      :: Bool,
    noise       :: Bool,
    parallel    :: Bool,
    qoc         :: Bool,
    generic     :: Bool,
    open_system :: Bool,
}

"""
    OptimizerEntry

Single registered optimizer.  Consumers select on `name` and use
`callable` directly — no wrapping logic lives here.

# Fields
- `name        :: Symbol` — registry key, e.g. `:lbfgs`, `:grape`, `:cmaes`
- `callable    :: Function` — the optimizer entry point (signature varies
  by `style`; see `supports.qoc` and `supports.generic`)
- `supports    :: OptimizerSupports` — capability metadata
- `description :: String` — one-line human-readable summary
"""
struct OptimizerEntry
    name        :: Symbol
    callable    :: Function
    supports    :: OptimizerSupports
    description :: String
end

"""
    OPTIMIZER_REGISTRY :: Dict{Symbol,OptimizerEntry}

Process-wide registry of available optimizers.  Mutated only via
[`register_optimizer!`](@ref).  Lookup via [`get_optimizer`](@ref); query
via [`list_optimizers`](@ref).
"""
const OPTIMIZER_REGISTRY = Dict{Symbol,OptimizerEntry}()

"""
    register_optimizer!(entry::OptimizerEntry; replace=false)

Insert or replace an entry in [`OPTIMIZER_REGISTRY`](@ref).  Throws
`ArgumentError` if `entry.name` is already registered and `replace=false`.

Idempotency: re-registering the *same* `OptimizerEntry` (`==`) is a
no-op — useful when a module is reloaded in the REPL.
"""
function register_optimizer!(entry::OptimizerEntry; replace::Bool = false)
    name = entry.name
    if haskey(OPTIMIZER_REGISTRY, name)
        existing = OPTIMIZER_REGISTRY[name]
        if existing.callable === entry.callable &&
           existing.supports == entry.supports &&
           existing.description == entry.description
            return entry   # idempotent re-register
        end
        replace || throw(ArgumentError(
            "Optimizer ':$name' already registered.  Pass replace=true to override."))
    end
    OPTIMIZER_REGISTRY[name] = entry
    return entry
end

"""
    register_optimizer!(name::Symbol, callable::Function, description::String;
                         gradient=false, hessian=false, bounds=false,
                         noise=false, parallel=false, qoc=false, generic=true,
                         replace=false)

Convenience overload that builds the [`OptimizerEntry`](@ref) inline.
Keyword defaults reflect the most common case (a generic gradient-free
function-interface optimizer).
"""
function register_optimizer!(name::Symbol, callable::Function, description::String;
                              gradient    :: Bool = false,
                              hessian     :: Bool = false,
                              bounds      :: Bool = false,
                              noise       :: Bool = false,
                              parallel    :: Bool = false,
                              qoc         :: Bool = false,
                              generic     :: Bool = true,
                              open_system :: Bool = false,
                              replace     :: Bool = false)
    sup = (gradient = gradient, hessian = hessian, bounds = bounds,
           noise = noise, parallel = parallel, qoc = qoc, generic = generic,
           open_system = open_system)
    return register_optimizer!(OptimizerEntry(name, callable, sup, description);
                                replace = replace)
end

"""
    get_optimizer(name::Symbol) -> OptimizerEntry

Look up an optimizer by registry name.  Throws `KeyError` with a
diagnostic message if the name is unknown.
"""
function get_optimizer(name::Symbol)::OptimizerEntry
    haskey(OPTIMIZER_REGISTRY, name) || throw(KeyError(
        "Optimizer ':$name' is not registered.  Known: $(sort(collect(keys(OPTIMIZER_REGISTRY))))"))
    return OPTIMIZER_REGISTRY[name]
end

"""
    list_optimizers(; gradient=nothing, hessian=nothing, bounds=nothing,
                     noise=nothing, parallel=nothing, qoc=nothing,
                     generic=nothing) -> Vector{Symbol}

Return the registry names matching every supplied capability filter.
A `nothing` filter is ignored; a `Bool` filter requires equality.

# Example
```julia
# Optimizers that accept a generic `f(θ), grad!(g, θ)` interface and
# use gradient information:
list_optimizers(gradient=true, generic=true)
```
"""
function list_optimizers(; gradient    :: Union{Nothing,Bool} = nothing,
                          hessian     :: Union{Nothing,Bool} = nothing,
                          bounds      :: Union{Nothing,Bool} = nothing,
                          noise       :: Union{Nothing,Bool} = nothing,
                          parallel    :: Union{Nothing,Bool} = nothing,
                          qoc         :: Union{Nothing,Bool} = nothing,
                          generic     :: Union{Nothing,Bool} = nothing,
                          open_system :: Union{Nothing,Bool} = nothing)::Vector{Symbol}
    out = Symbol[]
    for (name, e) in OPTIMIZER_REGISTRY
        s = e.supports
        gradient    === nothing || s.gradient    == gradient    || continue
        hessian     === nothing || s.hessian     == hessian     || continue
        bounds      === nothing || s.bounds      == bounds      || continue
        noise       === nothing || s.noise       == noise       || continue
        parallel    === nothing || s.parallel    == parallel    || continue
        qoc         === nothing || s.qoc         == qoc         || continue
        generic     === nothing || s.generic     == generic     || continue
        open_system === nothing || s.open_system == open_system || continue
        push!(out, name)
    end
    return sort(out)
end

"""
    is_registered(name::Symbol) -> Bool

Return `true` if `name` exists in [`OPTIMIZER_REGISTRY`](@ref).
"""
is_registered(name::Symbol)::Bool = haskey(OPTIMIZER_REGISTRY, name)

# ─────────────────────────────────────────────────────────────────────────────
# Built-in registrations
# ─────────────────────────────────────────────────────────────────────────────
# These describe the optimizer entry points that already exist in PULSAR.
# Each module owns its registration (callable + supports descriptor); this
# block populates the registry with the canonical names used by
# `recommend_optimizer` / `auto_optimize` / `_optimcon_generic`.
#
# The `callable` slot holds the **highest-level** entry point — i.e. the
# one that takes `(system, target, controls; config)` for QOC-aware
# optimizers and `(f, grad!, θ0; kwargs)` for generic ones.  Application
# wrappers that need a different signature are free to look up the
# `OptimizerEntry`, then dispatch on `entry.supports.qoc` /
# `entry.supports.generic` to pick a calling convention.

# Gradient-based, QOC interface.
# `open_system=true` is set for algorithms reachable on
# `LindbladMRControl` / `LindbladQCControl` via the application-layer
# kernel dispatch (`_mr_kernel` / `_qc_kernel`).
register_optimizer!(:grape,  grape_optimize,
    "GRadient Ascent Pulse Engineering (analytic gradient, fixed step or Adam).";
    gradient = true, qoc = true, parallel = true, generic = false,
    open_system = true)

register_optimizer!(:bfgs,   bfgs_optimize,
    "Broyden-Fletcher-Goldfarb-Shanno quasi-Newton (full-matrix Hessian approx).";
    gradient = true, hessian = true, qoc = true,
    open_system = false)

register_optimizer!(:lbfgs,  lbfgs_optimize,
    "Limited-memory BFGS (memory-efficient quasi-Newton).";
    gradient = true, hessian = true, qoc = true,
    open_system = true)

register_optimizer!(:newton, newton_optimize,
    "Newton-CG with Hessian-free linear solves.";
    gradient = true, hessian = true, qoc = true,
    open_system = false)

# Derivative-free, QOC interface
register_optimizer!(:cmaes, cmaes_optimize,
    "Covariance Matrix Adaptation Evolution Strategy (gradient-free, robust).";
    noise = true, qoc = true, generic = true, parallel = true,
    open_system = true)

register_optimizer!(:nelder_mead, nelder_mead_optimize,
    "Nelder-Mead simplex (gradient-free, low-dimensional).";
    qoc = true, generic = true,
    open_system = true)

register_optimizer!(:pso, pso_optimize,
    "Particle Swarm Optimization (gradient-free, global).";
    noise = true, qoc = true, generic = true, parallel = true,
    open_system = true)

# Constraint / robustness wrappers
register_optimizer!(:constrained_grape, constrained_optimize,
    "GRAPE under augmented-Lagrangian / penalty constraint handling.";
    gradient = true, bounds = true, qoc = true,
    open_system = false)

register_optimizer!(:robust_grape, robust_optimize,
    "Sample-averaged robust optimization over a parametric ensemble.";
    gradient = true, noise = true, parallel = true, qoc = true,
    open_system = false)

register_optimizer!(:trust_region, trust_region_optimize,
    "Trust-region Newton with Cauchy / dogleg subproblem solver.";
    gradient = true, hessian = true, qoc = true,
    open_system = false)
