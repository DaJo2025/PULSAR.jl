"""
    comparisons/Drivers/driver_interface.jl

Defines the abstract driver interface and the `BenchmarkResult` return type
used by every solver driver in the comparison suite.
"""

# ─── Abstract driver ──────────────────────────────────────────────────────────

"""
    AbstractSolverDriver

Abstract supertype for all solver drivers.  A concrete driver must implement:

    run_driver(driver::ConcreteDriver, problem::BenchmarkProblem) → BenchmarkResult

Drivers should never throw uncaught exceptions — wrap all external calls in
try/catch and return a failed `BenchmarkResult` on error.
"""
abstract type AbstractSolverDriver end

# ─── BenchmarkResult ──────────────────────────────────────────────────────────

"""
    BenchmarkResult

Result returned by every solver driver for a single (driver, problem) pair.

# Fields
- `driver_name`      — e.g. `"Pulsar/:lbfgs"` or `"QuantumControl/GRAPE"`
- `problem_id`       — matches `BenchmarkProblem.id`, e.g. `"BM01"`
- `fidelity`         — final ensemble-averaged fidelity ∈ [0,1]
- `wall_time_s`      — total wall-clock time in seconds
- `n_iterations`     — number of optimisation iterations completed
- `converged`        — `true` if the solver reported convergence
- `controls`         — optimised waveform `[n_ctrl × n_t]`, or `nothing`
- `fidelity_history` — per-iteration fidelity vector (may be empty)
- `available`        — `false` if the package is not installed
- `unavailable_msg`  — install hint shown in the report when `!available`
- `metadata`         — driver-specific key/value pairs
"""
struct BenchmarkResult
    driver_name       :: String
    problem_id        :: String
    fidelity          :: Float64
    wall_time_s       :: Float64
    n_iterations      :: Int
    converged         :: Bool
    controls          :: Union{Matrix{Float64}, Nothing}
    fidelity_history  :: Vector{Float64}
    available         :: Bool
    unavailable_msg   :: String
    metadata          :: Dict{String,Any}
end

# ─── Convenience constructor ──────────────────────────────────────────────────

"""
    not_available_result(driver_name, problem_id, msg) → BenchmarkResult

Create a `BenchmarkResult` representing a package that is not installed.
"""
function not_available_result(driver_name::String, problem_id::String, msg::String)
    return BenchmarkResult(
        driver_name,
        problem_id,
        0.0,           # fidelity
        0.0,           # wall_time_s
        0,             # n_iterations
        false,         # converged
        nothing,       # controls
        Float64[],     # fidelity_history
        false,         # available
        msg,           # unavailable_msg
        Dict{String,Any}(),
    )
end

"""
    error_result(driver_name, problem_id, err) → BenchmarkResult

Create a `BenchmarkResult` representing a run that threw an exception.
"""
function error_result(driver_name::String, problem_id::String, err)
    return BenchmarkResult(
        driver_name,
        problem_id,
        0.0,
        0.0,
        0,
        false,
        nothing,
        Float64[],
        true,          # package was available; it just errored
        "",
        Dict{String,Any}("error" => string(err)),
    )
end
