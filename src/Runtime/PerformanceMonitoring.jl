"""
    PerformanceMonitoring.jl

Runtime performance monitoring and profiling utilities for Pulsar optimizations.

Provides a lightweight `PerformanceMonitor` struct that accumulates per-iteration
diagnostics (fidelity, gradient norm, wall-clock time, memory usage) and exposes
helpers for:

  - Progress display during long runs
  - Summary statistics after completion
  - Stagnation detection
  - ETA estimation
  - Memory tracking via `Base.gc_live_bytes()`

Usage example:

    monitor = PerformanceMonitor()
    for iter in 1:max_iter
        t0 = time()
        F  = compute_fidelity(...)
        gn = norm(compute_gradient(...))
        record_iteration!(monitor, iter, F, gn, time() - t0)
        print_progress(monitor, iter, max_iter)
        detect_stagnation(monitor) && break
    end
    print_summary(monitor)
"""

using Dates
using Statistics
using Printf

# ============================================================================
# PerformanceMonitor type
# ============================================================================

"""
    PerformanceMonitor

Mutable container that accumulates per-iteration performance data during
a Pulsar optimization run.

# Fields
- `start_time::Float64`                — `time()` value at construction
- `iteration_times::Vector{Float64}`   — elapsed wall-clock time per iteration (s)
- `fidelity_history::Vector{Float64}`  — fidelity F ∈ [0,1] at each iteration
- `gradient_norm_history::Vector{Float64}` — Frobenius norm ‖∇F‖ per iteration
- `memory_usage_mb::Vector{Float64}`   — estimated heap usage in MB per iteration
- `backend_used::Vector{Symbol}`       — backend symbol (e.g. `:cpu`, `:cuda`) per iter
- `n_fidelity_calls::Int`              — total fidelity evaluations recorded
- `n_gradient_calls::Int`              — total gradient evaluations recorded
- `n_propagator_calls::Int`            — total propagator evaluations recorded

# Construction
Use the zero-argument constructor `PerformanceMonitor()`.
"""
mutable struct PerformanceMonitor
    start_time              :: Float64
    iteration_times         :: Vector{Float64}
    fidelity_history        :: Vector{Float64}
    gradient_norm_history   :: Vector{Float64}
    memory_usage_mb         :: Vector{Float64}
    backend_used            :: Vector{Symbol}
    n_fidelity_calls        :: Int
    n_gradient_calls        :: Int
    n_propagator_calls      :: Int
end

"""
    PerformanceMonitor() -> PerformanceMonitor

Construct a new `PerformanceMonitor` with empty history vectors and the
current wall-clock time recorded as the start time.

# Example
```julia
monitor = PerformanceMonitor()
```
"""
function PerformanceMonitor()
    return PerformanceMonitor(
        time(),              # start_time
        Float64[],           # iteration_times
        Float64[],           # fidelity_history
        Float64[],           # gradient_norm_history
        Float64[],           # memory_usage_mb
        Symbol[],            # backend_used
        0,                   # n_fidelity_calls
        0,                   # n_gradient_calls
        0,                   # n_propagator_calls
    )
end

# ============================================================================
# Recording
# ============================================================================

"""
    record_iteration!(monitor::PerformanceMonitor, iter::Int,
                       fidelity::Float64, gradient_norm::Float64,
                       iter_time::Float64, backend::Symbol = :cpu)

Record one iteration's worth of performance data into `monitor`.

# Arguments
- `monitor`       — the `PerformanceMonitor` to update (modified in-place)
- `iter`          — iteration index (used for sanity checking, not stored separately)
- `fidelity`      — fidelity value at this iteration (should be in [0, 1])
- `gradient_norm` — Frobenius norm of the gradient ‖∇F‖_F
- `iter_time`     — elapsed time for this iteration in seconds
- `backend`       — backend symbol used (`:cpu`, `:cuda`, `:metal`, `:hybrid`)

# Side effects
- Appends to all history vectors.
- Samples current heap usage via `get_memory_usage_mb()`.
- Increments `n_fidelity_calls` and `n_gradient_calls` by 1 each.

# Example
```julia
t0 = time()
F  = compute_fidelity(U, tgt)
gn = norm(compute_grape_gradient(sys, seq, tgt))
record_iteration!(monitor, iter, F, gn, time() - t0)
```
"""
function record_iteration!(
    monitor        :: PerformanceMonitor,
    iter           :: Int,
    fidelity       :: Float64,
    gradient_norm  :: Float64,
    iter_time      :: Float64,
    backend        :: Symbol = :cpu
)
    push!(monitor.iteration_times,       iter_time)
    push!(monitor.fidelity_history,      fidelity)
    push!(monitor.gradient_norm_history, gradient_norm)
    push!(monitor.memory_usage_mb,       get_memory_usage_mb())
    push!(monitor.backend_used,          backend)
    monitor.n_fidelity_calls   += 1
    monitor.n_gradient_calls   += 1
    return nothing
end

"""
    record_propagator_call!(monitor::PerformanceMonitor, n::Int = 1)

Increment the propagator call counter by `n`.  Call this whenever
`compute_propagator` or a batch propagator is invoked.

# Example
```julia
record_propagator_call!(monitor)    # one propagator call
record_propagator_call!(monitor, 5) # five propagator calls
```
"""
function record_propagator_call!(monitor::PerformanceMonitor, n::Int = 1)
    monitor.n_propagator_calls += n
    return nothing
end

# ============================================================================
# Summary statistics
# ============================================================================

"""
    get_summary(monitor::PerformanceMonitor) -> Dict{String, Any}

Compute and return a summary dictionary of optimization performance metrics.

# Returned keys
| Key                    | Type      | Description                                        |
|:-----------------------|:----------|:---------------------------------------------------|
| `"total_time_seconds"` | `Float64` | Total elapsed wall-clock time since construction   |
| `"n_iterations"`       | `Int`     | Number of recorded iterations                      |
| `"avg_iter_time"`      | `Float64` | Mean iteration time (s); `NaN` if no iters         |
| `"min_iter_time"`      | `Float64` | Minimum iteration time (s)                         |
| `"max_iter_time"`      | `Float64` | Maximum iteration time (s)                         |
| `"best_fidelity"`      | `Float64` | Maximum fidelity achieved                          |
| `"final_fidelity"`     | `Float64` | Fidelity at the last recorded iteration            |
| `"n_fidelity_calls"`   | `Int`     | Total fidelity evaluations                         |
| `"n_gradient_calls"`   | `Int`     | Total gradient evaluations                         |
| `"n_propagator_calls"` | `Int`     | Total propagator evaluations                       |
| `"memory_peak_mb"`     | `Float64` | Peak observed heap usage (MB)                      |
| `"avg_memory_mb"`      | `Float64` | Average heap usage (MB)                            |
| `"convergence_rate"`   | `Float64` | Estimated slope of log₁₀(1 − F) vs iteration      |

The `"convergence_rate"` is computed from a linear fit of `log10.(1 .- F)` vs
iteration number over the last half of recorded iterations.  A negative value
indicates the infidelity is decreasing (desired).

# Example
```julia
summary = get_summary(monitor)
println("Best fidelity: ", summary["best_fidelity"])
```
"""
function get_summary(monitor::PerformanceMonitor)::Dict{String, Any}
    n = length(monitor.fidelity_history)

    total_time = time() - monitor.start_time

    # Timing statistics
    if n > 0
        avg_iter = mean(monitor.iteration_times)
        min_iter = minimum(monitor.iteration_times)
        max_iter = maximum(monitor.iteration_times)
    else
        avg_iter = NaN
        min_iter = NaN
        max_iter = NaN
    end

    # Fidelity statistics
    best_F  = n > 0 ? maximum(monitor.fidelity_history) : 0.0
    final_F = n > 0 ? monitor.fidelity_history[end]     : 0.0

    # Memory statistics
    mem_peak = isempty(monitor.memory_usage_mb) ? 0.0 : maximum(monitor.memory_usage_mb)
    mem_avg  = isempty(monitor.memory_usage_mb) ? 0.0 : mean(monitor.memory_usage_mb)

    # Convergence rate: slope of log10(1 - F) vs iteration (last half)
    conv_rate = _estimate_convergence_rate(monitor.fidelity_history)

    return Dict{String, Any}(
        "total_time_seconds" => total_time,
        "n_iterations"       => n,
        "avg_iter_time"      => avg_iter,
        "min_iter_time"      => min_iter,
        "max_iter_time"      => max_iter,
        "best_fidelity"      => best_F,
        "final_fidelity"     => final_F,
        "n_fidelity_calls"   => monitor.n_fidelity_calls,
        "n_gradient_calls"   => monitor.n_gradient_calls,
        "n_propagator_calls" => monitor.n_propagator_calls,
        "memory_peak_mb"     => mem_peak,
        "avg_memory_mb"      => mem_avg,
        "convergence_rate"   => conv_rate,
    )
end

# ============================================================================
# Progress display
# ============================================================================

"""
    print_progress(monitor::PerformanceMonitor, iter::Int, max_iter::Int;
                    prefix::String = "")

Print a single-line progress report to stdout.

Output format:
```
Iter  150/1000 | F=0.984532 | |∇|=2.31e-03 | 0.023s/iter | ETA: 19s
```

# Arguments
- `monitor`   — performance monitor with recorded history
- `iter`      — current iteration number (for the left-hand side counter)
- `max_iter`  — total planned iterations (for the right-hand side counter)
- `prefix`    — optional string prepended to the output line (e.g. `"[GRAPE] "`)

# Notes
- If no iterations have been recorded yet, a placeholder line is printed.
- The ETA estimate uses `estimate_eta`.
"""
function print_progress(
    monitor  :: PerformanceMonitor,
    iter     :: Int,
    max_iter :: Int;
    prefix   :: String = ""
)
    n = length(monitor.fidelity_history)

    if n == 0
        @printf("%sIter %5d/%-5d | (no data yet)\n", prefix, iter, max_iter)
        return
    end

    F    = monitor.fidelity_history[end]
    gn   = isempty(monitor.gradient_norm_history) ? NaN :
           monitor.gradient_norm_history[end]
    tavg = mean(monitor.iteration_times)
    eta  = estimate_eta(monitor, iter, max_iter)

    if eta >= 3600
        eta_str = @sprintf("%.1fh", eta / 3600)
    elseif eta >= 60
        eta_str = @sprintf("%.1fm", eta / 60)
    else
        eta_str = @sprintf("%.0fs", eta)
    end

    @printf("%sIter %5d/%-5d | F=%.6f | |∇|=%.2e | %.3fs/iter | ETA: %s\n",
            prefix, iter, max_iter, F, gn, tavg, eta_str)
    return nothing
end

# ============================================================================
# Full summary printout
# ============================================================================

"""
    print_summary(monitor::PerformanceMonitor)

Pretty-print a complete optimization performance summary to stdout.

Displays:
- Total wall-clock time
- Number of iterations and calls
- Fidelity statistics (best, final)
- Iteration timing statistics (avg, min, max)
- Memory usage statistics
- Estimated convergence rate

# Example
```julia
print_summary(monitor)
```
"""
function print_summary(monitor::PerformanceMonitor)
    s = get_summary(monitor)
    n = s["n_iterations"]

    println()
    println("=" ^ 60)
    println("  Pulsar Optimization Performance Summary")
    println("=" ^ 60)

    @printf("  Total wall time : %.3f s\n",  s["total_time_seconds"])
    @printf("  Iterations      : %d\n",       n)
    @printf("  Fidelity calls  : %d\n",       s["n_fidelity_calls"])
    @printf("  Gradient calls  : %d\n",       s["n_gradient_calls"])
    @printf("  Propagator calls: %d\n",       s["n_propagator_calls"])
    println()

    if n > 0
        @printf("  Best fidelity   : %.8f\n",  s["best_fidelity"])
        @printf("  Final fidelity  : %.8f\n",  s["final_fidelity"])
        @printf("  Infidelity (1-F): %.3e\n",  1.0 - s["final_fidelity"])
        println()
        @printf("  Avg iter time   : %.4f s\n", s["avg_iter_time"])
        @printf("  Min iter time   : %.4f s\n", s["min_iter_time"])
        @printf("  Max iter time   : %.4f s\n", s["max_iter_time"])
        println()
        @printf("  Peak memory     : %.2f MB\n", s["memory_peak_mb"])
        @printf("  Avg memory      : %.2f MB\n", s["avg_memory_mb"])
        println()
        cr = s["convergence_rate"]
        if isnan(cr)
            println("  Convergence rate: (insufficient data)")
        else
            @printf("  Conv. rate      : %.4f decades / iter\n", cr)
            if cr < 0
                println("  Conv. trend     : decreasing infidelity (good)")
            else
                println("  Conv. trend     : stagnant or diverging")
            end
        end
    else
        println("  (No iterations recorded)")
    end

    println("=" ^ 60)
    println()
    return nothing
end

# ============================================================================
# Stagnation detection
# ============================================================================

"""
    detect_stagnation(monitor::PerformanceMonitor;
                       window::Int = 50, threshold::Float64 = 1e-8) -> Bool

Return `true` if the optimization has stagnated, defined as the range of
fidelity values over the last `window` iterations being smaller than `threshold`.

# Arguments
- `monitor`   — performance monitor with recorded history
- `window`    — number of recent iterations to examine (default 50)
- `threshold` — minimum fidelity improvement to be considered progress (default 1e-8)

# Returns
`true` if stagnated, `false` otherwise or if fewer than `window` iterations
have been recorded.

# Example
```julia
if detect_stagnation(monitor; window=30, threshold=1e-7)
    @warn "Optimization stagnated — stopping early"
    break
end
```
"""
function detect_stagnation(
    monitor   :: PerformanceMonitor;
    window    :: Int     = 50,
    threshold :: Float64 = 1e-8
)::Bool
    n = length(monitor.fidelity_history)
    n < window && return false

    recent = monitor.fidelity_history[(end - window + 1):end]
    improvement = maximum(recent) - minimum(recent)
    return improvement < threshold
end

# ============================================================================
# Memory usage
# ============================================================================

"""
    get_memory_usage_mb() -> Float64

Estimate the current Julia heap usage in megabytes.

Uses `Base.gc_live_bytes()` which returns the number of bytes of live heap
objects known to the GC.  This is a lower bound on actual memory consumption
(it excludes memory allocated outside the GC, e.g. BLAS workspaces) but is
available without any additional packages.

# Returns
Heap usage in MB as a `Float64`.

# Notes
The returned value reflects live objects at the time of the call.  For a
more accurate peak measurement call `GC.gc()` first (at the cost of a GC pause).

# Example
```julia
mem_mb = get_memory_usage_mb()
println("Current heap usage: \$(mem_mb) MB")
```
"""
function get_memory_usage_mb()::Float64
    return Base.gc_live_bytes() / 1024^2
end

# ============================================================================
# ETA estimation
# ============================================================================

"""
    estimate_eta(monitor::PerformanceMonitor, current_iter::Int, max_iter::Int)
    -> Float64

Estimate the remaining time to completion in seconds, based on the average
iteration time recorded so far.

# Arguments
- `monitor`      — performance monitor with recorded iteration times
- `current_iter` — index of the most recently completed iteration (1-based)
- `max_iter`     — total planned number of iterations

# Returns
Estimated seconds remaining.  Returns `Inf` if no iterations have been
recorded yet, or `0.0` if `current_iter >= max_iter`.

# Formula
    ETA = avg_iter_time × (max_iter - current_iter)

# Example
```julia
eta = estimate_eta(monitor, 150, 1000)   # → remaining time in seconds
```
"""
function estimate_eta(
    monitor      :: PerformanceMonitor,
    current_iter :: Int,
    max_iter     :: Int
)::Float64
    remaining = max(0, max_iter - current_iter)
    remaining == 0 && return 0.0

    n = length(monitor.iteration_times)
    n == 0 && return Inf

    avg_time = mean(monitor.iteration_times)
    return avg_time * remaining
end

# ============================================================================
# Backend usage summary
# ============================================================================

"""
    backend_usage_summary(monitor::PerformanceMonitor) -> Dict{Symbol, Int}

Return a count of how many iterations were executed on each backend.

# Returns
Dictionary mapping backend symbol to iteration count, e.g.
`Dict(:cpu => 950, :cuda => 50)`.

# Example
```julia
usage = backend_usage_summary(monitor)
println("CPU iterations: ", get(usage, :cpu, 0))
```
"""
function backend_usage_summary(monitor::PerformanceMonitor)::Dict{Symbol, Int}
    counts = Dict{Symbol, Int}()
    for b in monitor.backend_used
        counts[b] = get(counts, b, 0) + 1
    end
    return counts
end

# ============================================================================
# Fidelity statistics helpers
# ============================================================================

"""
    fidelity_at_time(monitor::PerformanceMonitor, elapsed_seconds::Float64)
    -> Union{Float64, Nothing}

Return the fidelity value at approximately `elapsed_seconds` seconds into the
optimization, by interpolating the cumulative iteration time axis.

Returns `nothing` if `elapsed_seconds` exceeds the total recorded time.

# Example
```julia
F_at_1s = fidelity_at_time(monitor, 1.0)   # fidelity after ~1 second
```
"""
function fidelity_at_time(
    monitor         :: PerformanceMonitor,
    elapsed_seconds :: Float64
)::Union{Float64, Nothing}
    n = length(monitor.fidelity_history)
    n == 0 && return nothing

    cumulative = cumsum(monitor.iteration_times)
    idx = searchsortedfirst(cumulative, elapsed_seconds)

    idx > n && return nothing
    return monitor.fidelity_history[idx]
end

"""
    time_to_fidelity(monitor::PerformanceMonitor, target_fidelity::Float64)
    -> Union{Float64, Nothing}

Return the approximate elapsed time (in seconds) at which the fidelity first
exceeded `target_fidelity`, or `nothing` if it was never reached.

# Example
```julia
t = time_to_fidelity(monitor, 0.99)
isnothing(t) ? println("0.99 not reached") : println("Reached 0.99 at \$(t) s")
```
"""
function time_to_fidelity(
    monitor          :: PerformanceMonitor,
    target_fidelity  :: Float64
)::Union{Float64, Nothing}
    n = length(monitor.fidelity_history)
    n == 0 && return nothing

    cumulative = cumsum(monitor.iteration_times)
    for i in 1:n
        if monitor.fidelity_history[i] >= target_fidelity
            return cumulative[i]
        end
    end
    return nothing
end

# ============================================================================
# Base.show
# ============================================================================

function Base.show(io::IO, m::PerformanceMonitor)
    n   = length(m.fidelity_history)
    tot = time() - m.start_time
    F   = n > 0 ? m.fidelity_history[end] : NaN
    @printf(io, "PerformanceMonitor(%d iters, F=%.6f, %.2f s elapsed)",
            n, F, tot)
end

# ============================================================================
# Internal helpers
# ============================================================================

"""
    _estimate_convergence_rate(fidelity_history::Vector{Float64}) -> Float64

Estimate the convergence rate as the linear regression slope of
log₁₀(1 − F(k)) vs iteration index k over the last half of the history.

Returns `NaN` if fewer than 4 data points are available or all infidelities
are zero (perfect convergence).
"""
function _estimate_convergence_rate(fidelity_history::Vector{Float64})::Float64
    n = length(fidelity_history)
    n < 4 && return NaN

    # Use the second half of the history to capture asymptotic behaviour
    half   = max(2, n ÷ 2)
    F_tail = fidelity_history[half:end]
    k_tail = Float64.(half:n)

    # Compute log10(1 - F), clamping to avoid -Inf from F = 1
    infid  = max.(1.0 .- F_tail, 1e-16)
    log_id = log10.(infid)

    # Check for degenerate data
    all(isinf, log_id) && return NaN
    all(isnan, log_id) && return NaN

    # Remove any Inf / NaN points
    valid  = .!isinf.(log_id) .& .!isnan.(log_id)
    sum(valid) < 2 && return NaN

    x = k_tail[valid]
    y = log_id[valid]

    # Linear regression slope: slope = Cov(x,y) / Var(x)
    x_mean = mean(x)
    y_mean = mean(y)
    num    = sum((x .- x_mean) .* (y .- y_mean))
    den    = sum((x .- x_mean) .^ 2)

    den < eps(Float64) && return NaN
    return num / den
end
