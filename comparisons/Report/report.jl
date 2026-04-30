"""
    comparisons/Report/report.jl

Plain-text comparison table printer for Pulsar benchmark results.

Functions:
  print_report(results)        — one detailed table per benchmark problem
  print_summary_table(results) — one-line-per-package summary across all problems
"""

# ─── Column widths ────────────────────────────────────────────────────────────

@isdefined(_COL_DRIVER)   || (const _COL_DRIVER   = 28)
@isdefined(_COL_FIDELITY) || (const _COL_FIDELITY = 10)
@isdefined(_COL_TIME)     || (const _COL_TIME     = 10)
@isdefined(_COL_ITERS)    || (const _COL_ITERS    = 8)
@isdefined(_COL_STATUS)   || (const _COL_STATUS   = 16)

@isdefined(_TABLE_WIDTH)  || (const _TABLE_WIDTH = _COL_DRIVER + _COL_FIDELITY + _COL_TIME + _COL_ITERS + _COL_STATUS + 8)

# ─── Helpers ──────────────────────────────────────────────────────────────────

_hr_double() = "═" ^ _TABLE_WIDTH
_hr_single() = "─" ^ _TABLE_WIDTH

function _status_str(r::BenchmarkResult)::String
    !r.available  && return "NOT AVAILABLE"
    r.fidelity == 0.0 && haskey(r.metadata, "error") && return "ERROR"
    r.converged   && return "converged"
    return "max_iter"
end

function _row(r::BenchmarkResult)::String
    if !r.available
        # Show full-width unavailable message
        msg = "$(lpad(r.driver_name, _COL_DRIVER))  NOT AVAILABLE — $(r.unavailable_msg)"
        return msg
    end
    fid_str  = @sprintf("%.4f", r.fidelity)
    time_str = if r.wall_time_s < 60
        @sprintf("%.2f s", r.wall_time_s)
    else
        @sprintf("%.1f min", r.wall_time_s / 60)
    end
    iter_str = string(r.n_iterations)
    stat_str = _status_str(r)

    return @sprintf("%-*s  %*s  %*s  %*s  %s",
        _COL_DRIVER,   r.driver_name,
        _COL_FIDELITY, fid_str,
        _COL_TIME,     time_str,
        _COL_ITERS,    iter_str,
        stat_str)
end

# ─── print_report ─────────────────────────────────────────────────────────────

"""
    print_report(results::Vector{BenchmarkResult})

Print one detailed comparison table per benchmark problem.
"""
function print_report(results::Vector{BenchmarkResult})
    # Group by problem_id
    problem_ids = unique(r.problem_id for r in results)

    for pid in sort(problem_ids)
        group = filter(r -> r.problem_id == pid, results)
        isempty(group) && continue

        # Find description from ALL_PROBLEMS if available
        desc = ""
        try
            idx = findfirst(p -> p.id == pid, ALL_PROBLEMS)
            !isnothing(idx) && (desc = ALL_PROBLEMS[idx].description)
        catch; end

        # Find fidelity target
        tgt = 0.0
        try
            idx = findfirst(p -> p.id == pid, ALL_PROBLEMS)
            !isnothing(idx) && (tgt = ALL_PROBLEMS[idx].fidelity_target)
        catch; end

        println()
        println(_hr_double())
        if isempty(desc)
            @printf("  %-s  |  target ≥ %.2f\n", pid, tgt)
        else
            @printf("  %-s  %-s  |  target ≥ %.2f\n", pid, desc, tgt)
        end
        println(_hr_double())

        @printf("  %-*s  %*s  %*s  %*s  %s\n",
            _COL_DRIVER,   "Package/Method",
            _COL_FIDELITY, "Fidelity",
            _COL_TIME,     "Time",
            _COL_ITERS,    "Iters",
            "Status")
        println(_hr_single())

        for r in group
            println("  " * _row(r))
        end
        println(_hr_single())
    end
    println()
end

# ─── print_summary_table ──────────────────────────────────────────────────────

"""
    print_summary_table(results::Vector{BenchmarkResult})

Print one summary line per driver across all problems.
Shows: driver name, mean fidelity, best fidelity, problems solved (above target),
total wall time.
"""
function print_summary_table(results::Vector{BenchmarkResult})
    drivers = unique(r.driver_name for r in results)

    println()
    println("  Summary across all benchmark problems")
    println("  " * "═"^76)
    @printf("  %-*s  %10s  %10s  %10s  %10s\n",
        _COL_DRIVER, "Package/Method",
        "MeanFid", "BestFid", "Solved", "TotalTime")
    println("  " * "─"^76)

    for dname in sort(drivers)
        group = filter(r -> r.driver_name == dname, results)
        avail = filter(r -> r.available, group)

        if isempty(avail)
            @printf("  %-*s  NOT AVAILABLE\n", _COL_DRIVER, dname)
            continue
        end

        fids = [r.fidelity for r in avail]
        mean_fid = sum(fids) / length(fids)
        best_fid = maximum(fids)
        t_total  = sum(r.wall_time_s for r in avail)

        # Count "solved" = fidelity >= target
        n_solved = 0
        for r in avail
            tgt = 0.0
            try
                idx = findfirst(p -> p.id == r.problem_id, ALL_PROBLEMS)
                !isnothing(idx) && (tgt = ALL_PROBLEMS[idx].fidelity_target)
            catch; end
            r.fidelity >= tgt && (n_solved += 1)
        end

        time_str = t_total < 3600 ? @sprintf("%.1f s", t_total) :
                                     @sprintf("%.1f h", t_total / 3600)

        @printf("  %-*s  %10.4f  %10.4f  %5d/%-4d  %10s\n",
            _COL_DRIVER, dname,
            mean_fid, best_fid,
            n_solved, length(avail),
            time_str)
    end
    println("  " * "─"^76)
    println()
end

# ─── save_results_json ────────────────────────────────────────────────────────

"""
    save_results_json(results, filename)

Save results to a JSON file via JSON3.jl (if available). Silently skips if
JSON3 is not installed.
"""
function save_results_json(results::Vector{BenchmarkResult}, filename::String)
    # Lazy-load JSON3 — skip silently if not installed
    JSON3 = try
        Base.require(Base.PkgId(Base.UUID("0f8b85d8-7e73-4b5b-a63d-d2c9b3a3b3a3"), "JSON3"))
    catch
        nothing
    end
    if isnothing(JSON3)
        JSON3 = try
            id = Base.identify_package("JSON3")
            isnothing(id) ? nothing : Base.require(id)
        catch
            nothing
        end
    end
    if isnothing(JSON3)
        @warn "JSON3.jl not available — skipping JSON output. Install with: ] add JSON3"
        return
    end

    # Serialise each result as a plain Dict (JSON3 handles Dict serialisation)
    data = map(results) do r
        Dict{String,Any}(
            "driver_name"      => r.driver_name,
            "problem_id"       => r.problem_id,
            "fidelity"         => r.fidelity,
            "wall_time_s"      => r.wall_time_s,
            "n_iterations"     => r.n_iterations,
            "converged"        => r.converged,
            "available"        => r.available,
            "unavailable_msg"  => r.unavailable_msg,
            "fidelity_history" => r.fidelity_history,
            "metadata"         => r.metadata,
        )
    end

    mkpath(dirname(filename))
    open(filename, "w") do f
        Base.invokelatest(JSON3.write, f, data)
    end
    @printf("Results saved to: %s\n", filename)
end
