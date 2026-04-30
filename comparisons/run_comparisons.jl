"""
    comparisons/run_comparisons.jl

Top-level entry point for the PULSAR cross-package optimal control benchmark.

Usage:
    julia --project=. comparisons/run_comparisons.jl
    julia --project=. comparisons/run_comparisons.jl --problems P_BB180,P_INEPT
    julia --project=. comparisons/run_comparisons.jl --packages PULSAR,QuantumControl
    julia --project=. comparisons/run_comparisons.jl --problems P_BB180 --packages PULSAR

CLI flags:
    --problems  <comma-separated list of problem IDs>   (default: all)
    --packages  <comma-separated list of packages> (default: all)

Available packages / driver names:
    PULSAR_lbfgs, PULSAR_cmaes, PULSAR_grape, PULSAR_lbfgsb, PULSAR_cg
    QuantumControl, Krotov, QuTiP, qopt, Spinach, SIMPSON, Quandary
"""

# ─── Parse CLI arguments ──────────────────────────────────────────────────────

function _parse_args(args)
    requested_problems = String[]
    requested_packages = String[]

    i = 1
    while i <= length(args)
        if args[i] == "--problems" && i < length(args)
            requested_problems = strip.(split(args[i+1], ","))
            i += 2
        elseif args[i] == "--packages" && i < length(args)
            requested_packages = strip.(split(args[i+1], ","))
            i += 2
        else
            i += 1
        end
    end

    return requested_problems, requested_packages
end

# ─── Load PULSAR ──────────────────────────────────────────────────────────────

# Ensure we're loading from the package root
_pulsar_root = dirname(dirname(@__FILE__))
if !in(_pulsar_root, LOAD_PATH)
    push!(LOAD_PATH, _pulsar_root)
end

using PULSAR
using Random
using Printf
using Dates

# ─── Load all driver and problem files ───────────────────────────────────────

@isdefined(_COMPARISONS_DIR) || (const _COMPARISONS_DIR = @__DIR__)

include(joinpath(_COMPARISONS_DIR, "Translator", "Translator.jl"))

# ─── Problem registry ────────────────────────────────────────────────────────
# Benchmark problems are not bundled with the public release. To run a
# comparison, define one or more `BenchmarkProblem`s and add them to
# `ALL_PROBLEMS` below (see `comparisons/README.md` for an example).
const _USER_PROBLEMS_FILE = joinpath(_COMPARISONS_DIR, "Problems", "all_problems.jl")
if isfile(_USER_PROBLEMS_FILE)
    include(_USER_PROBLEMS_FILE)
end

include(joinpath(_COMPARISONS_DIR, "Drivers", "driver_interface.jl"))
include(joinpath(_COMPARISONS_DIR, "Drivers", "pulsar_driver.jl"))
include(joinpath(_COMPARISONS_DIR, "Drivers", "quantumcontrol_driver.jl"))
include(joinpath(_COMPARISONS_DIR, "Drivers", "krotov_driver.jl"))
include(joinpath(_COMPARISONS_DIR, "Drivers", "qutip_driver.jl"))
include(joinpath(_COMPARISONS_DIR, "Drivers", "qopt_driver.jl"))
include(joinpath(_COMPARISONS_DIR, "Drivers", "spinach_driver.jl"))
include(joinpath(_COMPARISONS_DIR, "Drivers", "simpson_driver.jl"))
include(joinpath(_COMPARISONS_DIR, "Drivers", "quandary_driver.jl"))
include(joinpath(_COMPARISONS_DIR, "Report", "report.jl"))

# ─── Driver registry ─────────────────────────────────────────────────────────

@isdefined(ALL_DRIVERS) || (const ALL_DRIVERS = Dict{String, AbstractSolverDriver}(
    "PULSAR_lbfgs"     => PULSARDriver(:lbfgs),
    "PULSAR_cmaes"     => PULSARDriver(:cmaes),
    "PULSAR_grape"     => PULSARDriver(:grape),
    "PULSAR_lbfgsb"    => PULSARDriver(:lbfgsb),
    "PULSAR_cg"        => PULSARDriver(:cg),
    "QuantumControl"   => QuantumControlDriver(),
    "Krotov"           => KrotovDriver(),
    "QuTiP"            => QuTiPDriver(),
    "qopt"             => QoptDriver(),
    "Spinach"          => SpinachDriver(),
    "SIMPSON"          => SIMPSONDriver(),
    "Quandary"         => QuandaryDriver(),
))

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    requested_problem_ids, requested_package_keys = _parse_args(ARGS)

    if !@isdefined(ALL_PROBLEMS) || isempty(ALL_PROBLEMS)
        @warn "No benchmark problems registered. " *
              "Define `ALL_PROBLEMS::Vector{BenchmarkProblem}` " *
              "(see comparisons/README.md) before running."
        return
    end

    # Select problems
    problems = if isempty(requested_problem_ids)
        ALL_PROBLEMS
    else
        filter(p -> p.id in requested_problem_ids, ALL_PROBLEMS)
    end

    isempty(problems) && (
        @warn "No matching problems found. Available: $(join([p.id for p in ALL_PROBLEMS], ", "))";
        return
    )

    # Select drivers
    driver_keys = if isempty(requested_package_keys)
        sort(collect(keys(ALL_DRIVERS)))
    else
        requested_package_keys
    end

    # Validate driver keys
    unknown = setdiff(driver_keys, keys(ALL_DRIVERS))
    !isempty(unknown) && @warn "Unknown driver(s): $(join(unknown, ", "))"
    driver_keys = filter(k -> haskey(ALL_DRIVERS, k), driver_keys)

    # ── Print run header ──────────────────────────────────────────────────────
    println()
    println("  PULSAR Cross-Package Optimal Control Benchmark")
    println("  Pulse Design Library for Spin Control Algorithms and Rollout")
    println("  Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("  Julia $(VERSION)  |  $(Threads.nthreads()) thread(s)")
    println()
    @printf("  Problems : %s\n", join([p.id for p in problems], ", "))
    @printf("  Drivers  : %s\n", join(driver_keys, ", "))
    println()

    # ── Run all (driver, problem) pairs ──────────────────────────────────────
    results = BenchmarkResult[]

    for problem in problems
        @printf("  ── Problem %s ──\n", problem.id)
        for key in driver_keys
            driver = ALL_DRIVERS[key]
            @printf("    Running %-20s ... ", key)
            flush(stdout)

            t0 = time()
            r  = run_driver(driver, problem)
            dt = time() - t0

            if !r.available
                @printf("NOT AVAILABLE\n")
            elseif haskey(r.metadata, "error")
                @printf("ERROR: %s\n", r.metadata["error"])
            else
                @printf("F=%.4f  t=%.2f s  iter=%d  %s\n",
                    r.fidelity, r.wall_time_s, r.n_iterations,
                    r.converged ? "converged" : "max_iter")
            end

            push!(results, r)
        end
        println()
    end

    # ── Print detailed report ─────────────────────────────────────────────────
    print_report(results)

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary_table(results)

    # ── Save JSON results ─────────────────────────────────────────────────────
    ts        = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    json_path = joinpath(_COMPARISONS_DIR, "Results", "results_$(ts).json")
    save_results_json(results, json_path)

    return results
end

# ─── Entry point ─────────────────────────────────────────────────────────────

main()
