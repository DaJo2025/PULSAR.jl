"""
    comparisons/Drivers/simpson_driver.jl

Driver for SIMPSON (C binary, version 4.2.x).

Workflow:
  1. Emit a native SIMPSON .in script from the `PhysicsAnnotation` via
     [`emit_simpson`](@ref).
  2. Run the SIMPSON binary in a temp workdir.
  3. Parse the optimised shape (raw Bx/By in Hz) via
     [`parse_waveform_file`](@ref) with `convention = :hz`.
  4. Re-evaluate fidelity through PULSAR's canonical
     [`grape_state_kernel`](@ref) — this is the number reported in the table.

SIMPSON's internal fidelity is only used to drive its own L-BFGS; we don't
attempt to match it.

The driver has no per-problem branching — every benchmark with a compatible
annotation emits a native script from the same template.  Capability gating
is declared in `SIMPSON_CAPABILITIES` inside `SIMPSONEmitter.jl`.
"""

include(joinpath(@__DIR__, "..", "Translator", "Emitters", "SIMPSONEmitter.jl"))

struct SIMPSONDriver <: AbstractSolverDriver end

function run_driver(driver::SIMPSONDriver, problem::BenchmarkProblem)
    driver_name = "SIMPSON/GRAPE"
    problem_id  = problem.id

    if problem.physics === nothing
        return not_available_result(driver_name, problem_id,
            "Problem has no PhysicsAnnotation — SIMPSON driver needs annotation to emit.")
    end
    reason = check_supported(SIMPSON_CAPABILITIES, problem.physics)
    reason === nothing ||
        return not_available_result(driver_name, problem_id, reason)
    if problem.physics.target.target_per_drift
        return not_available_result(driver_name, problem_id,
            "SIMPSON emitter does not support target_per_drift (annotation pairs targets to drifts 1:1).")
    end

    simpson_bin = Sys.which("simpson")
    if isnothing(simpson_bin)
        candidate = "/usr/local/bin/simpson"
        if isfile(candidate)
            simpson_bin = candidate
        else
            return not_available_result(driver_name, problem_id,
                "SIMPSON binary not found on PATH. Install from http://inano.au.dk/about/nmr-software/simpson/")
        end
    end

    try
        return _run_simpson(problem, driver_name, simpson_bin)
    catch err
        return error_result(driver_name, problem_id, err)
    end
end

function _run_simpson(problem::BenchmarkProblem, driver_name::String, simpson_bin::String)
    ann     = problem.physics
    ctrl    = problem.ctrl
    workdir = mktempdir()
    log_out = joinpath(workdir, "simpson.out.log")
    log_err = joinpath(workdir, "simpson.err.log")

    script_path, shape_path = emit_simpson(ann, workdir; problem_id=problem.id)

    t_start = time()
    run_subprocess(simpson_bin, [script_path], workdir;
                    stdout_file=log_out, stderr_file=log_err)
    t_total = time() - t_start

    pwr_hz = ann.controls[1].pwr_max_hz    # shared across channels
    waveform = parse_waveform_file(shape_path,
                                    length(ann.controls), ann.n_time_steps;
                                    convention = :hz,
                                    pwr_levels = [2π * pwr_hz])
    clamp!(waveform, ctrl.l_bound, ctrl.u_bound)

    fidelity = canonical_rescore(waveform, ctrl)

    return BenchmarkResult(
        driver_name, problem.id,
        fidelity, t_total, ann.max_iter, false,
        waveform, Float64[],
        true, "",
        Dict{String,Any}("backend"     => "SIMPSON C binary",
                         "script_path" => script_path,
                         "log"         => log_out),
    )
end
