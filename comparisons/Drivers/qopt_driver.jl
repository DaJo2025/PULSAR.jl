"""
    comparisons/Drivers/qopt_driver.jl

Driver for `qopt` (Python package by Forschungszentrum Jülich).

Workflow:
  1. [`emit_qopt`](@ref) writes a native Python script driven by the
     [`PhysicsAnnotation`](@ref) — one `StateInfidelity` cost function per
     (drift × state-pair) combination, summed into a `SumOfCostFunctions`.
  2. Run `python <script>` as a subprocess.
  3. Parse the normalised waveform via [`parse_waveform_file`](@ref).
  4. Re-evaluate with PULSAR's [`grape_state_kernel`](@ref).

Installation:
    pip install qopt

Capabilities declared in `QOPT_CAPABILITIES` inside `QOptEmitter.jl`.
"""

include(joinpath(@__DIR__, "..", "Translator", "Emitters", "QOptEmitter.jl"))

struct QoptDriver <: AbstractSolverDriver end

function run_driver(driver::QoptDriver, problem::BenchmarkProblem)
    driver_name = "qopt/GRAPE"
    problem_id  = problem.id

    if problem.physics === nothing
        return not_available_result(driver_name, problem_id,
            "Problem has no PhysicsAnnotation — qopt driver needs annotation to emit.")
    end
    reason = check_supported(QOPT_CAPABILITIES, problem.physics)
    reason === nothing ||
        return not_available_result(driver_name, problem_id, reason)
    if problem.physics.target.target_per_drift
        return not_available_result(driver_name, problem_id,
            "qopt emitter does not support target_per_drift objectives.")
    end

    python_bin = _qopt_python_binary()
    python_bin === nothing && return not_available_result(driver_name, problem_id,
        "python3 not found on PATH.")

    try
        return _run_qopt(problem, driver_name, python_bin)
    catch err
        return error_result(driver_name, problem_id, err)
    end
end

function _qopt_python_binary()::Union{Nothing,String}
    for name in ("python3", "python")
        bin = Sys.which(name)
        bin === nothing || return bin
    end
    return nothing
end

function _run_qopt(problem::BenchmarkProblem, driver_name::String,
                    python_bin::String)
    workdir = mktempdir()
    log_out = joinpath(workdir, "qopt.out.log")
    log_err = joinpath(workdir, "qopt.err.log")

    script_path, waveform_path =
        emit_qopt(problem.physics, workdir;
                    problem_id=problem.id,
                    guess_seed=problem.guess_seed)

    t_start = time()
    run_subprocess(python_bin, [script_path], workdir;
                    stdout_file=log_out, stderr_file=log_err)
    t_total = time() - t_start

    ann  = problem.physics
    ctrl = problem.ctrl

    waveform = parse_waveform_file(waveform_path,
                                    length(ann.controls), ann.n_time_steps;
                                    convention = :normalised)
    clamp!(waveform, ctrl.l_bound, ctrl.u_bound)

    fidelity = canonical_rescore(waveform, ctrl)

    return BenchmarkResult(
        driver_name, problem.id,
        fidelity, t_total, ann.max_iter, false,
        waveform, Float64[],
        true, "",
        Dict{String,Any}("backend"     => "qopt (Python subprocess)",
                         "script_path" => script_path,
                         "log"         => log_out),
    )
end
