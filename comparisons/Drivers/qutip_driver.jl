"""
    comparisons/Drivers/qutip_driver.jl

Driver for QuTiP + `krotov` Python package.

Workflow:
  1. [`emit_qutip`](@ref) writes a native Python script driven by the
     [`PhysicsAnnotation`](@ref) — one `krotov.Objective` per
     (drift × state-pair) combination, so ensemble and multi-state-pair
     problems are translated, not dropped.
  2. Run `python <script>` as a subprocess.
  3. Parse the normalised waveform via [`parse_waveform_file`](@ref).
  4. Re-evaluate with Pulsar's [`grape_state_kernel`](@ref).

Installation:
    pip install qutip krotov

Capabilities declared in `QUTIP_CAPABILITIES` inside `QuTiPEmitter.jl`.
"""

include(joinpath(@__DIR__, "..", "Translator", "Emitters", "QuTiPEmitter.jl"))

struct QuTiPDriver <: AbstractSolverDriver end

function run_driver(driver::QuTiPDriver, problem::BenchmarkProblem)
    driver_name = "QuTiP/Krotov"
    problem_id  = problem.id

    if problem.physics === nothing
        return not_available_result(driver_name, problem_id,
            "Problem has no PhysicsAnnotation — QuTiP driver needs annotation to emit.")
    end
    reason = check_supported(QUTIP_CAPABILITIES, problem.physics)
    reason === nothing ||
        return not_available_result(driver_name, problem_id, reason)
    if problem.physics.target.target_per_drift
        return not_available_result(driver_name, problem_id,
            "QuTiP emitter does not support target_per_drift objectives.")
    end

    python_bin = _python_binary()
    python_bin === nothing && return not_available_result(driver_name, problem_id,
        "python3 not found on PATH.")

    try
        return _run_qutip(problem, driver_name, python_bin)
    catch err
        return error_result(driver_name, problem_id, err)
    end
end

function _python_binary()::Union{Nothing,String}
    for name in ("python3", "python")
        bin = Sys.which(name)
        bin === nothing || return bin
    end
    return nothing
end

function _run_qutip(problem::BenchmarkProblem, driver_name::String,
                     python_bin::String)
    workdir = mktempdir()
    log_out = joinpath(workdir, "qutip.out.log")
    log_err = joinpath(workdir, "qutip.err.log")

    script_path, waveform_path =
        emit_qutip(problem.physics, workdir;
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
        Dict{String,Any}("backend"     => "QuTiP + krotov (Python subprocess)",
                         "script_path" => script_path,
                         "log"         => log_out),
    )
end
