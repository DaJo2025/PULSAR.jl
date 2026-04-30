"""
    comparisons/Drivers/quandary_driver.jl

Driver for Quandary (LLNL optimal control solver).

Workflow:
  1. [`emit_quandary`](@ref) writes a native Python script driven by the
     [`PhysicsAnnotation`](@ref) — uses Quandary's custom-Hamiltonian mode
     (`standardmodel = False`) so no transmon parameterisation is assumed.
  2. Run `python <script>` as a subprocess.
  3. Parse the normalised waveform via [`parse_waveform_file`](@ref).
  4. Re-evaluate with Pulsar's [`grape_state_kernel`](@ref).

Installation:
    https://github.com/LLNL/quandary  (build + pip install .)

Capabilities declared in `QUANDARY_CAPABILITIES` inside `QuandaryEmitter.jl`.
"""

include(joinpath(@__DIR__, "..", "Translator", "Emitters", "QuandaryEmitter.jl"))

struct QuandaryDriver <: AbstractSolverDriver end

function run_driver(driver::QuandaryDriver, problem::BenchmarkProblem)
    driver_name = "Quandary/GrapeOC"
    problem_id  = problem.id

    if problem.physics === nothing
        return not_available_result(driver_name, problem_id,
            "Problem has no annotation — Quandary driver needs one to emit.")
    end
    reason = check_supported(QUANDARY_CAPABILITIES, problem.physics)
    reason === nothing ||
        return not_available_result(driver_name, problem_id, reason)
    # target_per_drift only applies to NMR PhysicsAnnotation; TransmonAnnotation
    # has no such concept (single rotating-frame drift always).
    if problem.physics isa PhysicsAnnotation && problem.physics.target.target_per_drift
        return not_available_result(driver_name, problem_id,
            "Quandary emitter does not support target_per_drift objectives.")
    end

    python_bin = _quandary_python_binary()
    python_bin === nothing && return not_available_result(driver_name, problem_id,
        "python3 not found on PATH.")

    try
        return _run_quandary(problem, driver_name, python_bin)
    catch err
        return error_result(driver_name, problem_id, err)
    end
end

function _quandary_python_binary()::Union{Nothing,String}
    for name in ("python3", "python")
        bin = Sys.which(name)
        bin === nothing || return bin
    end
    return nothing
end

function _run_quandary(problem::BenchmarkProblem, driver_name::String,
                        python_bin::String)
    workdir = mktempdir()
    log_out = joinpath(workdir, "quandary.out.log")
    log_err = joinpath(workdir, "quandary.err.log")

    script_path, waveform_path =
        emit_quandary(problem.physics, workdir;
                        problem_id=problem.id,
                        guess_seed=problem.guess_seed)

    # The Quandary Python wrapper shells out to the C++ `quandary` binary,
    # which must be on $PATH.  If the user did not export it, look in the
    # common Pulsar checkout location and prepend its dir to subprocess PATH.
    child_env = copy(ENV)
    for candidate in (joinpath(dirname(dirname(@__DIR__)), "quandary", "quandary"),
                      expanduser("~/quandary/quandary"))
        if isfile(candidate)
            qd = dirname(candidate)
            occursin(qd, child_env["PATH"]) ||
                (child_env["PATH"] = qd * ":" * child_env["PATH"])
            break
        end
    end

    t_start = time()
    run_subprocess(python_bin, [script_path], workdir;
                    stdout_file=log_out, stderr_file=log_err,
                    env=child_env)
    t_total = time() - t_start

    ann  = problem.physics
    ctrl = problem.ctrl

    n_ctrl = ann isa TransmonAnnotation ? 2 * n_qubits(ann) : length(ann.controls)
    waveform = parse_waveform_file(waveform_path,
                                    n_ctrl, ann.n_time_steps;
                                    convention = :normalised)
    clamp!(waveform, ctrl.l_bound, ctrl.u_bound)

    fidelity = canonical_rescore(waveform, ctrl)

    return BenchmarkResult(
        driver_name, problem.id,
        fidelity, t_total, ann.max_iter, false,
        waveform, Float64[],
        true, "",
        Dict{String,Any}("backend"     => "Quandary (Python subprocess)",
                         "script_path" => script_path,
                         "log"         => log_out),
    )
end
