"""
    comparisons/Drivers/spinach_driver.jl

Driver for Spinach (MATLAB toolbox by Ilya Kuprov, University of Southampton).

Workflow:
  1. Emit a native Spinach .m script from the [`PhysicsAnnotation`](@ref) via
     [`emit_spinach`](@ref) — no per-problem branching.
  2. Run MATLAB (in-process via MATLAB.jl if available, otherwise via
     `matlab -batch` subprocess).
  3. Parse the normalised waveform (already divided by `pwr_levels`) via
     [`parse_waveform_file`](@ref) with `convention = :normalised`.
  4. Re-evaluate fidelity through Pulsar's [`grape_state_kernel`](@ref).

Capabilities declared in `SPINACH_CAPABILITIES` (inside `SpinachEmitter.jl`).
"""

include(joinpath(@__DIR__, "..", "Translator", "Emitters", "SpinachEmitter.jl"))

struct SpinachDriver <: AbstractSolverDriver end

function run_driver(driver::SpinachDriver, problem::BenchmarkProblem)
    driver_name = "Spinach/GRAPE"
    problem_id  = problem.id

    if problem.physics === nothing
        return not_available_result(driver_name, problem_id,
            "Problem has no PhysicsAnnotation — Spinach driver needs annotation to emit.")
    end
    reason = check_supported(SPINACH_CAPABILITIES, problem.physics)
    reason === nothing ||
        return not_available_result(driver_name, problem_id, reason)
    if problem.physics.target.target_per_drift
        return not_available_result(driver_name, problem_id,
            "Spinach emitter does not support target_per_drift objectives.")
    end

    matlab_id    = Base.identify_package("MATLAB")
    matlab_jl_ok = !isnothing(matlab_id) && haskey(Base.loaded_modules, matlab_id)

    matlab_bin = Sys.which("matlab")
    if isnothing(matlab_bin)
        for candidate in ("/Applications/MATLAB_R2024a.app/bin/matlab",
                          "/Applications/MATLAB_R2023b.app/bin/matlab",
                          "/usr/local/bin/matlab")
            if isfile(candidate)
                matlab_bin = candidate
                break
            end
        end
    end
    matlab_bin_ok = !isnothing(matlab_bin)

    if !matlab_jl_ok && !matlab_bin_ok
        return not_available_result(driver_name, problem_id,
            "MATLAB binary not found. Install MATLAB + Spinach, or add MATLAB to PATH.")
    end

    try
        return matlab_jl_ok ?
            _run_spinach_matlab_jl(problem, driver_name) :
            _run_spinach_subprocess(problem, driver_name, matlab_bin)
    catch err
        return error_result(driver_name, problem_id, err)
    end
end

# ─── Mode 1: MATLAB.jl ────────────────────────────────────────────────────────

function _run_spinach_matlab_jl(problem::BenchmarkProblem, driver_name::String)
    MATLAB = Base.loaded_modules[Base.identify_package("MATLAB")]
    workdir = mktempdir()
    script_path, waveform_path = emit_spinach(problem.physics, workdir;
                                                problem_id=problem.id,
                                                guess_seed=problem.guess_seed)
    t_start = time()
    MATLAB.eval_string("run('$(escape_string(script_path))')")
    t_total = time() - t_start
    return _spinach_parse_and_eval(waveform_path, problem, driver_name,
                                     t_total, script_path)
end

# ─── Mode 2: subprocess ───────────────────────────────────────────────────────

function _run_spinach_subprocess(problem::BenchmarkProblem, driver_name::String,
                                  matlab_bin::String)
    workdir = mktempdir()
    log_out = joinpath(workdir, "spinach.out.log")
    log_err = joinpath(workdir, "spinach.err.log")
    script_path, waveform_path = emit_spinach(problem.physics, workdir;
                                                problem_id=problem.id,
                                                guess_seed=problem.guess_seed)

    # MATLAB's parpool breaks when OMP_NUM_THREADS (or sibling vars) cap its
    # worker count. Strip those from the subprocess env; the emitted script
    # honours Pulsar_NCORES via maxNumCompThreads / parcluster.
    matlab_env = copy(ENV)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        delete!(matlab_env, k)
    end

    t_start = time()
    run_subprocess(matlab_bin,
                    ["-batch", "run('$(escape_string(script_path))')"],
                    workdir;
                    stdout_file=log_out, stderr_file=log_err,
                    env=matlab_env)
    t_total = time() - t_start

    return _spinach_parse_and_eval(waveform_path, problem, driver_name,
                                     t_total, script_path)
end

# ─── Parse output and canonical re-eval ───────────────────────────────────────

function _spinach_parse_and_eval(waveform_path::String, problem::BenchmarkProblem,
                                   driver_name::String, t_total::Float64,
                                   script_path::String)
    ann    = problem.physics
    ctrl   = problem.ctrl

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
        Dict{String,Any}("backend"     => "Spinach/MATLAB",
                         "script_path" => script_path),
    )
end
