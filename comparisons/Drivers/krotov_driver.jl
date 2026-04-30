"""
    comparisons/Drivers/krotov_driver.jl

Driver for Krotov.jl — runs as a Julia subprocess.

Workflow:
  1. [`emit_krotov`](@ref) writes a native Julia script driven by the
     [`PhysicsAnnotation`](@ref) — no per-problem branching, no serialised
     matrix literals.
  2. Run `julia --project=<root> <script>` as a subprocess.
  3. Parse the optimised waveform via [`parse_waveform_file`](@ref)
     (convention `:normalised`).
  4. Re-evaluate with [`grape_state_kernel`](@ref).

Capabilities declared in `KROTOV_CAPABILITIES` inside `KrotovEmitter.jl`.
"""

include(joinpath(@__DIR__, "..", "Translator", "Emitters", "KrotovEmitter.jl"))

struct KrotovDriver <: AbstractSolverDriver end

function run_driver(driver::KrotovDriver, problem::BenchmarkProblem)
    driver_name = "Krotov/Krotov"
    problem_id  = problem.id

    if problem.physics === nothing
        return not_available_result(driver_name, problem_id,
            "Problem has no PhysicsAnnotation — Krotov driver needs annotation to emit.")
    end
    reason = check_supported(KROTOV_CAPABILITIES, problem.physics)
    reason === nothing ||
        return not_available_result(driver_name, problem_id, reason)
    if problem.physics.target.target_per_drift
        return not_available_result(driver_name, problem_id,
            "Krotov emitter does not support target_per_drift objectives.")
    end

    if isnothing(Base.identify_package("Krotov"))
        return not_available_result(driver_name, problem_id,
            "Install: ] add QuantumControl; add Krotov")
    end

    try
        return _run_krotov(problem, driver_name)
    catch err
        return error_result(driver_name, problem_id, err)
    end
end

function _run_krotov(problem::BenchmarkProblem, driver_name::String)
    workdir = mktempdir()
    log_out = joinpath(workdir, "krotov.out.log")
    log_err = joinpath(workdir, "krotov.err.log")

    script_path, waveform_path =
        emit_krotov(problem.physics, workdir;
                     problem_id=problem.id,
                     guess_seed=problem.guess_seed)

    project_root = dirname(dirname(dirname(@__FILE__)))
    julia_bin    = joinpath(Sys.BINDIR, "julia")

    t_start = time()
    run_subprocess(julia_bin,
                    ["--project=$(project_root)", script_path],
                    workdir;
                    stdout_file=log_out, stderr_file=log_err)
    t_total = time() - t_start

    ann  = problem.physics
    ctrl = problem.ctrl

    waveform = parse_waveform_file(waveform_path,
                                    length(ann.controls), ann.n_time_steps;
                                    convention = :normalised)
    clamp!(waveform, ctrl.l_bound, ctrl.u_bound)

    krotov_fidelity = _extract_krotov_fidelity(log_out)

    fidelity = canonical_rescore(waveform, ctrl)

    @printf("  [Krotov fidelity: %.4f | Pulsar fidelity: %.4f]\n",
            isnan(krotov_fidelity) ? 0.0 : krotov_fidelity, fidelity)

    return BenchmarkResult(
        driver_name, problem.id,
        fidelity, t_total, ann.max_iter, false,
        waveform, Float64[],
        true, "",
        Dict{String,Any}(
            "backend"         => "Krotov.jl (subprocess)",
            "script_path"     => script_path,
            "krotov_fidelity" => krotov_fidelity,
            "log"             => log_out,
        ),
    )
end

function _extract_krotov_fidelity(log_file::String)::Float64
    isfile(log_file) || return NaN
    for line in readlines(log_file)
        startswith(line, "Pulsar_KROTOV_FIDELITY:") || continue
        return parse(Float64, split(line)[2])
    end
    return NaN
end
