"""
    comparisons/Drivers/pulsar_driver.jl

Pulsar's own solver driver. Supports methods:
    :lbfgs   — L-BFGS with Armijo backtracking (default, recommended)
    :grape   — normalised gradient ascent
    :lbfgsb  — L-BFGS-B with Wolfe line search
    :cg      — nonlinear conjugate gradient
    :cmaes   — CMA-ES derivative-free

The driver rebuilds the MRControl (or LindbladMRControl) with the requested
method, runs optimcon, then re-evaluates fidelity using the canonical kernel.
"""

struct PulsarDriver <: AbstractSolverDriver
    method :: Symbol
end

function run_driver(driver::PulsarDriver, problem::BenchmarkProblem)
    driver_name = "Pulsar/:$(driver.method)"
    try
        return _run_pulsar(driver.method, problem, driver_name)
    catch err
        return error_result(driver_name, problem.id, err)
    end
end

# ─── Internal implementation ──────────────────────────────────────────────────

function _pulsar_ctrl_with_method(ctrl, method::Symbol)
    # Return a new control struct with the requested method and verbose=false.
    # We rebuild via the keyword constructor (which normalises types).
    if ctrl isa LindbladMRControl
        return LindbladMRControl(
            drifts      = ctrl.drifts,
            operators   = ctrl.operators,
            jump_ops    = ctrl.jump_ops,
            decay_rates = ctrl.decay_rates,
            rho_init    = ctrl.rho_init,
            rho_targ    = ctrl.rho_targ,
            pwr_levels  = ctrl.pwr_levels,
            pulse_dt    = ctrl.pulse_dt,
            penalties   = ctrl.penalties,
            p_weights   = ctrl.p_weights,
            l_bound     = ctrl.l_bound,
            u_bound     = ctrl.u_bound,
            method      = method,
            max_iter    = ctrl.max_iter,
            grad_tol    = ctrl.grad_tol,
            fidelity    = ctrl.fidelity,
            lbfgs_memory= ctrl.lbfgs_memory,
            verbose     = false,
            print_interval = ctrl.print_interval,
            backend     = ctrl.backend,
            precision   = ctrl.precision,
        )
    else
        return MRControl(
            drifts     = ctrl.drifts,
            operators  = ctrl.operators,
            rho_init   = ctrl.rho_init,
            rho_targ   = ctrl.rho_targ,
            pwr_levels = ctrl.pwr_levels,
            pulse_dt   = ctrl.pulse_dt,
            penalties  = ctrl.penalties,
            p_weights  = ctrl.p_weights,
            l_bound    = ctrl.l_bound,
            u_bound    = ctrl.u_bound,
            method     = method,
            max_iter   = ctrl.max_iter,
            grad_tol   = ctrl.grad_tol,
            fidelity   = ctrl.fidelity,
            lbfgs_memory = ctrl.lbfgs_memory,
            verbose    = false,
            print_interval = ctrl.print_interval,
            backend    = ctrl.backend,
        )
    end
end

function _is_lindblad(ctrl)::Bool
    return ctrl isa LindbladMRControl
end

function _reeval_fidelity(waveform::Matrix{Float64}, ctrl)::Float64
    # Re-evaluate fidelity using the canonical Pulsar kernel.
    if _is_lindblad(ctrl)
        fid, _ = grape_lindblad_kernel(waveform, ctrl)
    else
        fid, _ = grape_state_kernel(waveform, ctrl)
    end
    return fid
end

function _run_pulsar(method::Symbol, problem::BenchmarkProblem, driver_name::String)
    ctrl = _pulsar_ctrl_with_method(problem.ctrl, method)

    # Build reproducible random initial guess
    rng   = Random.MersenneTwister(problem.guess_seed)
    n_ctrl = length(ctrl.operators)
    n_t    = length(ctrl.pulse_dt)
    guess  = 0.05 .* randn(rng, n_ctrl, n_t)
    clamp!(guess, ctrl.l_bound, ctrl.u_bound)

    t_start = time()
    result  = optimcon(ctrl, guess)
    t_total = time() - t_start

    # Canonical fidelity re-evaluation with the original ctrl (not the rebuilt one)
    # Uses the same method-independent kernel.
    fidelity = _reeval_fidelity(result.controls, ctrl)

    return BenchmarkResult(
        driver_name,
        problem.id,
        fidelity,
        t_total,
        result.n_iterations,
        result.converged,
        result.controls,
        result.fidelity_history,
        true,   # available
        "",
        Dict{String,Any}(
            "method"            => string(method),
            "termination"       => result.termination_reason,
            "reported_fidelity" => result.fidelity,
            "n_fidelity_evals"  => result.n_fidelity_evaluations,
        ),
    )
end
