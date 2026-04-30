"""
    comparisons/Drivers/quantumcontrol_driver.jl

Driver for QuantumControl.jl + GRAPE.jl.

QuantumControl.jl provides a unified interface to quantum optimal control
methods. The GRAPE method is loaded via the QuantumControlBase/GRAPE sub-package.

Installation:
    ]add QuantumControl
    ]add GRAPE

Translation of BenchmarkProblem → QuantumControl problem:
  - drift         → `generator` (time-independent part of the Hamiltonian)
  - operators     → `controls` (list of time-dependent control operators)
  - state pairs   → `objectives` (WeightedObjective list)
  - ensemble      → multiple objectives, one per drift × state pair

NOTE: This translation is a best-effort approximation based on the
QuantumControl.jl 0.3+ API. Install QuantumControl and GRAPE to run this driver.
"""

struct QuantumControlDriver <: AbstractSolverDriver end

const QUANTUMCONTROL_CAPABILITIES = SolverCapabilities(
    multi_spin       = true,
    ensemble         = true,
    multi_state_pair = true,
    lindblad         = false,
    multichannel     = true,
    nonuniform_dt    = true,
    heteronuclear    = true,
    csa              = false,
    dipolar          = false,
    j_coupling       = true,
    amplitude_bounds = true,
)

function run_driver(driver::QuantumControlDriver, problem::BenchmarkProblem)
    driver_name = "QuantumControl/GRAPE"
    problem_id  = problem.id

    # ── Capability check (when the problem carries a PhysicsAnnotation) ───────
    if problem.physics !== nothing
        reason = check_supported(QUANTUMCONTROL_CAPABILITIES, problem.physics)
        reason === nothing ||
            return not_available_result(driver_name, problem_id, reason)
    end

    # ── Availability check ────────────────────────────────────────────────────
    qc_id = Base.identify_package("QuantumControl")
    if isnothing(qc_id) || !haskey(Base.loaded_modules, qc_id)
        # Try to load it lazily
        qc_avail = try
            Base.require(Base.PkgId(qc_id, "QuantumControl"))
            true
        catch
            false
        end
        if !qc_avail
            return not_available_result(driver_name, problem_id,
                "Install: ] add QuantumControl; add GRAPE")
        end
    end

    try
        return _run_quantumcontrol(problem, driver_name)
    catch err
        return error_result(driver_name, problem_id, err)
    end
end

function _run_quantumcontrol(problem::BenchmarkProblem, driver_name::String)
    # Dynamically load QuantumControl and GRAPE
    QC   = Base.loaded_modules[Base.identify_package("QuantumControl")]
    GRAPE_pkg = try
        Base.loaded_modules[Base.identify_package("GRAPE")]
    catch
        error("GRAPE.jl not loaded. Install with: ]add GRAPE")
    end

    ctrl = problem.ctrl
    n_ctrl = length(ctrl.operators)
    n_t    = length(ctrl.pulse_dt)

    # ── Build initial guess ───────────────────────────────────────────────────
    rng   = Random.MersenneTwister(problem.guess_seed)
    guess = 0.05 .* randn(rng, n_ctrl, n_t)
    clamp!(guess, ctrl.l_bound, ctrl.u_bound)

    # ── Translate to QuantumControl objectives ────────────────────────────────
    # QuantumControl.jl uses a generator (H) with controls attached:
    #   H = H_drift + Σ_k u_k(t) * H_ctrl_k
    # For ensemble problems we create multiple objectives.
    #
    # The QuantumControl Hamiltonian tuple format:
    #   H = (H_drift, (H_ctrl_1, u1), (H_ctrl_2, u2), ...)
    # where u_k is the initial control array for operator k.

    t_grid = cumsum(ctrl.pulse_dt)   # time grid (right endpoints)

    # One initial control vector per operator
    u_inits = [guess[k, :] for k in 1:n_ctrl]

    objectives = []
    for (j, H_drift) in enumerate(ctrl.drifts)
        # Build QuantumControl-style generator tuple
        H_ctrl_terms = [(ctrl.operators[k], u_inits[k]) for k in 1:n_ctrl]
        H_gen = tuple(H_drift, H_ctrl_terms...)

        for s in 1:length(ctrl.rho_init)
            obj = QC.Objective(ctrl.rho_init[s], H_gen; target_state=ctrl.rho_targ[s])
            push!(objectives, obj)
        end
    end

    # ── Set up problem ────────────────────────────────────────────────────────
    qc_problem = QC.ControlProblem(
        objectives = objectives,
        tlist      = t_grid,
    )

    # ── Run optimisation ──────────────────────────────────────────────────────
    t_start = time()
    opt_result = QC.optimize(
        qc_problem;
        method     = GRAPE_pkg.GRAPE(),
        iter_stop  = ctrl.max_iter,
        info_hook  = nothing,
    )
    t_total = time() - t_start

    # ── Extract waveform ──────────────────────────────────────────────────────
    # QuantumControl stores optimised controls in opt_result.optimized_controls
    opt_controls = opt_result.optimized_controls
    waveform = Matrix{Float64}(undef, n_ctrl, n_t)
    for k in 1:n_ctrl
        waveform[k, :] = opt_controls[k]
    end
    clamp!(waveform, ctrl.l_bound, ctrl.u_bound)

    # ── Re-evaluate fidelity with Pulsar canonical kernel ─────────────────────
    fidelity = canonical_rescore(waveform, ctrl)

    n_iter    = get(opt_result, :iter, ctrl.max_iter)
    converged = get(opt_result, :converged, false)
    fid_hist  = Float64[]
    try; fid_hist = Float64.(opt_result.tau_vals_history); catch; end

    return BenchmarkResult(
        driver_name, problem.id,
        fidelity, t_total, n_iter, converged,
        waveform, fid_hist,
        true, "",
        Dict{String,Any}("backend" => "QuantumControl.jl + GRAPE.jl"),
    )
end
