# Application/QuantumComputing/OptControl.jl
#
# QCControl — unified problem-specification struct for quantum-computing
# optimal control.  Wraps system, target, initial waveform, and solver
# settings in one object so that `optimcon(ctx)` dispatches correctly.
#
# Loaded after all QC platform files so it can see their types, but the
# struct itself only depends on Layer 1–4 exports.

using LinearAlgebra
using Printf

# ─── QCControl ───────────────────────────────────────────────────────────────

"""
    QCControl <: AbstractOptimizationContext

Unified problem specification for quantum-computing optimal control.

Stores the assembled quantum system, target, initial control waveform,
and all solver settings in a single object.  Pass it to `optimcon(ctx)`
to run the optimisation.

Unlike `MRControl` (which requires a separate `guess` matrix), the
initial control waveform is embedded in the `ctrl` field.

# Fields
- `system`           — assembled `QuantumSystem`
- `target`           — `QuantumTarget` (state-transfer or unitary)
- `ctrl`             — `ControlSequence` carrying the initial waveform
- `method`           — optimisation algorithm (`:grape`, `:lbfgs`, …)
- `grape_config`     — `GRAPEConfig` (max_iter, step_size, verbose, …)
- `penalty_fns`      — optional penalty callables `(sys, ctrl, tgt) → Float64`
- `penalty_grad_fns` — corresponding gradient callables
- `verbose`          — print progress during optimisation
- `metadata`         — user-defined key-value pairs

# Supported methods

| Symbol         | Algorithm                          |
|----------------|------------------------------------|
| `:grape`       | GRAPE gradient ascent (default)    |
| `:lbfgs`       | QOC L-BFGS with backtracking       |
| `:lbfgsb`      | L-BFGS-B with Wolfe line search    |
| `:cg`          | Nonlinear conjugate gradient (PR+) |
| `:cmaes`       | CMA-ES (derivative-free)           |
| `:pso`         | Particle Swarm (derivative-free)   |
| `:nelder_mead` | Nelder-Mead simplex                |

# Example
```julia
sys    = transmon_system(5.0e9, -200e6)
target = unitary_target([0 1; 1 0] .+ 0im)
ctrl   = ControlSequence(0.01 .* randn(2, 200), 5e-9, 1e-6, 200)

ctx    = QCControl(sys, target, ctrl; method=:lbfgs, max_iter=500)
result = optimcon(ctx)
```
"""
struct QCControl <: AbstractOptimizationContext
    system           :: QuantumSystem
    target           :: QuantumTarget
    ctrl             :: ControlSequence
    method           :: Symbol
    grape_config     :: GRAPEConfig
    penalty_fns      :: Vector{Function}
    penalty_grad_fns :: Vector{Function}
    verbose          :: Bool
    metadata         :: Dict{Symbol,Any}
end

"""
    QCControl(sys, target, ctrl; method=:grape, max_iter=1000, kwargs...) -> QCControl

Keyword constructor.  `sys` can be any `AbstractQuantumSystem`; the fields
`H_drift`, `H_controls`, `dim`, and `n_controls` are extracted automatically.

# Keyword arguments
- `method`            — see supported methods above (default `:grape`)
- `max_iter`          — maximum iterations (default 1000)
- `convergence_tol`   — fidelity convergence threshold (default 1e-8)
- `gradient_norm_tol` — gradient norm convergence threshold (default 1e-6)
- `step_size`         — initial GRAPE step size (default 0.01)
- `verbose`           — print per-iteration progress (default false)
- `print_interval`    — logging frequency in iterations (default 100)
- `penalty_fns`       — `Vector{Function}` of penalty callables (default `[]`)
- `penalty_grad_fns`  — corresponding gradient callables (default `[]`)
- `metadata`          — `Dict{Symbol,Any}` of user annotations (default `Dict()`)
"""
function QCControl(
    sys              :: AbstractQuantumSystem,
    target           :: QuantumTarget,
    ctrl             :: ControlSequence;
    method           :: Symbol           = :grape,
    max_iter         :: Int              = 1000,
    convergence_tol  :: Float64          = 1e-8,
    gradient_norm_tol :: Float64         = 1e-6,
    step_size        :: Float64          = 0.01,
    verbose          :: Bool             = false,
    print_interval   :: Int              = 100,
    penalty_fns      :: Vector{Function} = Function[],
    penalty_grad_fns :: Vector{Function} = Function[],
    metadata         :: Dict{Symbol,Any} = Dict{Symbol,Any}(),
)
    qs = QuantumSystem(
        sys.H_drift,
        sys.H_controls,
        sys.dim,
        sys.n_controls,
        Dict{String,Any}(),
    )
    cfg = GRAPEConfig(
        max_iter          = max_iter,
        convergence_tol   = convergence_tol,
        gradient_norm_tol = gradient_norm_tol,
        step_size         = step_size,
        verbose           = verbose,
        print_interval    = print_interval,
    )
    return QCControl(qs, target, ctrl, method, cfg,
                     penalty_fns, penalty_grad_fns, verbose, metadata)
end

# ─── Kernel + closure builders (Theme 6b) ────────────────────────────────────
#
# `_qc_kernel(w, ctx)` is the QC analogue of `_mr_kernel(w, ctrl)` on the MR
# side: a single dispatch surface that picks the right physical kernel
# (Hilbert vs Liouville) for the given context type.  Closure builders wrap
# the kernel into the `(f, grad!, θ0)` interface every generic optimizer
# expects, so any algorithm that does not embed system structure can run on
# either `QCControl` or `LindbladQCControl` without per-method branching.

"""
    _qc_kernel(w, ctx::QCControl) -> (F::Float64, grad::Matrix{Float64})

Closed-system Hilbert-space kernel for a `QCControl` context.  Reuses
`_so_fidelity` / `_so_gradient` from `Optimization/SecondOrder/SecondOrderMethods.jl`
and applies any user-supplied penalty terms in additive form
`F_total = F − Σ pf(w)`, `∇F_total = ∇F − Σ pg(w)`.
"""
function _qc_kernel(w::Matrix{Float64}, ctx::QCControl)
    sys, tgt = ctx.system, ctx.target
    n_c, n_t = size(w)
    dt       = ctx.ctrl.dt
    u_vec    = vec(w)
    F = _so_fidelity(sys, tgt, u_vec, n_c, n_t, dt)
    G = reshape(_so_gradient(sys, tgt, u_vec, n_c, n_t, dt), n_c, n_t)
    for (pf, pg) in zip(ctx.penalty_fns, ctx.penalty_grad_fns)
        F -= pf(w)
        G .-= pg(w)
    end
    return F, G
end

"""
    _qc_make_fg_closures(ctx, n_c, n_t) -> (f, grad!)

Return cached negated-fidelity closures suitable for any generic optimizer
expecting `f(θ)` + `grad!(g, θ)` where the user wants to *minimise*.  The
underlying kernel is called once per unique `θ` and the result is cached in
the closure's lexical scope so calling `f` and `grad!` in either order does
not double-evaluate.
"""
function _qc_make_fg_closures(ctx, n_c::Int, n_t::Int)
    last_θ = fill(NaN, n_c * n_t)
    last_F = Ref(0.0)
    last_G = zeros(Float64, n_c, n_t)

    refresh! = (θ_flat) -> begin
        if θ_flat != last_θ
            w = reshape(Float64.(θ_flat), n_c, n_t)
            F, G = _qc_kernel(w, ctx)
            last_F[] = F
            last_G  .= G
            last_θ  .= θ_flat
        end
    end

    f      = θ -> (refresh!(θ); -last_F[])
    grad!  = (g, θ) -> (refresh!(θ); g .= -vec(last_G); g)
    return f, grad!
end

"""
    _qc_make_f_only_closure(ctx, n_c, n_t) -> f

Return a derivative-free closure `f(θ) = -F(reshape(θ, n_c, n_t))` for use
with metaheuristic optimizers that only need scalar function evaluations.
The kernel is invoked twice (fidelity + dropped gradient); for genuinely
gradient-free fast paths, a per-method specialisation can override this.
"""
function _qc_make_f_only_closure(ctx, n_c::Int, n_t::Int)
    return θ -> begin
        w = reshape(Float64.(θ), n_c, n_t)
        F, _ = _qc_kernel(w, ctx)
        return -F
    end
end

# ─── optimcon dispatch ────────────────────────────────────────────────────────

"""
    optimcon(ctx::QCControl) -> OptimizationResult

Run the optimisation algorithm specified by `ctx.method`.

Internally delegates to the canonical algorithm implementation so the
result is identical to calling that algorithm directly.
"""
function optimcon(ctx::QCControl)::OptimizationResult
    sys  = ctx.system
    tgt  = ctx.target
    ctrl = ctx.ctrl
    cfg  = ctx.grape_config

    if ctx.method == :grape
        return grape_optimize(sys, tgt, ctrl;
                              config           = cfg,
                              penalty_fns      = ctx.penalty_fns,
                              penalty_grad_fns = ctx.penalty_grad_fns)

    elseif ctx.method == :lbfgs
        lbfgs_cfg = LBFGSConfig(
            max_iter     = cfg.max_iter,
            gradient_tol = cfg.gradient_norm_tol,
            verbose      = ctx.verbose,
        )
        return lbfgs_optimize(sys, tgt, ctrl; config=lbfgs_cfg)

    elseif ctx.method == :lbfgsb
        nc = sys.n_controls; nt = ctrl.n_timesteps
        f_neg, grad_neg! = _qc_make_fg_closures(ctx, nc, nt)
        u0  = vec(copy(ctrl.controls))
        res = grape_lbfgsb_optimize(f_neg, grad_neg!, u0;
                  max_iter = cfg.max_iter,
                  tol      = cfg.gradient_norm_tol,
                  verbose  = ctx.verbose)
        # grape_lbfgsb_optimize shapes controls as [1 × nc*nt]; reshape to [nc × nt]
        return OptimizationResult(
            reshape(vec(res.controls), nc, nt),
            res.fidelity,
            res.fidelity_history,
            res.gradient_norm_history,
            res.n_iterations,
            res.converged,
            res.termination_reason,
            res.total_time,
            res.n_fidelity_evaluations,
            res.n_gradient_evaluations,
            res.metadata,
        )

    elseif ctx.method == :cg
        nc = sys.n_controls; nt = ctrl.n_timesteps
        f_neg, grad_neg! = _qc_make_fg_closures(ctx, nc, nt)
        u0      = vec(copy(ctrl.controls))
        t_start = time()
        θ_best, f_best, stats = cg_optimize(f_neg, grad_neg!, u0;
            max_iter = cfg.max_iter,
            tol      = cfg.gradient_norm_tol,
            verbose  = ctx.verbose)
        reason = stats.converged ? "gradient norm < tol" : "maximum iterations reached"
        return OptimizationResult(
            reshape(θ_best, nc, nt),
            -f_best,
            Float64[],
            Float64[],
            stats.iters,
            stats.converged,
            reason,
            time() - t_start,
            stats.evals,
            stats.evals,
            Dict{String,Any}("algorithm" => "CG"),
        )

    elseif ctx.method == :cmaes
        cmaes_cfg = CMAESConfig(max_iter=cfg.max_iter, verbose=ctx.verbose)
        return cmaes_optimize(sys, tgt, ctrl; config=cmaes_cfg)

    elseif ctx.method == :pso
        pso_cfg = PSOConfig(max_iter=cfg.max_iter, verbose=ctx.verbose)
        return pso_optimize(sys, tgt, ctrl; config=pso_cfg)

    elseif ctx.method == :nelder_mead
        nm_cfg = NelderMeadConfig(max_iter=cfg.max_iter, verbose=ctx.verbose)
        return nelder_mead_optimize(sys, tgt, ctrl; config=nm_cfg)

    else
        throw(ArgumentError(
            "Unknown QCControl method :$(ctx.method). " *
            "Supported: :grape, :lbfgs, :lbfgsb, :cg, :cmaes, :pso, :nelder_mead"))
    end
end
