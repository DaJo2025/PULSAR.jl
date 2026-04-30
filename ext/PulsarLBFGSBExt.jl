module PulsarLBFGSBExt

using Pulsar
using LBFGSB
using LinearAlgebra
import Pulsar: _ext_lbfgsb_optimize, _LBFGSB_LOADED

function __init__()
    _LBFGSB_LOADED[] = true
end

# Map Pulsar's tol / f_tol / convergence_mode onto LBFGSB.jl's `pgtol` and
# `factr` knobs. LBFGSB stops on
#   max|proj_g_i| ≤ pgtol      (gradient criterion), and on
#   (f^k - f^{k+1}) / max{|f^k|,|f^{k+1}|,1} ≤ factr · ε_mach (Δf criterion).
#
# We pick whichever knob the user asked for and clamp the other one to a
# permissive value, so LBFGSB respects the requested mode.
function _lbfgsb_knobs(convergence_mode::Symbol, tol::Float64, f_tol::Float64)
    eps_mach = eps(Float64)
    if convergence_mode === :gradient_norm
        pgtol  = tol
        factr  = 1e1                      # ≈ machine precision in Δf
    elseif convergence_mode === :fidelity_change
        pgtol  = 1e-15                    # effectively off
        factr  = max(f_tol / eps_mach, 1.0)
    elseif convergence_mode === :both
        pgtol  = tol
        factr  = max(f_tol / eps_mach, 1.0)
    else
        throw(ArgumentError(
            "convergence_mode must be :gradient_norm, :fidelity_change, or :both"))
    end
    return pgtol, factr
end

function _ext_lbfgsb_optimize(
    f       :: Function,
    grad!   :: Function,
    θ0      :: AbstractVector{<:Real};
    lower   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper   :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    memory  :: Int     = 10,
    max_iter:: Int     = 1_000,
    tol     :: Float64 = 1e-6,
    f_tol   :: Float64 = 1e-8,
    convergence_mode :: Symbol = :gradient_norm,
    verbose :: Bool    = false,
    callback = nothing,
)
    n  = length(θ0)
    lb = lower === nothing ? fill(-Inf, n) : Float64.(lower)
    ub = upper === nothing ? fill( Inf, n) : Float64.(upper)
    x0 = clamp.(float.(θ0), lb, ub)

    # bounds matrix in LBFGSB convention: [nbd; lower; upper] per column
    bounds = Matrix{Float64}(undef, 3, n)
    @inbounds for i in 1:n
        lo_finite = isfinite(lb[i])
        up_finite = isfinite(ub[i])
        if      lo_finite &&  up_finite ; bounds[1, i] = 2.0
        elseif  lo_finite && !up_finite ; bounds[1, i] = 1.0
        elseif !lo_finite &&  up_finite ; bounds[1, i] = 3.0
        else                              bounds[1, i] = 0.0
        end
        bounds[2, i] = lo_finite ? lb[i] : 0.0
        bounds[3, i] = up_finite ? ub[i] : 0.0
    end

    pgtol, factr = _lbfgsb_knobs(convergence_mode, tol, f_tol)
    iprint       = verbose ? 1 : -1

    # Tracker for evaluation counts and best-seen point
    n_evals_f    = Ref(0)
    n_evals_g    = Ref(0)
    best_f       = Ref(Inf)
    best_x       = copy(x0)
    iter_seen    = Ref(0)

    func = function (x)
        v = f(x)
        n_evals_f[] += 1
        if v < best_f[]
            best_f[] = v
            copyto!(best_x, x)
        end
        if callback !== nothing
            iter_seen[] += 1
            callback(iter_seen[], v)
        end
        return v
    end

    grad_w = function (g, x)
        grad!(g, x)
        n_evals_g[] += 1
        return g
    end

    obj = L_BFGS_B(n, max(memory, 3))
    f_opt, x_opt = obj(func, grad_w, x0, bounds;
                       m       = max(memory, 3),
                       factr   = factr,
                       pgtol   = pgtol,
                       iprint  = iprint,
                       maxfun  = 4 * max_iter,
                       maxiter = max_iter)

    iters     = Int(obj.isave[30])
    n_evals   = n_evals_f[] + n_evals_g[]
    stats = (
        evals     = n_evals,
        iters     = iters,
        converged = iters < max_iter,
        backend   = :LBFGSB_jl,
    )
    # `θ_best` ≡ `best_x` here (LBFGSB's monotone Wolfe accepts non-monotone
    # f temporarily, but returns the final x; we track best_x for safety).
    final_x = best_f[] ≤ f_opt ? best_x : x_opt
    final_f = best_f[] ≤ f_opt ? best_f[] : f_opt
    return final_x, final_f, stats
end

end  # module PulsarLBFGSBExt
