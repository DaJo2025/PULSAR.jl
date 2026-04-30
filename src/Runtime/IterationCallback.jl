# ============================================================================
# Utilities/IterationCallback.jl
#
# Unified iteration-progress reporting for all PULSAR optimizers.
#
# Usage
# ─────
#   cb = iteration_callback(print_interval=10)
#   θ, f, stats = grape_optimize(f, grad!, θ0; callback=cb)
#   θ, f, stats = ga_optimize(f, θ0; callback=cb)
#
# For QOC objectives where `f` returns −fidelity (minimisation convention),
# pass `negate=true` so the display shows the fidelity (positive value):
#   cb = iteration_callback(print_interval=10, negate=true)
# ============================================================================

using Printf

"""
    IterationCallback

Callable struct for standardised iteration-progress reporting across all
PULSAR optimizers. Create with [`iteration_callback`](@ref).

## Fields
- `print_interval::Int` — print every N iterations (and iteration 1).
- `negate::Bool`        — display `−f` instead of `f` (useful for QOC
  objectives that minimise `−fidelity`; set `negate=true` to show fidelity).
- `show_grad::Bool`     — show gradient-norm column (where available).
- `show_evals::Bool`    — show function-evaluation count.
- `show_step::Bool`     — show step size / line-search α.
- `label::String`       — optional prefix shown in the header.
"""
struct IterationCallback
    print_interval :: Int
    negate         :: Bool
    show_grad      :: Bool
    show_evals     :: Bool
    show_step      :: Bool
    label          :: String
    _header_printed :: Ref{Bool}   # mutable flag: true once header is shown
end

"""
    iteration_callback(; print_interval=10, negate=false, show_grad=true,
                         show_evals=false, show_step=false, label="")
                       → IterationCallback

Create a callback for standardised per-iteration output.  Pass the returned
object as `callback=cb` to any PULSAR optimizer.

# Keyword arguments
- `print_interval::Int` — print every N iterations (default 10).
  Set to 1 to print every iteration.
- `negate::Bool`        — set `true` when the objective is `−fidelity`
  (minimisation convention) so the displayed value is the fidelity (default false).
- `show_grad::Bool`     — print `|∇|` column when the optimizer provides it
  (gradient-based methods only; default true).
- `show_evals::Bool`    — print cumulative function-evaluation count (default false).
- `show_step::Bool`     — print step size / line-search α (default false).
- `label::String`       — optional tag prepended to every line (default "").

# Example
```julia
# Basic callback (print every 10 iterations)
cb = iteration_callback(print_interval=10)
θ_opt, f_opt, stats = lbfgs_optimize(f, grad!, θ0; callback=cb)

# QOC example: objective = −fidelity, display fidelity (negate=true)
cb = iteration_callback(print_interval=25, negate=true, show_grad=true)
θ_opt, f_opt, stats = grape_optimize(nmr_obj, nmr_grad!, θ0; callback=cb)

# Metaheuristic (no gradient available)
cb = iteration_callback(print_interval=50, show_grad=false, show_evals=true)
θ_opt, f_opt, stats = ga_optimize(f, θ0; callback=cb)
```
"""
function iteration_callback(;
    print_interval :: Int    = 10,
    negate         :: Bool   = false,
    show_grad      :: Bool   = true,
    show_evals     :: Bool   = false,
    show_step      :: Bool   = false,
    label          :: String = "",
)
    return IterationCallback(
        print_interval, negate, show_grad, show_evals, show_step,
        label, Ref(false))
end

# ─── Callable ─────────────────────────────────────────────────────────────────

"""
    (cb::IterationCallback)(iter, f_val; grad=nothing, evals=nothing, step=nothing)

Called by optimizers at the end of each iteration.  Prints a standardised
progress line according to the callback's settings.

# Arguments
- `iter`         — current iteration number (Int).
- `f_val`        — current objective value (Real); negated for display if
  `cb.negate == true`.
- `grad=nothing` — gradient norm (Real or nothing).
- `evals=nothing`— cumulative function evaluations (Int or nothing).
- `step=nothing` — step size / line-search α (Real or nothing).
"""
function (cb::IterationCallback)(
    iter   :: Int,
    f_val  :: Real;
    grad   :: Union{Nothing,Real} = nothing,
    evals  :: Union{Nothing,Int}  = nothing,
    step   :: Union{Nothing,Real} = nothing,
)
    (iter % cb.print_interval == 0 || iter == 1) || return nothing

    # Print column header on first call
    if !cb._header_printed[]
        _icb_print_header(cb)
        cb._header_printed[] = true
    end

    val  = cb.negate ? -Float64(f_val) : Float64(f_val)
    lbl  = isempty(cb.label) ? "" : "[$(cb.label)] "

    grad_s  = (cb.show_grad  && grad  !== nothing) ? @sprintf("  |∇|=%.3e", Float64(grad))  : ""
    evals_s = (cb.show_evals && evals !== nothing) ? @sprintf("  ev=%d",     Int(evals))     : ""
    step_s  = (cb.show_step  && step  !== nothing) ? @sprintf("  α=%.3e",   Float64(step))  : ""

    @printf("  %s%6d  F=%+.8f%s%s%s\n", lbl, iter, val, grad_s, evals_s, step_s)
    return nothing
end

# ─── Internal helpers ─────────────────────────────────────────────────────────

function _icb_print_header(cb::IterationCallback)
    lbl      = isempty(cb.label) ? "" : "[$(cb.label)] "
    grad_h   = cb.show_grad  ? "  |∇|" : ""
    evals_h  = cb.show_evals ? "  evals" : ""
    step_h   = cb.show_step  ? "  α" : ""
    val_label = cb.negate ? "Fidelity (−obj)" : "Objective f"
    @printf("  %s%6s  %-20s%s%s%s\n", lbl, "Iter", val_label, grad_h, evals_h, step_h)
    lw = 8 + 22 +
         (cb.show_grad  ? 14 : 0) +
         (cb.show_evals ? 10 : 0) +
         (cb.show_step  ? 12 : 0) +
         length(lbl)
    println("  " * "─"^lw)
end

"""
    reset!(cb::IterationCallback)

Reset the header-printed flag so the header prints again on the next call.
Useful when reusing the same callback across multiple optimization runs.
"""
reset!(cb::IterationCallback) = (cb._header_printed[] = false; nothing)
