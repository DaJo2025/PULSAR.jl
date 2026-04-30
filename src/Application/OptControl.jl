# Application/OptControl.jl
#
# Shared optimcon dispatch for AbstractOptimizationContext.
# Loaded after all concrete subtypes (MRControl, QCControl) are defined,
# so this catch-all is only reached for unregistered subtypes.

"""
    optimcon(ctx::AbstractOptimizationContext) -> OptimizationResult

Unified pulse-optimisation entry point.  Dispatches to the solver
registered for the concrete context type:

| Context type   | Required call form                                  |
|----------------|-----------------------------------------------------|
| `QCControl`    | `optimcon(ctx)` — initial waveform is in `ctx.ctrl` |
| `MRControl`    | `optimcon(ctrl, guess)` — separate guess matrix     |

# Example — quantum computing
```julia
ctx    = QCControl(sys, target, ctrl; method=:lbfgs, max_iter=500)
result = optimcon(ctx)
```

# Example — MR
```julia
ctrl   = MRControl(drifts=..., operators=..., ...)
guess  = 0.05 .* randn(2, 250)
result = optimcon(ctrl, guess)
```
"""
function optimcon(ctx::AbstractOptimizationContext)
    if ctx isa AbstractMRControl
        error(
            "optimcon(ctrl::MRControl) requires an explicit initial guess.\n" *
            "Use: optimcon(ctrl, guess)  where guess is [n_ctrl × n_ts] Matrix{Float64}.")
    end
    error("optimcon is not implemented for $(typeof(ctx)).\n" *
          "Define a method: optimcon(::$(typeof(ctx))) -> OptimizationResult")
end
