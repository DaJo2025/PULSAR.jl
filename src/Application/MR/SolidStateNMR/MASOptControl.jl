# Application/MR/SolidStateNMR/MASOptControl.jl
# Optimal control for MAS solid-state NMR via the generic GRAPE interface.
using LinearAlgebra

"""
    optimcon(sys::MASSpinSystem, target, ctrl::ControlSequence;
             config::GRAPEConfig, orientations=nothing) -> OptimizationResult

Run GRAPE for MAS solid-state NMR.

- Without orientations: single-crystal GRAPE.
- With orientations (powder_grid output): powder-averaged GRAPE.
"""
function optimcon(sys::MASSpinSystem,
                  target::QuantumTarget,
                  ctrl::ControlSequence;
                  config::GRAPEConfig = GRAPEConfig(),
                  orientations::Union{Nothing,Vector{NTuple{4,Float64}}} = nothing)::OptimizationResult
    if orientations === nothing
        return grape_optimize(sys, target, ctrl;
            gradient_fn = (s,c,t) -> compute_grape_gradient(s,c,t),
            config      = config)
    else
        return grape_optimize(sys, target, ctrl;
            gradient_fn = (s,c,t) -> compute_grape_gradient_powder(s,c,t,orientations),
            config      = config)
    end
end
