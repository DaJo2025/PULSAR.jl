# Application/QuantumComputing/Platforms/SpinQubit.jl
# Optimal control dispatch for SpinQubitSystem.
# Type definition moved to src/Types/SpinQubitSystem.jl

"""
    optimcon(sys::SpinQubitSystem, target::QuantumTarget, ctrl::ControlSequence;
             config, b1_factors) -> OptimizationResult

Spin-qubit optimal control via GRAPE.

Optionally constructs a B₁-robustness ensemble over a vector of B₁ scaling
factors (e.g. `[0.9, 1.0, 1.1]` for ±10 % inhomogeneity).  When
`b1_factors` has more than one element the fidelity is ensemble-averaged.

# Arguments
- `sys`         — `SpinQubitSystem`
- `target`      — `QuantumTarget`
- `ctrl`        — initial `ControlSequence`
- `config`      — `GRAPEConfig`
- `b1_factors`  — Vector{Float64} B₁ scaling factors for robustness (default [1.0])

# Returns
`OptimizationResult`
"""
function optimcon(sys         :: SpinQubitSystem,
                  target      :: QuantumTarget,
                  ctrl        :: ControlSequence;
                  config      :: GRAPEConfig     = GRAPEConfig(),
                  b1_factors  :: Vector{Float64} = [1.0])::OptimizationResult

    if length(b1_factors) == 1
        return _platform_grape(sys, target, ctrl; config = config)
    end

    # Ensemble over B1 scaling
    systems_ens = [QuantumSystem(sys.H_drift,
                                  [b1 .* Hk for Hk in sys.H_controls],
                                  sys.dim, sys.n_controls, sys.metadata)
                   for b1 in b1_factors]
    return grape_optimize_ensemble(systems_ens, target, ctrl; config=config)
end
