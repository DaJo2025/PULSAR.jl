# Application/QuantumComputing/Platforms/TrappedIon.jl
# Optimal control dispatch for TrappedIonSystem.
# Type definition moved to src/Types/TrappedIonSystem.jl

"""
    optimcon(sys::TrappedIonSystem, target::QuantumTarget, ctrl::ControlSequence;
             config, leakage_weight) -> OptimizationResult

Trapped-ion optimal control via GRAPE.

# Arguments
- `sys`            — [`TrappedIonSystem`](@ref)
- `target`         — [`QuantumTarget`](@ref)
- `ctrl`           — initial `ControlSequence`
- `config`         — `GRAPEConfig` (default: `GRAPEConfig()`)
- `leakage_weight` — currently unused (no leakage in spin-1/2 model); reserved

# Returns
`OptimizationResult`
"""
function optimcon(sys            :: TrappedIonSystem,
                  target         :: QuantumTarget,
                  ctrl           :: ControlSequence;
                  config         :: GRAPEConfig = GRAPEConfig(),
                  leakage_weight :: Float64     = 0.0)::OptimizationResult
    return _platform_grape(sys, target, ctrl; config = config)
end
