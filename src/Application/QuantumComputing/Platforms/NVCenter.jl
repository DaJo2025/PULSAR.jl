# Application/QuantumComputing/Platforms/NVCenter.jl
# Optimal control dispatch for NVCenterSystem.
# Type definition moved to src/Types/NVCenterSystem.jl

"""
    optimcon(sys::NVCenterSystem, target::QuantumTarget, ctrl::ControlSequence;
             config, use_lindblad) -> OptimizationResult

NV center optimal control via GRAPE.

# Arguments
- `use_lindblad` — propagate with Lindblad master equation (T₁/T₂ decoherence)
"""
function optimcon(sys           :: NVCenterSystem,
                  target        :: QuantumTarget,
                  ctrl          :: ControlSequence;
                  config        :: GRAPEConfig = GRAPEConfig(),
                  use_lindblad  :: Bool        = false)::OptimizationResult
    return _platform_grape(sys, target, ctrl;
                           config       = config,
                           use_lindblad = use_lindblad)
end
