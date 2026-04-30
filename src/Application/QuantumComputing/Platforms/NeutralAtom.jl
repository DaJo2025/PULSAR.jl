# Application/QuantumComputing/Platforms/NeutralAtom.jl
# Optimal control dispatch for NeutralAtomSystem.
# Type definition moved to src/Types/NeutralAtomSystem.jl

"""
    optimcon(sys::NeutralAtomSystem, target::QuantumTarget, ctrl::ControlSequence;
             config) -> OptimizationResult

Neutral-atom optimal control via GRAPE.
"""
function optimcon(sys    :: NeutralAtomSystem,
                  target :: QuantumTarget,
                  ctrl   :: ControlSequence;
                  config :: GRAPEConfig = GRAPEConfig())::OptimizationResult
    return _platform_grape(sys, target, ctrl; config = config)
end
