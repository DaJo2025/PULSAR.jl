# Application/MR/OptControlExtensions.jl
# New optimcon overloads using the generic GRAPE physics-hook interface.
# Loaded after Types/HeteronuclearSystem.jl and Physics/MRPhysics.jl.
using LinearAlgebra

# ============================================================================
# optimcon overloads: MRSpinSystem with band-selective support
# ============================================================================

"""
    optimcon(sys::MRSpinSystem, target::QuantumTarget, ctrl::ControlSequence;
             config, bands) -> OptimizationResult

Closed-system (Hilbert space) liquid-state NMR. With bands: band-selective.
"""
function optimcon(sys::MRSpinSystem,
                  target::QuantumTarget,
                  ctrl::ControlSequence;
                  config::GRAPEConfig        = GRAPEConfig(),
                  bands::Vector{BandWeight}  = BandWeight[])::OptimizationResult
    if isempty(bands)
        return grape_optimize(sys, target, ctrl; config=config)
    else
        return grape_optimize(sys, target, ctrl;
            fidelity_fn = (s,c,t) -> band_selective_fidelity(s,c,t,bands),
            gradient_fn = (s,c,t) -> band_selective_gradient(s,c,t,bands),
            config      = config)
    end
end

# ============================================================================
# optimcon overload: HeteronuclearSystem
# ============================================================================

"""
    optimcon(sys::HeteronuclearSystem, target::QuantumTarget, ctrl::ControlSequence;
             offsets_hz, J_couplings, config, bands) -> OptimizationResult

Heteronuclear liquid-state NMR optimal control.
"""
function optimcon(sys::HeteronuclearSystem,
                  target::QuantumTarget,
                  ctrl::ControlSequence;
                  offsets_hz::Vector{Float64}                     = zeros(sum(sys.spins_per_subsystem)),
                  J_couplings::Dict{Tuple{Int,Int},Float64}       = Dict{Tuple{Int,Int},Float64}(),
                  config::GRAPEConfig                             = GRAPEConfig(),
                  bands::Vector{BandWeight}                       = BandWeight[])::OptimizationResult
    H_drift_actual = hamiltonian(sys; offsets_hz=offsets_hz, J_couplings=J_couplings)
    qs = QuantumSystem(H_drift_actual, sys.H_controls, sys.dim, sys.n_controls, Dict{String,Any}())
    if isempty(bands)
        return grape_optimize(qs, target, ctrl; config=config)
    else
        return grape_optimize(qs, target, ctrl;
            fidelity_fn = (s,c,t) -> band_selective_fidelity(s,c,t,bands),
            gradient_fn = (s,c,t) -> band_selective_gradient(s,c,t,bands),
            config      = config)
    end
end
