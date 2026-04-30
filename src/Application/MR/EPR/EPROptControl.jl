# Application/MR/EPR/EPROptControl.jl
# Optimal control for pulsed EPR via the generic GRAPE interface.
using LinearAlgebra

"""
    optimcon(sys::EPRSpinSystem, target, ctrl;
             config, bands, orientations) -> OptimizationResult

Run band-selective GRAPE for EPR, optionally powder-averaged.
"""
function optimcon(sys::EPRSpinSystem,
                  target::QuantumTarget,
                  ctrl::ControlSequence;
                  config::GRAPEConfig        = GRAPEConfig(),
                  bands::Vector{BandWeight}   = BandWeight[],
                  orientations::Union{Nothing,Vector{NTuple{4,Float64}}} = nothing)::OptimizationResult

    # Default path: no bands, no orientations — let grape_optimize use its built-ins.
    isempty(bands) && orientations === nothing &&
        return grape_optimize(sys, target, ctrl; config=config)

    rotate_fn = (α, β, γ) -> _epr_oriented_sys(sys, α, β, γ)

    # Pick base (non-orientation-averaged) fidelity / gradient hooks.
    if isempty(bands)
        base_fid  = nothing
        base_grad = (s, c, t) -> compute_grape_gradient(s, c, t)
    else
        base_fid  = (s, c, t) -> band_selective_fidelity(s, c, t, bands)
        base_grad = (s, c, t) -> band_selective_gradient(s, c, t, bands)
    end

    grad_fn = _wrap_orient_gradient(base_grad, orientations, rotate_fn)
    if base_fid === nothing
        return grape_optimize(sys, target, ctrl;
                              gradient_fn = grad_fn, config = config)
    else
        fid_fn = _wrap_orient_fidelity(base_fid, orientations, rotate_fn)
        return grape_optimize(sys, target, ctrl;
                              fidelity_fn = fid_fn,
                              gradient_fn = grad_fn,
                              config      = config)
    end
end

function _epr_oriented_hamiltonian(sys::EPRSpinSystem, α::Float64, β::Float64, γ::Float64)::Matrix{ComplexF64}
    # Rotate g-tensor: gzz in lab frame (secular approximation)
    sinβ, cosβ = sin(β), cos(β)
    cosα, sinα = cos(α), sin(α)
    gzz_lab = sys.g_vals[1]*sinβ^2*cosα^2 + sys.g_vals[2]*sinβ^2*sinα^2 + sys.g_vals[3]*cosβ^2
    ω_S_lab = _EPR_μ_B_SI * sys.B0_tesla * gzz_lab / _EPR_hbar_SI
    H = (ω_S_lab - 2π*sys.mw_freq_hz) .* sys.S_ops[3]
    for (k, Ak) in enumerate(sys.A_vals)
        Azz_lab = Ak[1]*sinβ^2*cosα^2 + Ak[2]*sinβ^2*sinα^2 + Ak[3]*cosβ^2
        H .+= (Azz_lab * _MHZ_TO_RADS) .* (sys.S_ops[3] * sys.I_ops[k][3])
    end
    return H
end

function _epr_oriented_sys(sys::EPRSpinSystem, α::Float64, β::Float64, γ::Float64)::EPRSpinSystem
    H_new = _epr_oriented_hamiltonian(sys, α, β, γ)
    return EPRSpinSystem(sys.S_electron, sys.I_nuclei, sys.dim, sys.B0_tesla,
                         sys.mw_freq_hz, sys.g_vals, sys.g_euler,
                         sys.A_vals, sys.A_euler, sys.D_mhz, sys.E_mhz,
                         sys.S_ops, sys.I_ops, H_new, sys.H_controls, sys.n_controls)
end
