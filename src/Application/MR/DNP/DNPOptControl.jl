# Application/MR/DNP/DNPOptControl.jl
# Optimal control for Dynamic Nuclear Polarization (DNP).
using LinearAlgebra

"""
    optimcon_dnp(sys::DNPSpinSystem, ctrl::ControlSequence;
                 config::GRAPEConfig,
                 orientations=nothing) -> OptimizationResult

Run GRAPE to maximize nuclear polarization via DNP.
"""
function optimcon_dnp(sys::DNPSpinSystem,
                      ctrl::ControlSequence;
                      config::GRAPEConfig = GRAPEConfig(),
                      orientations::Union{Nothing,Vector{NTuple{4,Float64}}} = nothing,
                      use_lindblad::Bool = false,
                      T1_electron_s::Float64 = Inf,
                      T2_electron_s::Float64 = Inf,
                      T1_nuclear_s::Union{Float64,Vector{Float64}} = Inf,
                      cross_relax_rate::Float64 = 0.0,
                      )::OptimizationResult
    use_lindblad && return grape_dnp_lindblad_kernel(sys, ctrl;
        T1_electron_s   = T1_electron_s,
        T2_electron_s   = T2_electron_s,
        T1_nuclear_s    = T1_nuclear_s,
        cross_relax_rate = cross_relax_rate,
        config          = config,
    )
    rotate_fn = (α, β, γ) -> _rotate_dnp_system(sys, α, β, γ)
    base_fid  = (s, c, _) -> dnp_polarization_fidelity(s, c)
    base_grad = (s, c, _) -> compute_grape_gradient(s, c, Val{:DNP})

    fid_fn  = _wrap_orient_fidelity(base_fid,  orientations, rotate_fn)
    grad_fn = _wrap_orient_gradient(base_grad, orientations, rotate_fn)

    return grape_optimize(sys, nothing, ctrl;
                          fidelity_fn = fid_fn,
                          gradient_fn = grad_fn,
                          config      = config)
end

"""
    _rotate_dnp_system(sys::DNPSpinSystem, α, β, γ) -> DNPSpinSystem

Return a copy of sys with Euler angles shifted for orientation (α,β,γ).
"""
function _rotate_dnp_system(sys::DNPSpinSystem, α::Float64, β::Float64, γ::Float64)::DNPSpinSystem
    new_csa_e = [CSATensor(c.delta_iso, c.delta_aniso, c.eta,
                            (c.euler_PAS_to_MF[1]+α, c.euler_PAS_to_MF[2]+β, c.euler_PAS_to_MF[3]+γ))
                 for c in sys.csa_electron]
    new_csa_n = [CSATensor(c.delta_iso, c.delta_aniso, c.eta,
                            (c.euler_PAS_to_MF[1]+α, c.euler_PAS_to_MF[2]+β, c.euler_PAS_to_MF[3]+γ))
                 for c in sys.csa_nuclei]
    new_dip = [DipolarCoupling(d.spin_i, d.spin_j, d.b_ij,
                               (d.euler_DD_to_MF[1]+α, d.euler_DD_to_MF[2]+β, d.euler_DD_to_MF[3]+γ))
               for d in sys.dipolar]
    dnp_system(sys.S_electron, sys.I_nuclei, sys.B0_tesla,
               sys.mw_freq_hz, sys.omega_r/(2π);
               g_vals=sys.g_vals, g_euler=sys.g_euler,
               A_vals=sys.A_vals, A_euler=sys.A_euler,
               csa_electron=new_csa_e, csa_nuclei=new_csa_n,
               dipolar=new_dip, rf_freq_hz=sys.rf_freq_hz)
end
