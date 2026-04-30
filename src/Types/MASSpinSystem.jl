# Types/MASSpinSystem.jl
# Magic-Angle Spinning (MAS) solid-state NMR spin system.
using LinearAlgebra

"""
    CSATensor

Chemical shift anisotropy tensor in its principal axis system (PAS).

# Fields
- `delta_iso`       — isotropic shift (rad/s)
- `delta_aniso`     — anisotropy (rad/s), Haeberlen convention
- `eta`             — asymmetry parameter ∈ [0,1]
- `euler_PAS_to_MF` — (α,β,γ) Euler angles rotating PAS to molecular frame (rad)
"""
struct CSATensor
    delta_iso    :: Float64
    delta_aniso  :: Float64
    eta          :: Float64
    euler_PAS_to_MF :: NTuple{3,Float64}
end

"""
    DipolarCoupling

Dipole–dipole coupling between two spins.

# Fields
- `spin_i`, `spin_j`  — 1-based spin indices
- `b_ij`              — dipolar coupling constant (rad/s)
- `euler_DD_to_MF`    — (α,β,γ) Euler angles rotating DD-PAS to molecular frame (rad)
"""
struct DipolarCoupling
    spin_i :: Int
    spin_j :: Int
    b_ij   :: Float64
    euler_DD_to_MF :: NTuple{3,Float64}
end

"""
    MASSpinSystem <: AbstractQuantumSystem

Solid-state NMR spin system under Magic-Angle Spinning.

Under MAS the rank-2 interaction tensors are modulated as:

    H_int(t) = Σ_{m=-2}^{2} H_m exp(im ωr t)

where the H_m Fourier components are precomputed and stored in `H_fourier`.

# Fields
- `base_system`   — underlying liquid-state MRSpinSystem
- `csa`           — CSA tensor per spin
- `dipolar`       — dipolar couplings between spin pairs
- `omega_r`       — MAS rotor frequency (rad/s)
- `rotor_period_s`— 2π/omega_r (s)
- `H_fourier`     — Dict m => H_m matrix for m ∈ {-2,-1,0,1,2}
- `H_drift`       — zero matrix (time-dependent drift via H_fourier)
- `H_controls`        — control operators from base_system [Ix_1, Iy_1, ...]
- `dim`, `n_controls`
"""
struct MASSpinSystem <: AbstractQuantumSystem
    base_system    :: MRSpinSystem
    csa            :: Vector{CSATensor}
    dipolar        :: Vector{DipolarCoupling}
    omega_r        :: Float64
    rotor_period_s :: Float64
    H_fourier      :: Dict{Int,Matrix{ComplexF64}}
    H_drift        :: Matrix{ComplexF64}
    H_controls         :: Vector{Matrix{ComplexF64}}
    dim            :: Int
    n_controls     :: Int
end

"""
    mas_spin_system(base_system, omega_r_hz; csa, dipolar) -> MASSpinSystem

Construct a MASSpinSystem. omega_r_hz is the MAS rate in Hz.
"""
function mas_spin_system(base_system::MRSpinSystem,
                         omega_r_hz::Float64;
                         csa::Vector{CSATensor}        = CSATensor[],
                         dipolar::Vector{DipolarCoupling} = DipolarCoupling[])::MASSpinSystem
    omega_r = 2π * omega_r_hz
    rotor_period_s = 2π / omega_r

    dim = base_system.dim
    H_drift_zero = zeros(ComplexF64, dim, dim)

    H_fourier = Dict{Int,Matrix{ComplexF64}}(m => zeros(ComplexF64, dim, dim) for m in -2:2)

    _add_csa_fourier!(H_fourier, base_system, csa)
    _add_dipolar_fourier!(H_fourier, base_system, dipolar)

    # Build control operators: [Ix_1, Iy_1, Ix_2, Iy_2, ...]
    H_controls = Matrix{ComplexF64}[]
    for k in 1:base_system.n_spins
        push!(H_controls, base_system.Ix[k])
        push!(H_controls, base_system.Iy[k])
    end

    return MASSpinSystem(base_system, csa, dipolar, omega_r, rotor_period_s,
                         H_fourier, H_drift_zero, H_controls,
                         dim, length(H_controls))
end

# Internal: add CSA Fourier components
function _add_csa_fourier!(H_fourier::Dict{Int,Matrix{ComplexF64}},
                            sys::MRSpinSystem,
                            tensors::Vector{CSATensor})
    n = sys.n_spins
    for (i, csa) in enumerate(tensors)
        i <= n || break
        Iz_i = sys.Iz[i]

        T20_PAS   =  csa.delta_aniso * sqrt(2/3)

        α, β, γ = csa.euler_PAS_to_MF
        cosβ = cos(β)
        sinβ = sin(β)

        # Secular (m=0) term
        H_fourier[0] .+= (csa.delta_iso + T20_PAS * (3*cosβ^2 - 1)/2) .* Iz_i

        if sinβ > 1e-12
            # m=±1 sidebands
            c1 = -sqrt(3/2) * T20_PAS * sinβ * cosβ
            H_fourier[1]  .+= c1 * exp(-1im*α) .* Iz_i
            H_fourier[-1] .+= conj(c1 * exp(-1im*α)) .* Iz_i
            # m=±2 sidebands
            c2 = sqrt(3/8) * T20_PAS * sinβ^2
            H_fourier[2]  .+= c2 * exp(-2im*α) .* Iz_i
            H_fourier[-2] .+= conj(c2 * exp(-2im*α)) .* Iz_i
        end
    end
end

# Internal: add dipolar Fourier components
function _add_dipolar_fourier!(H_fourier::Dict{Int,Matrix{ComplexF64}},
                                sys::MRSpinSystem,
                                dipolar::Vector{DipolarCoupling})
    n = sys.n_spins
    for dd in dipolar
        i, j = dd.spin_i, dd.spin_j
        (i <= n && j <= n) || continue
        Iz_i = sys.Iz[i]
        Iz_j = sys.Iz[j]
        IzIz = Iz_i * Iz_j

        α, β, γ = dd.euler_DD_to_MF
        cosβ = cos(β)
        sinβ = sin(β)

        # Secular dipolar: b_ij (3cos²β-1)/2 * 2 IzIz
        d_secular = dd.b_ij * (3*cosβ^2 - 1) / 2
        H_fourier[0] .+= 2 * d_secular .* IzIz

        if sinβ > 1e-12
            c1 = -dd.b_ij * sqrt(6.0) * sinβ * cosβ
            H_fourier[1]  .+= c1 * exp(-1im*(α+γ)) .* IzIz
            H_fourier[-1] .+= conj(c1 * exp(-1im*(α+γ))) .* IzIz
            c2 = dd.b_ij * sqrt(6.0)/4 * sinβ^2
            H_fourier[2]  .+= c2 * exp(-2im*(α+γ)) .* IzIz
            H_fourier[-2] .+= conj(c2 * exp(-2im*(α+γ))) .* IzIz
        end
    end
end

"""
    build_mas_hamiltonian(sys::MASSpinSystem, t::Float64) -> Matrix{ComplexF64}

Evaluate the time-dependent MAS drift Hamiltonian at time t:
    H_drift(t) = Σ_{m=-2}^{2} H_m exp(im ωr t)
"""
function build_mas_hamiltonian(sys::MASSpinSystem, t::Float64)::Matrix{ComplexF64}
    H = zeros(ComplexF64, sys.dim, sys.dim)
    @inbounds for m in -2:2
        phase = exp(im * m * sys.omega_r * t)
        LinearAlgebra.axpy!(phase, sys.H_fourier[m], H)
    end
    return H
end

"""
    rotate_spin_system(sys::MASSpinSystem, α, β, γ) -> MASSpinSystem

Return a copy of `sys` with Fourier components recomputed for crystal orientation (α,β,γ).
Used for powder averaging.
"""
function rotate_spin_system(sys::MASSpinSystem, α::Float64, β::Float64, γ::Float64)::MASSpinSystem
    new_csa = [CSATensor(c.delta_iso, c.delta_aniso, c.eta,
                         (c.euler_PAS_to_MF[1]+α, c.euler_PAS_to_MF[2]+β, c.euler_PAS_to_MF[3]+γ))
               for c in sys.csa]
    new_dip = [DipolarCoupling(d.spin_i, d.spin_j, d.b_ij,
                               (d.euler_DD_to_MF[1]+α, d.euler_DD_to_MF[2]+β, d.euler_DD_to_MF[3]+γ))
               for d in sys.dipolar]
    return mas_spin_system(sys.base_system, sys.omega_r/(2π);
                           csa=new_csa, dipolar=new_dip)
end
