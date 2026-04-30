# Types/BlochSystem.jl
# Bloch equation system for MRI optimal control.
using LinearAlgebra

"""
    BlochIsochromat

Single spin isochromat (voxel) in an MRI experiment.

# Fields
- `position`  — spatial position [x,y,z] in meters
- `T1`        — longitudinal relaxation time (s)
- `T2`        — transverse relaxation time (s)
- `delta_B0`  — local B0 offset (T)
- `rho_0`     — proton density (arbitrary units, used as fidelity weight)
- `M0`        — equilibrium magnetization [Mx,My,Mz]
"""
struct BlochIsochromat
    position  :: NTuple{3,Float64}
    T1        :: Float64
    T2        :: Float64
    delta_B0  :: Float64
    rho_0     :: Float64
    M0        :: NTuple{3,Float64}
end

"""
    GradientSystem

MRI gradient hardware parameters.

# Fields
- `max_amplitude_Tm`  — maximum gradient amplitude (T/m)
- `max_slew_Tms`      — maximum slew rate (T/m/s)
- `raster_time_s`     — gradient raster time (s)
"""
struct GradientSystem
    max_amplitude_Tm :: Float64
    max_slew_Tms     :: Float64
    raster_time_s    :: Float64
end

"""
    BlochSystem <: AbstractQuantumSystem

Collection of spin isochromats under MRI excitation.

# Fields
- `isochromats`     — vector of BlochIsochromat
- `n_isochromats`   — length(isochromats)
- `gamma`           — gyromagnetic ratio (rad/s/T); default ¹H
- `gradient_system` — hardware gradient constraints
- `B1_max_tesla`    — maximum B1 amplitude (T)
- `SAR_limit_Wkg`   — SAR limit (W/kg)
- `dim`             — 3 (dummy for interface)
- `H_drift`, `H_controls`, `n_controls` — dummy fields for interface compatibility
"""
struct BlochSystem <: AbstractQuantumSystem
    isochromats     :: Vector{BlochIsochromat}
    n_isochromats   :: Int
    gamma           :: Float64
    gradient_system :: GradientSystem
    B1_max_tesla    :: Float64
    SAR_limit_Wkg   :: Float64
    dim             :: Int
    H_drift         :: Matrix{ComplexF64}
    H_controls          :: Vector{Matrix{ComplexF64}}
    n_controls      :: Int
end

const GAMMA_1H_BLOCH = 2π * 42.577e6  # rad/s/T

"""
    bloch_system(isochromats; gamma, gradient_system, B1_max_tesla, SAR_limit_Wkg)

Construct a BlochSystem from a list of isochromats.
"""
function bloch_system(isochromats::Vector{BlochIsochromat};
                      gamma::Float64           = GAMMA_1H_BLOCH,
                      gradient_system::GradientSystem = GradientSystem(40e-3, 200.0, 10e-6),
                      B1_max_tesla::Float64    = 15e-6,
                      SAR_limit_Wkg::Float64   = 3.2)::BlochSystem
    n = length(isochromats)
    dummy_H = zeros(ComplexF64, 3, 3)
    return BlochSystem(isochromats, n, gamma, gradient_system,
                       B1_max_tesla, SAR_limit_Wkg, 3,
                       dummy_H, Matrix{ComplexF64}[], 5)
end

"""
    MRIControlSequence

Control sequence for MRI: complex B1 field + 3-axis gradients.

# Fields
- `B1`          — RF waveform, size [2 × n_steps] (B1x, B1y in Tesla)
- `G`           — gradient waveform, size [3 × n_steps] (Gx,Gy,Gz in T/m)
- `dt`          — dwell time (s)
- `total_time`  — n_steps * dt (s)
- `n_steps`     — number of time points
- `amplitudes`  — [5 × n_steps] vcat of B1 and G for interface compatibility
"""
struct MRIControlSequence
    B1          :: Matrix{Float64}
    G           :: Matrix{Float64}
    dt          :: Float64
    total_time  :: Float64
    n_steps     :: Int
    amplitudes  :: Matrix{Float64}
end

"""
    mri_control_sequence(B1, G, dt) -> MRIControlSequence
"""
function mri_control_sequence(B1::Matrix{Float64}, G::Matrix{Float64}, dt::Float64)::MRIControlSequence
    size(B1,1) == 2 || throw(ArgumentError("B1 must be 2×n_steps"))
    size(G,1)  == 3 || throw(ArgumentError("G must be 3×n_steps"))
    size(B1,2) == size(G,2) || throw(DimensionMismatch("B1 and G must have same n_steps"))
    n = size(B1, 2)
    return MRIControlSequence(B1, G, dt, dt*n, n, vcat(B1, G))
end
