# Application/QuantumComputing/NoiseModels/Markovian.jl
# Markovian (memoryless) open-system noise models for quantum optimal control.
#
# Markovian noise is described by the Lindblad master equation:
#
#   dρ/dt = −i[H(t), ρ]  +  Σ_k γ_k D[L_k](ρ)
#
# where D[L](ρ) = LρL† − ½{L†L, ρ} is the Lindblad dissipator.
#
# Provides:
#   MarkovianNoise             — struct: T1/T2/dephasing specification
#   markovian_noise(...)       — constructor
#   depolarizing_channel(...)  — depolarizing Lindblad operators
#   amplitude_damping(...)     — amplitude damping (T1) Lindblad operators
#   phase_damping(...)         — pure dephasing (T2*) Lindblad operators
#   lindblad_optimcon(...)     — optimal control with Lindblad propagation

using LinearAlgebra

# ============================================================================
# Jump operator constructors
# ============================================================================

"""
    amplitude_damping(T1_s, dim=2) -> (Vector{Matrix{ComplexF64}}, Vector{Float64})

Lindblad jump operators for amplitude damping (energy relaxation, T₁ process)
on a d-level system.

For a d-level system the T1 process corresponds to all downward transitions:
    L_{k→j} = |j⟩⟨k|,  k > j,  rate γ = 1/T1

Returns `(jump_ops, rates)` tuple.

# Arguments
- `T1_s` — amplitude decay time (s)
- `dim`  — Hilbert space dimension (default 2 for qubit)

# Example
```julia
Ls, γs = amplitude_damping(50e-6)
qs = lindblad_system_from_jump_ops(H, Ls, γs, H_controls)
```
"""
function amplitude_damping(T1_s :: Float64,
                             dim  :: Int = 2
                             )::Tuple{Vector{Matrix{ComplexF64}}, Vector{Float64}}
    @assert T1_s > 0.0 "T1_s must be positive"
    ops   = Matrix{ComplexF64}[]
    rates = Float64[]
    for k in 1:dim, j in 1:(k-1)
        op = zeros(ComplexF64, dim, dim)
        op[j, k] = 1.0
        push!(ops, op)
        push!(rates, 1.0 / T1_s)
    end
    return ops, rates
end

"""
    phase_damping(T2_s, T1_s=Inf, dim=2) -> (Vector{Matrix{ComplexF64}}, Vector{Float64})

Lindblad jump operators for pure dephasing (T₂* or T₂ process).

The pure dephasing rate is γ_φ = 1/T₂ − 1/(2T₁).
The jump operator is L = n̂ (number/population operator), whose diagonal
elements encode the dephasing of off-diagonal coherences.

# Arguments
- `T2_s` — total coherence time (s): 1/T₂ = 1/(2T₁) + 1/T₂*
- `T1_s` — amplitude decay time (s); used to compute pure dephasing rate (default Inf)
- `dim`  — Hilbert space dimension (default 2)
"""
function phase_damping(T2_s :: Float64,
                        T1_s  :: Float64 = Inf,
                        dim   :: Int     = 2
                        )::Tuple{Vector{Matrix{ComplexF64}}, Vector{Float64}}
    @assert T2_s > 0.0 "T2_s must be positive"
    γ1 = isfinite(T1_s) && T1_s > 0.0 ? 1.0 / T1_s : 0.0
    γ2 = 1.0 / T2_s
    γφ = max(γ2 - γ1 / 2.0, 0.0)
    iszero(γφ) && return (Matrix{ComplexF64}[], Float64[])

    # Dephasing operator: diag(0, 1, 2, …, d−1)
    nhat = zeros(ComplexF64, dim, dim)
    for k in 1:dim
        nhat[k, k] = Float64(k - 1)
    end
    return ([nhat], [γφ])
end

"""
    depolarizing_channel(p, dim=2) -> (Vector{Matrix{ComplexF64}}, Vector{Float64})

Lindblad jump operators for a depolarizing channel with error rate `p` per
gate/time step.

The depolarizing channel maps ρ → (1−p) ρ + p/d I.
In Lindblad form this corresponds to d²−1 traceless Hermitian jump operators
(generalised Pauli operators) each with rate γ = p / ((d²−1) × dt):

For a single qubit (d=2): X, Y, Z Pauli operators, each with rate γ = p/3.

**Note**: The rates returned are per-unit-time (rad/s = 1/s). To obtain a
per-gate depolarizing rate, divide `p` by the gate duration.

# Arguments
- `p`    — depolarizing probability per unit time (1/s)
- `dim`  — Hilbert space dimension (default 2)

# Example
```julia
Ls, γs = depolarizing_channel(0.001 / 50e-9)  # 0.1% error in 50 ns
```
"""
function depolarizing_channel(p   :: Float64,
                                dim :: Int = 2
                                )::Tuple{Vector{Matrix{ComplexF64}}, Vector{Float64}}
    @assert p >= 0.0 "depolarizing probability must be non-negative"
    iszero(p) && return (Matrix{ComplexF64}[], Float64[])

    ops   = Matrix{ComplexF64}[]
    rates = Float64[]

    if dim == 2
        # Single qubit: X, Y, Z Paulis
        push!(ops, ComplexF64[0 1; 1 0])          # X
        push!(ops, ComplexF64[0 -im; im 0])       # Y
        push!(ops, ComplexF64[1 0; 0 -1])         # Z
        γ = p / 3.0
        append!(rates, fill(γ, 3))
    else
        # General: construct d²−1 traceless Hermitian basis (generalised Paulis)
        # Diagonal generators: √(2/(k(k+1))) diag(1,…,1,−k,0,…,0)
        for k in 1:(dim-1)
            op = zeros(ComplexF64, dim, dim)
            for j in 1:k
                op[j, j] = sqrt(2.0 / (k * (k + 1)))
            end
            op[k+1, k+1] = -k * sqrt(2.0 / (k * (k + 1)))
            push!(ops, op)
            push!(rates, p / Float64(dim^2 - 1))
        end
        # Off-diagonal generators: symmetric and anti-symmetric pairs
        for r in 1:dim, c in (r+1):dim
            op_s = zeros(ComplexF64, dim, dim)
            op_s[r, c] = 1.0 / sqrt(2.0)
            op_s[c, r] = 1.0 / sqrt(2.0)
            push!(ops, op_s)
            push!(rates, p / Float64(dim^2 - 1))

            op_a = zeros(ComplexF64, dim, dim)
            op_a[r, c] = -im / sqrt(2.0)
            op_a[c, r] =  im / sqrt(2.0)
            push!(ops, op_a)
            push!(rates, p / Float64(dim^2 - 1))
        end
    end
    return ops, rates
end

# ============================================================================
# MarkovianNoise struct
# ============================================================================

"""
    MarkovianNoise

Convenience container for a complete Markovian noise model.

# Fields
- `jump_ops`    — Vector{Matrix{ComplexF64}} Lindblad jump operators
- `decay_rates` — Vector{Float64} corresponding rates γ_k (rad/s)
- `dim`         — Hilbert space dimension N
- `description` — String: human-readable noise model label
"""
struct MarkovianNoise
    jump_ops    :: Vector{Matrix{ComplexF64}}
    decay_rates :: Vector{Float64}
    dim         :: Int
    description :: String
end

"""
    markovian_noise(dim; T1_s=Inf, T2_s=Inf, depol_rate=0.0,
                    description="") -> MarkovianNoise

Build a [`MarkovianNoise`](@ref) from standard physical noise parameters for a
d-level system.  Combines amplitude damping, pure dephasing, and depolarizing
contributions.

# Arguments
- `dim`         — Hilbert space dimension
- `T1_s`        — amplitude decay time (s); `Inf` to omit
- `T2_s`        — total coherence time (s); `Inf` to omit
- `depol_rate`  — depolarizing rate (1/s); 0 to omit
- `description` — optional label

# Example
```julia
noise = markovian_noise(2; T1_s=50e-6, T2_s=30e-6)
qs = lindblad_system_from_jump_ops(H, noise.jump_ops, noise.decay_rates, Hctrls)
```
"""
function markovian_noise(dim          :: Int;
                          T1_s         :: Float64 = Inf,
                          T2_s         :: Float64 = Inf,
                          depol_rate   :: Float64 = 0.0,
                          description  :: String  = "")::MarkovianNoise
    ops   = Matrix{ComplexF64}[]
    rates = Float64[]

    if isfinite(T1_s) && T1_s > 0.0
        Ls, γs = amplitude_damping(T1_s, dim)
        append!(ops,   Ls)
        append!(rates, γs)
    end
    if isfinite(T2_s) && T2_s > 0.0
        Ls, γs = phase_damping(T2_s, T1_s, dim)
        append!(ops,   Ls)
        append!(rates, γs)
    end
    if depol_rate > 0.0
        Ls, γs = depolarizing_channel(depol_rate, dim)
        append!(ops,   Ls)
        append!(rates, γs)
    end

    desc = isempty(description) ?
        "Markovian noise: T1=$(T1_s*1e6) µs, T2=$(T2_s*1e6) µs" : description

    return MarkovianNoise(ops, rates, dim, desc)
end

# ============================================================================
# Lindblad-based optimal control entry point
# ============================================================================

"""
    lindblad_optimcon(H_drift, H_controls, noise::MarkovianNoise,
                      target, ctrl; config, pwr_levels) -> OptimizationResult

Optimal control with full Lindblad open-system propagation.

Constructs a Liouville-space [`QuantumSystem`](@ref) from the drift Hamiltonian
and the Markovian noise model, then runs GRAPE in the N²-dimensional Liouville
space.

# Arguments
- `H_drift`     — N×N drift Hamiltonian (rad/s)
- `H_controls`  — Vector of N×N control Hamiltonians
- `noise`       — [`MarkovianNoise`](@ref) model
- `target`      — [`QuantumTarget`](@ref)
- `ctrl`        — initial [`ControlSequence`](@ref)
- `config`      — `GRAPEConfig`
- `pwr_levels`  — drive power scalings per channel (default: ones)

# Returns
`OptimizationResult`

# Example
```julia
noise = markovian_noise(2; T1_s=50e-6, T2_s=30e-6)
result = lindblad_optimcon(H, [Hx, Hy], noise, state_tgt, ctrl)
```
"""
function lindblad_optimcon(H_drift    :: Matrix{ComplexF64},
                             H_controls :: Vector{Matrix{ComplexF64}},
                             noise      :: MarkovianNoise,
                             target     :: QuantumTarget,
                             ctrl       :: ControlSequence;
                             config     :: GRAPEConfig    = GRAPEConfig(),
                             pwr_levels :: Vector{Float64} = ones(length(H_controls))
                             )::OptimizationResult
    H_ctrl_scaled = _scale_controls(H_controls, pwr_levels)
    qs = lindblad_system_from_jump_ops(H_drift, noise.jump_ops, noise.decay_rates,
                                        H_ctrl_scaled)
    return grape_optimize(qs, target, ctrl; config=config)
end
