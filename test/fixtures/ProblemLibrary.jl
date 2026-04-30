"""
    ProblemLibrary.jl

Canonical problem library for PULSAR quantum control benchmarks.

Provides ready-to-use problem instances covering:
  - Single-qubit gates (Hadamard, NOT/X, π/2 rotations)
  - Two-qubit gates (CNOT)
  - State transfer problems with known analytic solutions
  - NMR pulse sequence benchmarks (INEPT, spin-echo)
  - Robust control test cases
  - Random unitary targets for numerical benchmarking

Every function returns a triple `(system, target, controls_init)` that can be
passed directly to any PULSAR optimizer:

    sys, tgt, u0 = hadamard_gate_problem()
    result = grape_optimize(sys, tgt, u0; config = GRAPEConfig(max_iter=500))

All frequencies are in rad/s (angular) unless otherwise noted.  Time is in
seconds throughout.  Control amplitudes returned by the library are small
random perturbations around zero so the optimizer has a non-degenerate
starting point.
"""

using LinearAlgebra
using Random

# ============================================================================
# Module-level Pauli matrix constants
# ============================================================================

"""Pauli σx matrix for spin-1/2."""
const σ_x = ComplexF64[0 1; 1 0]

"""Pauli σy matrix for spin-1/2."""
const σ_y = ComplexF64[0 -1im; 1im 0]

"""Pauli σz matrix for spin-1/2."""
const σ_z = ComplexF64[1 0; 0 -1]

"""2×2 identity matrix (σ_0)."""
const σ_0 = ComplexF64[1 0; 0 1]

# ============================================================================
# Spin operator helpers
# ============================================================================

"""
    spin_half_operators() -> Tuple{Matrix, Matrix, Matrix}

Return the three spin-1/2 angular momentum operators (Sx, Sy, Sz) in units
of ħ (i.e. each operator is σ/2 where σ are the Pauli matrices).

# Returns
Tuple `(Sx, Sy, Sz)` each of type `Matrix{ComplexF64}`, size `2 × 2`.

# Example
```julia
Sx, Sy, Sz = spin_half_operators()
```
"""
function spin_half_operators()
    Sx = σ_x / 2
    Sy = σ_y / 2
    Sz = σ_z / 2
    return Sx, Sy, Sz
end

"""
    spin_operators(s::Float64) -> Tuple{Matrix, Matrix, Matrix}

Return the (Sx, Sy, Sz) angular momentum matrices for a general spin-s particle.

# Arguments
- `s` — spin quantum number (0.5, 1.0, 1.5, …).  Must satisfy 2s ∈ ℤ⁺.

# Returns
Tuple `(Sx, Sy, Sz)` each of size `(2s+1) × (2s+1)`, `Matrix{ComplexF64}`.

# Construction
The magnetic quantum numbers m range from +s to -s in integer steps.  The
diagonal (Sz) and off-diagonal (S±) matrix elements follow:

    ⟨m | Sz | m⟩   = m
    ⟨m+1 | S+ | m⟩ = √(s(s+1) - m(m+1))
    Sx = (S+ + S-) / 2
    Sy = (S+ - S-) / (2i)

# Throws
`ArgumentError` if `s < 0.5` or `2s` is not an integer.

# Example
```julia
Sx, Sy, Sz = spin_operators(1.0)   # spin-1 system (d = 3)
```
"""
function spin_operators(s::Float64)
    if s < 0.5 || !isinteger(2s)
        throw(ArgumentError(
            "spin quantum number s = $s is invalid; must be a half-integer ≥ 1/2"))
    end
    d  = Int(2s + 1)
    ms = [s - Float64(i) for i in 0:(d-1)]   # m values from +s to −s

    # Sz is diagonal
    Sz = ComplexF64.(diagm(ms))

    # S+ raising operator: ⟨m+1|S+|m⟩ = √(s(s+1) − m(m+1))
    Splus = zeros(ComplexF64, d, d)
    for col in 1:d
        m   = ms[col]          # ket |m⟩
        row = col - 1          # bra ⟨m+1| (larger m → lower row index)
        if row >= 1
            Splus[row, col] = sqrt(s*(s+1) - m*(m+1))
        end
    end
    Sminus = Splus'            # S- = (S+)†

    Sx = (Splus + Sminus) / 2
    Sy = (Splus - Sminus) / (2im)

    return Sx, Sy, Sz
end

"""
    tensor_product_operators(ops_list::Vector{Matrix{ComplexF64}}) -> Matrix{ComplexF64}

Compute the Kronecker tensor product of a list of matrices.

Evaluates `ops_list[1] ⊗ ops_list[2] ⊗ … ⊗ ops_list[end]` sequentially
from left to right using `kron`.

# Arguments
- `ops_list` — non-empty list of square `ComplexF64` matrices of any sizes.

# Returns
A `Matrix{ComplexF64}` of size `(Π dᵢ) × (Π dᵢ)`.

# Throws
`ArgumentError` if `ops_list` is empty.

# Example
```julia
# Two-qubit σz ⊗ σz
ZZ = tensor_product_operators([σ_z, σ_z])
```
"""
function tensor_product_operators(ops_list::Vector{Matrix{ComplexF64}})::Matrix{ComplexF64}
    isempty(ops_list) && throw(ArgumentError("ops_list must be non-empty"))
    result = ops_list[1]
    for i in 2:length(ops_list)
        result = kron(result, ops_list[i])
    end
    return result
end

# ============================================================================
# Internal helper: embed single-qubit operator into n-qubit space
# ============================================================================

"""
    _embed_qubit_op(op::Matrix{ComplexF64}, n_qubits::Int, qubit_index::Int)
    -> Matrix{ComplexF64}

Embed a 2×2 single-qubit operator into the full `2^n_qubits`-dimensional
Hilbert space, acting on qubit number `qubit_index` (1-based).

Returns `I ⊗ … ⊗ op ⊗ … ⊗ I` where `op` appears at position `qubit_index`.
"""
function _embed_qubit_op(op::Matrix{ComplexF64}, n_qubits::Int, qubit_index::Int)
    parts = Matrix{ComplexF64}[
        i == qubit_index ? op : σ_0
        for i in 1:n_qubits
    ]
    return tensor_product_operators(parts)
end

# ============================================================================
# Problem 1: Hadamard gate
# ============================================================================

"""
    hadamard_gate_problem(; total_time=1e-6, n_timesteps=100, dt=nothing)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Create an instance of the single-qubit Hadamard gate problem.

# System
- H_drift = 0 (on-resonance, no free precession)
- Control Hamiltonians: `2π × [σ_x/2, σ_y/2]` (x and y RF channels, 1 MHz scale)

# Target
The Hadamard unitary:

    U_H = [1  1; 1 -1] / √2

# Initial controls
Small random amplitudes drawn from U(-0.01, 0.01).

# Keyword Arguments
- `total_time`  — total pulse duration in seconds (default `1e-6`)
- `n_timesteps` — number of time slices (default `100`)
- `dt`          — if given, overrides `n_timesteps` to give `round(total_time/dt)` steps
- `seed`        — RNG seed for reproducible initial controls (default `42`)

# Example
```julia
sys, tgt, u0 = hadamard_gate_problem(n_timesteps=50)
result = grape_optimize(sys, tgt, u0; config = GRAPEConfig(max_iter=300))
```
"""
function hadamard_gate_problem(;
    total_time::Float64  = 1e-6,
    n_timesteps::Int     = 100,
    dt::Union{Float64, Nothing} = nothing,
    seed::Int            = 42
)
    # Optionally override n_timesteps from dt
    if dt !== nothing
        n_timesteps = max(1, round(Int, total_time / dt))
    end

    # System definition
    H_drift    = zeros(ComplexF64, 2, 2)
    H_controls = [2π * σ_x / 2, 2π * σ_y / 2]   # 1 MHz amplitude scale
    system     = quantum_system(H_drift, H_controls)

    # Target: Hadamard gate U_H = [1 1; 1 -1] / √2
    U_H    = ComplexF64[1 1; 1 -1] ./ sqrt(2)
    target = unitary_target(U_H)

    # Initial controls: small random amplitudes
    rng          = MersenneTwister(seed)
    ctrl_mat     = 0.01 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt          = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Problem 2: NOT (X) gate
# ============================================================================

"""
    not_gate_problem(; total_time=1e-6, n_timesteps=100, seed=42)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Create a single-qubit NOT (Pauli-X) gate problem with a small resonance offset.

# System
- H_drift = `2π × 200e3 × σ_z / 2`  (200 kHz off-resonance drift)
- Controls: `[2π × σ_x/2, 2π × σ_y/2]`

# Target
The NOT gate: `U_X = σ_x = [0 1; 1 0]`

# Example
```julia
sys, tgt, u0 = not_gate_problem(n_timesteps=80)
```
"""
function not_gate_problem(;
    total_time::Float64 = 1e-6,
    n_timesteps::Int    = 100,
    seed::Int           = 42
)
    # 200 kHz resonance offset (typical in NMR / qubit experiments)
    H_drift    = 2π * 200e3 * σ_z / 2
    H_controls = [2π * σ_x / 2, 2π * σ_y / 2]
    system     = quantum_system(H_drift, H_controls)

    # Target: X gate
    U_X    = ComplexF64[0 1; 1 0]
    target = unitary_target(U_X)

    rng           = MersenneTwister(seed)
    ctrl_mat      = 0.01 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt           = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Problem 3: CNOT gate (two-qubit)
# ============================================================================

"""
    cnot_gate_problem(; total_time=5e-6, n_timesteps=200, seed=42)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Create a two-qubit CNOT gate problem with ZZ coupling.

# System (4-dimensional Hilbert space)
Drift Hamiltonian (in rotating frame):

    H_drift = 2π × [50e3 (σz⊗I)/2 + 30e3 (I⊗σz)/2 + 10e3 (σz⊗σz)/4]

The ZZ term models a longitudinal coupling (e.g. cross-resonance interaction).

Control Hamiltonians: σx⊗I, σy⊗I, I⊗σx, I⊗σy, each scaled by `2π`.

# Target
The CNOT gate:

    CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ σx
         = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]

# Example
```julia
sys, tgt, u0 = cnot_gate_problem(n_timesteps=150)
result = grape_optimize(sys, tgt, u0; config = GRAPEConfig(max_iter=500))
```
"""
function cnot_gate_problem(;
    total_time::Float64 = 5e-6,
    n_timesteps::Int    = 200,
    seed::Int           = 42
)
    I2 = σ_0

    # Single-qubit operators embedded in 4-dim space
    ZI = kron(σ_z, I2)   # σz ⊗ I
    IZ = kron(I2, σ_z)   # I ⊗ σz
    ZZ = kron(σ_z, σ_z)  # σz ⊗ σz
    XI = kron(σ_x, I2)   # σx ⊗ I
    YI = kron(σ_y, I2)   # σy ⊗ I
    IX = kron(I2, σ_x)   # I ⊗ σx
    IY = kron(I2, σ_y)   # I ⊗ σy

    H_drift = 2π * (50e3 .* ZI ./ 2 .+ 30e3 .* IZ ./ 2 .+ 10e3 .* ZZ ./ 4)
    H_controls = [2π .* XI, 2π .* YI, 2π .* IX, 2π .* IY]
    system     = quantum_system(H_drift, H_controls)

    # CNOT target
    U_cnot = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    target = unitary_target(U_cnot)

    rng           = MersenneTwister(seed)
    ctrl_mat      = 0.01 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt           = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Problem 4: State transfer |0⟩ → |1⟩
# ============================================================================

"""
    state_transfer_0_to_1(; total_time=5e-7, n_timesteps=50, seed=42)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Single-qubit state transfer from |0⟩ to |1⟩ (bit-flip / π pulse).

# System
- H_drift = `2π × 1e3 × σ_z / 2`  (1 kHz drift — nearly on resonance)
- Controls: `[2π × σ_x/2, 2π × σ_y/2]`

# Target
The state |1⟩ = [0, 1]ᵀ.

# Known solution
At resonance a π pulse of duration `T` on the x-axis has amplitude `1/(2T)` Hz
(nutation frequency).  The optimizer should find a near-π-pulse solution.

# Example
```julia
sys, tgt, u0 = state_transfer_0_to_1()
result = grape_optimize(sys, tgt, u0; config = GRAPEConfig(max_iter=200))
@show result.fidelity   # should approach 1.0
```
"""
function state_transfer_0_to_1(;
    total_time::Float64 = 5e-7,
    n_timesteps::Int    = 50,
    seed::Int           = 42
)
    # Small drift — optimizer must compensate for it
    H_drift    = 2π * 1e3 * σ_z / 2
    H_controls = [2π * σ_x / 2, 2π * σ_y / 2]
    system     = quantum_system(H_drift, H_controls)

    target = state_target(ComplexF64[0.0, 1.0])

    rng           = MersenneTwister(seed)
    ctrl_mat      = 0.05 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt           = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Problem 5: State transfer |0⟩ → |+⟩
# ============================================================================

"""
    state_transfer_0_to_plus(; total_time=5e-7, n_timesteps=50, seed=42)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Single-qubit state transfer from |0⟩ to |+⟩ = (|0⟩ + |1⟩)/√2 (π/2 pulse).

# System
- H_drift = 0 (on resonance)
- Controls: `[2π × σ_x/2, 2π × σ_y/2]`

# Target
The state |+⟩ = [1, 1]ᵀ / √2.

# Known solution
A π/2 pulse of amplitude `1/(4T)` Hz on the y-axis (Ry(π/2) maps |0⟩ to |+⟩).

# Example
```julia
sys, tgt, u0 = state_transfer_0_to_plus()
result = grape_optimize(sys, tgt, u0; config = GRAPEConfig(max_iter=200))
```
"""
function state_transfer_0_to_plus(;
    total_time::Float64 = 5e-7,
    n_timesteps::Int    = 50,
    seed::Int           = 42
)
    H_drift    = zeros(ComplexF64, 2, 2)
    H_controls = [2π * σ_x / 2, 2π * σ_y / 2]
    system     = quantum_system(H_drift, H_controls)

    psi_plus = ComplexF64[1.0, 1.0] ./ sqrt(2)
    target   = state_target(psi_plus)

    rng           = MersenneTwister(seed)
    ctrl_mat      = 0.05 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt           = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Problem 6: INEPT NMR pulse sequence
# ============================================================================

"""
    inept_problem(; total_time=5e-3, n_timesteps=500, seed=42)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Insensitive Nuclei Enhancement by Polarization Transfer (INEPT) benchmark.

A classic two-spin 1H–13C NMR polarization transfer experiment used widely in
NMR spectroscopy.  The goal is to transfer 1H longitudinal magnetization Iz(H)
to 13C transverse magnetization Ix(C).

# Physical parameters
- 1H chemical shift (rotating frame): δ_H = 100 Hz
- 13C chemical shift (rotating frame): δ_C = 50 Hz
- One-bond 1H–13C J-coupling: J_HC = 145 Hz (scalar coupling)
- Magnetic field B₀ is implicit (rotating frame removes Larmor precession)

# System (4-dimensional: ¹H ⊗ ¹³C)
Drift Hamiltonian:

    H_drift = 2π × [δ_H Iz(H)⊗I + δ_C I⊗Iz(C) + J/4 × σz(H)⊗σz(C)]

Controls: x and y RF pulses on each spin (4 channels total):
  channel 1: σx(H)⊗I, channel 2: σy(H)⊗I,
  channel 3: I⊗σx(C), channel 4: I⊗σy(C)

# Target
A unitary that maps Iz(H)⊗I → I⊗Ix(C) (INEPT transfer).

The target unitary is constructed as the INEPT sequence propagator computed
from the exact product-operator solution:

    U_INEPT = Rx(π, H) × Ry(π/2, H) × τ_evolution × Ry(π/2, C)

For simplicity in this benchmark the target is set to the operator that
implements the exact INEPT coherence transfer, represented as a 4×4 unitary.
We use the INEPT product-operator solution as a reference target.

# Example
```julia
sys, tgt, u0 = inept_problem(n_timesteps=250)
result = grape_optimize(sys, tgt, u0; config = GRAPEConfig(max_iter=1000))
```
"""
function inept_problem(;
    total_time::Float64 = 5e-3,
    n_timesteps::Int    = 500,
    seed::Int           = 42
)
    # Physical parameters (Hz)
    δ_H  = 100.0    # 1H chemical shift offset (Hz)
    δ_C  = 50.0     # 13C chemical shift offset (Hz)
    J_HC = 145.0    # one-bond J-coupling (Hz)

    I2 = σ_0

    # Spin operators in full 4-dim space
    Ix_H = kron(σ_x / 2, I2)
    Iy_H = kron(σ_y / 2, I2)
    Iz_H = kron(σ_z / 2, I2)
    Ix_C = kron(I2, σ_x / 2)
    Iy_C = kron(I2, σ_y / 2)
    Iz_C = kron(I2, σ_z / 2)
    ZZ   = kron(σ_z, σ_z)   # = 4 * Iz_H * Iz_C

    # Drift: Zeeman + isotropic scalar coupling (only the Iz⊗Iz part survives
    # in the doubly-rotating frame for a scalar-coupled pair)
    H_drift = 2π * (δ_H .* Iz_H .+ δ_C .* Iz_C .+ (J_HC / 4) .* ZZ)

    # Control Hamiltonians (2π factor so amplitude unit = Hz)
    H_controls = [
        2π .* Ix_H,   # x pulse on 1H
        2π .* Iy_H,   # y pulse on 1H
        2π .* Ix_C,   # x pulse on 13C
        2π .* Iy_C,   # y pulse on 13C
    ]
    system = quantum_system(H_drift, H_controls)

    # Target: INEPT coherence transfer.
    # We construct the exact INEPT propagator using product-operator theory:
    # The net effect is a unitary U such that:
    #   U (Iz_H) U† = Ix_C  (polarization transfer)
    # We parameterize as the product of hard-pulse propagators at the INEPT
    # transfer delay τ = 1/(4J).
    τ   = 1.0 / (4 * J_HC)   # INEPT delay (seconds)

    # Build the INEPT target unitary analytically via matrix exponentials:
    # Step 1: π/2 pulse on 1H (y-axis)
    U1 = exp(-im * (π/2) .* Iy_H)
    # Step 2: free evolution for time τ under J-coupling
    U2 = exp(-im * τ .* H_drift)
    # Step 3: π pulses on both spins (refocus chemical shifts, keep J-evolution)
    U3 = exp(-im * π .* Iz_H) * exp(-im * π .* Iz_C)
    # Step 4: free evolution for another τ
    U4 = exp(-im * τ .* H_drift)
    # Step 5: π/2 pulse on 13C (y-axis)
    U5 = exp(-im * (π/2) .* Iy_C)

    U_inept = U5 * U4 * U3 * U2 * U1

    target = unitary_target(U_inept)

    # Initial controls: small random amplitudes
    rng           = MersenneTwister(seed)
    ctrl_mat      = 0.01 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt           = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Problem 7: Hahn spin-echo
# ============================================================================

"""
    spin_echo_problem(; total_time=1e-3, n_timesteps=200, seed=42)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Hahn spin-echo refocusing problem.

A single spin in the presence of an unknown resonance offset (inhomogeneous
broadening) is refocused by a π pulse at the midpoint of the echo sequence.
The optimal control task is to find a broadband π-pulse that refocuses
transverse magnetization at the echo time for a range of offsets.

For this single-system formulation the target is a broadband π pulse:
  - Drift: `H_drift = 2π × 500 Hz × σz / 2` (500 Hz resonance offset)
  - Controls: `[2π × σx/2, 2π × σy/2]`
  - Target: inversion unitary `exp(-i π Ix) = -i σx`

# Note
For a proper broadband spin-echo robust to offset uncertainty, use
`robust_optimize` with an ensemble of offset values.

# Example
```julia
sys, tgt, u0 = spin_echo_problem()
result = grape_optimize(sys, tgt, u0; config = GRAPEConfig(max_iter=300))
```
"""
function spin_echo_problem(;
    total_time::Float64 = 1e-3,
    n_timesteps::Int    = 200,
    seed::Int           = 42
)
    # 500 Hz resonance offset (moderate inhomogeneous broadening)
    ω_offset   = 500.0  # Hz
    H_drift    = 2π * ω_offset * σ_z / 2
    H_controls = [2π * σ_x / 2, 2π * σ_y / 2]
    system     = quantum_system(H_drift, H_controls)

    # Target: π rotation about x-axis — refocusing pulse
    # Rx(π) = exp(-i π σx/2) = -i σx  (up to global phase, which fidelity ignores)
    Ix   = σ_x / 2
    U_pi = exp(-im * π .* Ix)
    target = unitary_target(U_pi)

    rng           = MersenneTwister(seed)
    ctrl_mat      = 0.01 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt           = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Problem 8: Robust Hadamard
# ============================================================================

"""
    robust_hadamard_problem(; uncertainty=0.1, total_time=1e-6, n_timesteps=100, seed=42)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Hadamard gate problem with nominal system, intended for robust optimization
against ±`uncertainty` fractional errors in control amplitudes.

Returns the *nominal* system and target; robustness over the uncertainty set
should be handled by the caller using `robust_optimize` with an ensemble
generated from the returned system by perturbing control scale factors.

# Arguments
- `uncertainty` — relative amplitude uncertainty (e.g. 0.1 = ±10%)
- `total_time`, `n_timesteps`, `seed` — as for `hadamard_gate_problem`

# Returns
The same `(system, target, controls_init)` triple as `hadamard_gate_problem`,
with an additional entry in `system.metadata` recording the uncertainty level.

# Example
```julia
sys, tgt, u0 = robust_hadamard_problem(uncertainty=0.05)
# Pass to robust_optimize with amplitude uncertainty ensemble:
result = robust_optimize(sys, tgt, u0; n_samples=20,
                          uncertainty=0.05, config=RobustConfig())
```
"""
function robust_hadamard_problem(;
    uncertainty::Float64 = 0.1,
    total_time::Float64  = 1e-6,
    n_timesteps::Int     = 100,
    seed::Int            = 42
)
    0.0 <= uncertainty <= 1.0 || throw(ArgumentError(
        "uncertainty must be in [0, 1], got $uncertainty"))

    H_drift    = zeros(ComplexF64, 2, 2)
    H_controls = [2π * σ_x / 2, 2π * σ_y / 2]

    metadata = Dict{String, Any}(
        "problem"     => "robust_hadamard",
        "uncertainty" => uncertainty,
    )
    system = quantum_system(H_drift, H_controls; metadata=metadata)

    U_H    = ComplexF64[1 1; 1 -1] ./ sqrt(2)
    target = unitary_target(U_H)

    rng           = MersenneTwister(seed)
    ctrl_mat      = 0.01 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt           = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Problem 9: Random unitary (benchmarking)
# ============================================================================

"""
    random_unitary_problem(n_qubits::Int=2;
                            total_time=5e-6, n_timesteps=200, seed=42)
    -> (QuantumSystem, QuantumTarget, ControlSequence)

Generate a random quantum control problem for benchmarking purposes.

# Arguments
- `n_qubits`    — number of qubits (Hilbert space dimension = `2^n_qubits`)
- `total_time`  — total pulse duration (seconds)
- `n_timesteps` — number of control time slices
- `seed`        — RNG seed for reproducibility

# System
- H_drift: random Hermitian matrix with entries drawn from N(0,1) scaled to
  give eigenvalues ∼ `2π × 1e6` rad/s (MHz scale)
- H_controls: `2 * n_qubits` control channels (σx and σy on each qubit),
  each scaled by `2π`

# Target
A Haar-random unitary generated from a random complex matrix via QR
decomposition (Schmidt construction).

# Notes
The random seed is fixed so that results are reproducible across calls with
the same arguments.  Change `seed` to get a different random problem.

# Example
```julia
sys, tgt, u0 = random_unitary_problem(2; n_timesteps=100, seed=7)
result = grape_optimize(sys, tgt, u0; config = GRAPEConfig(max_iter=500))
```
"""
function random_unitary_problem(
    n_qubits::Int = 2;
    total_time::Float64 = 5e-6,
    n_timesteps::Int    = 200,
    seed::Int           = 42
)
    n_qubits >= 1 || throw(ArgumentError("n_qubits must be ≥ 1"))

    rng = MersenneTwister(seed)
    dim = 2^n_qubits

    # Random Hermitian drift at MHz scale
    A       = randn(rng, ComplexF64, dim, dim)
    H_rand  = (A + A') / 2
    # Normalise to ~2π × 1 MHz operator norm
    nrm     = maximum(abs.(eigvals(Hermitian(H_rand))))
    H_drift = nrm > 0 ? H_rand .* (2π * 1e6 / nrm) : H_rand

    # Control Hamiltonians: σx_k, σy_k for each qubit k
    H_controls = Matrix{ComplexF64}[]
    for q in 1:n_qubits
        push!(H_controls, 2π .* _embed_qubit_op(σ_x, n_qubits, q))
        push!(H_controls, 2π .* _embed_qubit_op(σ_y, n_qubits, q))
    end

    system = quantum_system(H_drift, H_controls;
                             metadata=Dict{String,Any}("seed" => seed,
                                                       "n_qubits" => n_qubits))

    # Haar-random target unitary via QR of random complex matrix
    Z        = randn(rng, ComplexF64, dim, dim)
    Q, _R    = qr(Z)
    # Canonical form: adjust phases so R has positive diagonal
    D        = Diagonal(sign.(diag(_R)))
    U_random = Matrix(Q) * D
    target   = unitary_target(U_random)

    # Small random initial controls
    ctrl_mat      = 0.01 .* randn(rng, Float64, system.n_controls, n_timesteps)
    _dt           = total_time / n_timesteps
    controls_init = ControlSequence(ctrl_mat, _dt, total_time, n_timesteps)

    return system, target, controls_init
end

# ============================================================================
# Exports (collected here for documentation clarity)
# ============================================================================

# The following functions are exported from PULSAR.jl via the main module:
#   spin_half_operators, spin_operators, tensor_product_operators
#   hadamard_gate_problem, not_gate_problem, cnot_gate_problem
#   state_transfer_0_to_1, state_transfer_0_to_plus
#   inept_problem, spin_echo_problem
#   robust_hadamard_problem, random_unitary_problem
