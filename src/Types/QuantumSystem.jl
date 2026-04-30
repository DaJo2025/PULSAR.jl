# Types/QuantumSystem.jl
# Quantum system type definitions and constructors.
# Extracted from Core/QuantumSystem.jl.

using LinearAlgebra
using Random

# ============================================================================
# Abstract base type
# ============================================================================

"""
    AbstractQuantumSystem

Abstract supertype for all quantum systems in Pulsar. Every concrete subtype
must expose the fields:

  - `H_drift  :: Matrix{ComplexF64}` — time-independent drift Hamiltonian (rad/s)
  - `H_controls :: Vector{Matrix{ComplexF64}}` — control Hamiltonians
  - `dim       :: Int`               — Hilbert space dimension
  - `n_controls :: Int`              — number of independent control fields
"""
abstract type AbstractQuantumSystem end

"""
    AbstractOptimizationContext

Abstract supertype for all Pulsar optimal-control problem specifications.
Concrete subtypes store the system, target, algorithm settings, and
(optionally) the initial waveform in a single object so that the unified
dispatch `optimcon(ctx)` can route to the correct solver without the
caller having to know which backend is in use.

Concrete subtypes:
- `MRControl` / `LindbladMRControl` (via `AbstractMRControl`)
- `QCControl` — quantum-computing context
"""
abstract type AbstractOptimizationContext end

# ============================================================================
# Concrete system types
# ============================================================================

"""
    QuantumSystem <: AbstractQuantumSystem

General-purpose quantum system defined by explicit Hamiltonian matrices.

# Fields
- `H_drift    :: Matrix{ComplexF64}` — Drift Hamiltonian (rad/s), must be Hermitian
- `H_controls :: Vector{Matrix{ComplexF64}}` — Control Hamiltonians (rad/s each),
  each must be Hermitian
- `dim        :: Int`  — Hilbert space dimension (size of H_drift)
- `n_controls :: Int`  — number of control fields (length of H_controls)
- `metadata   :: Dict{String,Any}` — arbitrary user-defined metadata

# Physics
The system evolves under H(t) = H_drift + Σ_j u_j(t) H_controls[j].
"""
struct QuantumSystem <: AbstractQuantumSystem
    H_drift    :: Matrix{ComplexF64}
    H_controls :: Vector{Matrix{ComplexF64}}
    dim        :: Int
    n_controls :: Int
    metadata   :: Dict{String,Any}
end

"""
    SpinSystem <: AbstractQuantumSystem

NMR / EPR multi-spin system constructed from first principles.

# Fields
- `spins            :: Vector{Float64}` — spin quantum numbers (e.g. [0.5, 0.5] for two spin-1/2)
- `couplings        :: Matrix{Float64}` — symmetric J-coupling matrix in Hz; element (i,j) is J_ij
- `chemical_shifts  :: Vector{Float64}` — isotropic chemical shifts in Hz
- `H_drift          :: Matrix{ComplexF64}` — total drift Hamiltonian (Zeeman + J), rad/s
- `H_controls       :: Vector{Matrix{ComplexF64}}` — RF control Hamiltonians; 2*n_spins elements,
  alternating x and y pulses: [Ix1, Iy1, Ix2, Iy2, ...]
- `dim              :: Int` — total Hilbert space dimension = Π_i (2*s_i + 1)
- `n_controls       :: Int` — 2 * n_spins (x and y channel per spin)

# Physics
The drift Hamiltonian (in rad/s, rotating frame at carrier frequency) is:

    H_drift = 2π * [ Σ_i δ_i * Iz_i  +  Σ_{i<j} J_ij * (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j) ]

where operators are embedded in the full tensor-product Hilbert space.
Control Hamiltonians for spin i are 2π * Ix_i and 2π * Iy_i (so that amplitudes
are in Hz = 1/s, matching the nutation frequency convention).
"""
struct SpinSystem <: AbstractQuantumSystem
    spins           :: Vector{Float64}
    couplings       :: Matrix{Float64}
    chemical_shifts :: Vector{Float64}
    H_drift         :: Matrix{ComplexF64}
    H_controls      :: Vector{Matrix{ComplexF64}}
    dim             :: Int
    n_controls      :: Int
end

"""
    QubitSystem <: AbstractQuantumSystem

System of n_qubits two-level systems (qubits) with specified connectivity.

# Fields
- `n_qubits     :: Int`  — number of qubits
- `H_drift      :: Matrix{ComplexF64}` — drift Hamiltonian (rad/s)
- `H_controls   :: Vector{Matrix{ComplexF64}}` — control Hamiltonians (rad/s)
- `dim          :: Int`  — 2^n_qubits
- `n_controls   :: Int`  — number of control channels
- `connectivity :: Matrix{Bool}` — symmetric adjacency matrix; entry (i,j) = true
  means qubits i and j are coupled in H_drift

# Physics
Qubits are spin-1/2 particles.  A typical H_drift takes the form:

    H_drift = Σ_i ω_i / 2 * σz_i  +  Σ_{(i,j) ∈ edges} g_ij * (σx_i σx_j + σy_i σy_j) / 4

but the exact form is user-supplied via `qubit_system`.
"""
struct QubitSystem <: AbstractQuantumSystem
    n_qubits     :: Int
    H_drift      :: Matrix{ComplexF64}
    H_controls   :: Vector{Matrix{ComplexF64}}
    dim          :: Int
    n_controls   :: Int
    connectivity :: Matrix{Bool}
end

# ============================================================================
# Spin operator construction helpers
# ============================================================================

"""
    spin_Sz(s::Float64) -> Matrix{ComplexF64}

Return the z-component angular momentum operator Sz for a spin-s particle.

# Arguments
- `s` — spin quantum number (0.5, 1.0, 1.5, ...)

# Returns
Diagonal matrix of size `(2s+1) × (2s+1)` with entries

    ⟨m | Sz | m⟩ = m,    m = s, s-1, …, -s

in the standard Zeeman basis ordered from m = +s (top) to m = -s (bottom).

# Example
```julia
julia> spin_Sz(0.5)
2×2 Matrix{ComplexF64}:
 0.5+0.0im   0.0+0.0im
 0.0+0.0im  -0.5+0.0im
```
"""
function spin_Sz(s::Real)::Matrix{ComplexF64}
    d = Int(2s + 1)
    ms = [s - i for i in 0:(d-1)]   # m values from +s to -s
    return ComplexF64.(diagm(ms))
end

"""
    spin_Splus(s::Float64) -> Matrix{ComplexF64}

Raising operator S+ for a spin-s particle.

# Arguments
- `s` — spin quantum number

# Returns
Matrix of size `(2s+1) × (2s+1)` with off-diagonal entries

    ⟨m+1 | S+ | m⟩ = √(s(s+1) - m(m+1))

# Physics
S+ = Sx + i*Sy.  Together with S- = (S+)†, the Cartesian components are
recovered as Sx = (S+ + S-)/2,  Sy = (S+ - S-)/(2i).
"""
function spin_Splus(s::Real)::Matrix{ComplexF64}
    d = Int(2s + 1)
    ms = [s - i for i in 0:(d-1)]   # m values: row index corresponds to ket m
    S = zeros(ComplexF64, d, d)
    for col in 1:d
        m = ms[col]          # ket m
        row = col - 1        # bra m+1  (higher index = lower m value, so row = col-1)
        if row >= 1
            S[row, col] = sqrt(s*(s+1) - m*(m+1))
        end
    end
    return S
end

"""
    spin_Sminus(s::Float64) -> Matrix{ComplexF64}

Lowering operator S- = (S+)† for a spin-s particle.
"""
function spin_Sminus(s::Real)::Matrix{ComplexF64}
    return spin_Splus(s)'
end

"""
    spin_Sx(s::Float64) -> Matrix{ComplexF64}

x-component angular momentum operator Sx = (S+ + S-) / 2 for a spin-s particle.
"""
function spin_Sx(s::Real)::Matrix{ComplexF64}
    Sp = spin_Splus(s)
    Sm = spin_Sminus(s)
    return (Sp + Sm) / 2
end

"""
    spin_Sy(s::Float64) -> Matrix{ComplexF64}

y-component angular momentum operator Sy = (S+ - S-) / (2i) for a spin-s particle.
"""
function spin_Sy(s::Real)::Matrix{ComplexF64}
    Sp = spin_Splus(s)
    Sm = spin_Sminus(s)
    return (Sp - Sm) / (2im)
end

"""
    pauli_x() -> Matrix{ComplexF64}

Pauli σx matrix for spin-1/2.

    σx = [0  1; 1  0]
"""
pauli_x() = ComplexF64[0 1; 1 0]

"""
    pauli_y() -> Matrix{ComplexF64}

Pauli σy matrix for spin-1/2.

    σy = [0  -i; i  0]
"""
pauli_y() = ComplexF64[0 -im; im 0]

"""
    pauli_z() -> Matrix{ComplexF64}

Pauli σz matrix for spin-1/2.

    σz = [1  0; 0  -1]
"""
pauli_z() = ComplexF64[1 0; 0 -1]

"""
    embed_operator(op::Matrix{ComplexF64}, spins::Vector{Float64}, spin_index::Int)
    -> Matrix{ComplexF64}

Embed a single-spin operator `op` acting on spin number `spin_index` into the
full tensor-product Hilbert space of `n_spins` spins.

# Arguments
- `op`         — operator in the local Hilbert space of spin `spin_index`
- `spins`      — vector of spin quantum numbers for all spins
- `spin_index` — 1-based index of the spin that `op` acts on

# Returns
Full-space operator of dimension `Π_i (2*s_i+1)`.

# Construction
The embedding is: I_{s_1} ⊗ … ⊗ op_{s_k} ⊗ … ⊗ I_{s_n}
where I_{s_i} is the identity on the local space of spin i.
"""
function embed_operator(op::Matrix{ComplexF64}, spins::Vector{Float64},
                        spin_index::Int)::Matrix{ComplexF64}
    n = length(spins)
    dims = [Int(2*s + 1) for s in spins]
    result = ones(ComplexF64, 1, 1)
    for i in 1:n
        if i == spin_index
            result = kron(result, op)
        else
            result = kron(result, Matrix{ComplexF64}(I, dims[i], dims[i]))
        end
    end
    return result
end

# ============================================================================
# Constructor functions
# ============================================================================

"""
    quantum_system(H_drift::Matrix, H_controls::Vector{<:Matrix};
                   metadata::Dict{String,Any}=Dict{String,Any}()) -> QuantumSystem

Construct a general `QuantumSystem` from explicit Hamiltonian matrices.

# Arguments
- `H_drift`    — Hermitian drift Hamiltonian, size `dim × dim`
- `H_controls` — vector of Hermitian control Hamiltonians, each `dim × dim`
- `metadata`   — optional dictionary of user metadata

# Returns
A validated `QuantumSystem`.

# Throws
- `ArgumentError` if `H_drift` is not square, not Hermitian, or any
  `H_controls[j]` is not the same size as `H_drift` or not Hermitian.

# Example
```julia
using LinearAlgebra
H0 = diagm([0.0, 1.0, 2.0] .+ 0im)
Hx = [0 1 0; 1 0 1; 0 1 0] ./ sqrt(2) .+ 0im
sys = quantum_system(H0, [Hx])
```
"""
function quantum_system(H_drift::Matrix, H_controls::AbstractVector;
                        metadata::Dict{String,Any}=Dict{String,Any}())::QuantumSystem
    for (j, Hc) in enumerate(H_controls)
        Hc isa AbstractMatrix || throw(ArgumentError(
            "H_controls[$j] must be a Matrix; got $(typeof(Hc))"))
    end
    Hd = ComplexF64.(H_drift)
    Hcs = Matrix{ComplexF64}[ComplexF64.(H) for H in H_controls]

    dim = size(Hd, 1)
    n_controls = length(Hcs)

    # Validate
    _check_hermitian_matrix(Hd, "H_drift")
    for (j, Hc) in enumerate(Hcs)
        if size(Hc) != (dim, dim)
            throw(ArgumentError(
                "H_controls[$j] has size $(size(Hc)) but expected ($dim, $dim)"))
        end
        _check_hermitian_matrix(Hc, "H_controls[$j]")
    end

    return QuantumSystem(Hd, Hcs, dim, n_controls, metadata)
end

"""
    spin_system(spins::Vector{Float64}, couplings::Matrix{Float64},
                chemical_shifts::Vector{Float64}) -> SpinSystem

Construct a multi-spin NMR/EPR system from physical parameters.

# Arguments
- `spins`           — vector of spin quantum numbers, e.g. `[0.5, 0.5]`
- `couplings`       — symmetric J-coupling matrix in Hz; `couplings[i,j]` is J_{ij}
  (diagonal elements are ignored; must satisfy `couplings[i,j] == couplings[j,i]`)
- `chemical_shifts` — isotropic chemical shifts (offset frequencies from carrier) in Hz

# Returns
A `SpinSystem` with H_drift (rad/s) and H_controls (rad/s).

# Physics
The drift Hamiltonian in the rotating frame is (ħ = 1, frequencies in rad/s):

    H_drift = 2π Σ_i δ_i Iz_i  +  2π Σ_{i<j} J_ij (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j)

Control Hamiltonians (one x and one y channel per spin):

    H_ctrl[2i-1] = 2π Ix_i   (x RF pulse on spin i)
    H_ctrl[2i]   = 2π Iy_i   (y RF pulse on spin i)

With amplitudes in Hz, the product u_j * H_ctrl[j] is in rad/s.

# Throws
- `ArgumentError` if `couplings` is not symmetric or has wrong size.

# Example
```julia
# Two coupled spin-1/2 nuclei, 100 Hz shift difference, 10 Hz J-coupling
sys = spin_system([0.5, 0.5], [0.0 10.0; 10.0 0.0], [500.0, 600.0])
```
"""
function spin_system(spins::Vector{Float64}, couplings::Matrix{Float64},
                     chemical_shifts::Vector{Float64})::SpinSystem
    n = length(spins)

    # Input validation
    if length(chemical_shifts) != n
        throw(ArgumentError(
            "Length of chemical_shifts ($(length(chemical_shifts))) must equal " *
            "number of spins ($n)"))
    end
    if size(couplings) != (n, n)
        throw(ArgumentError(
            "couplings matrix must be $n × $n, got $(size(couplings))"))
    end
    if !issymmetric(couplings)
        throw(ArgumentError("J-coupling matrix must be symmetric"))
    end
    for s in spins
        if s < 0.5 || !isinteger(2s)
            throw(ArgumentError(
                "Spin quantum number $s is invalid; must be a half-integer ≥ 1/2"))
        end
    end

    # Hilbert space dimension
    dims = [Int(2*s + 1) for s in spins]
    dim = prod(dims)

    # Build embedded single-spin operators for each spin
    Ix = Vector{Matrix{ComplexF64}}(undef, n)
    Iy = Vector{Matrix{ComplexF64}}(undef, n)
    Iz = Vector{Matrix{ComplexF64}}(undef, n)
    for i in 1:n
        Ix[i] = embed_operator(spin_Sx(spins[i]), spins, i)
        Iy[i] = embed_operator(spin_Sy(spins[i]), spins, i)
        Iz[i] = embed_operator(spin_Sz(spins[i]), spins, i)
    end

    # Drift Hamiltonian: Zeeman (chemical shift) + isotropic J-coupling
    H_drift = zeros(ComplexF64, dim, dim)

    # Zeeman part: 2π δ_i Iz_i
    for i in 1:n
        H_drift .+= (2π * chemical_shifts[i]) .* Iz[i]
    end

    # J-coupling part: 2π J_ij (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j)
    for i in 1:n
        for j in (i+1):n
            if couplings[i,j] != 0.0
                Jij = couplings[i,j]
                H_drift .+= (2π * Jij) .* (Ix[i]*Ix[j] .+ Iy[i]*Iy[j] .+ Iz[i]*Iz[j])
            end
        end
    end

    # Control Hamiltonians: 2π Ix_i and 2π Iy_i for each spin
    H_controls = Vector{Matrix{ComplexF64}}(undef, 2*n)
    for i in 1:n
        H_controls[2i-1] = (2π) .* Ix[i]   # x-channel
        H_controls[2i]   = (2π) .* Iy[i]   # y-channel
    end

    n_controls = 2 * n

    return SpinSystem(spins, couplings, chemical_shifts,
                      H_drift, H_controls, dim, n_controls)
end

"""
    spin_system(spin::Real, omega_drift::Real,
                H_controls::Vector{<:AbstractMatrix}) -> SpinSystem

Single-spin convenience constructor. Builds `H_drift = omega_drift * Iz`
on the (2s+1)-dimensional Hilbert space and uses the supplied control
Hamiltonians directly. `omega_drift` is in angular-frequency units.
"""
function spin_system(spin::Real, omega_drift::Real,
                     H_controls::Vector{<:AbstractMatrix})::SpinSystem
    s_f = Float64(spin)
    if s_f < 0.5 || !isinteger(2 * s_f)
        throw(ArgumentError(
            "spin must be a half-integer ≥ 1/2, got $spin"))
    end
    dim = Int(2 * s_f + 1)
    Iz_op = ComplexF64.(spin_Sz(s_f))
    H_drift = ComplexF64(omega_drift) .* Iz_op
    Hc = Vector{Matrix{ComplexF64}}(undef, length(H_controls))
    for (i, H) in enumerate(H_controls)
        size(H) == (dim, dim) ||
            throw(ArgumentError("H_controls[$i] must be $dim × $dim, got $(size(H))"))
        Hc[i] = ComplexF64.(H)
    end
    couplings        = zeros(Float64, 1, 1)
    chemical_shifts  = [Float64(omega_drift) / (2π)]
    return SpinSystem([s_f], couplings, chemical_shifts,
                      H_drift, Hc, dim, length(Hc))
end

"""
    qubit_system(n_qubits::Int, H_drift::Matrix, H_controls::Vector{<:Matrix};
                 connectivity::Union{Matrix{Bool},Nothing}=nothing) -> QubitSystem

Construct a `QubitSystem` for `n_qubits` two-level systems.

# Arguments
- `n_qubits`     — number of qubits
- `H_drift`      — Hermitian drift Hamiltonian of size `2^n_qubits × 2^n_qubits` (rad/s)
- `H_controls`   — vector of Hermitian control Hamiltonians (rad/s)
- `connectivity` — optional `n_qubits × n_qubits` symmetric Boolean adjacency matrix.
  If `nothing`, defaults to a fully-connected graph for `n_qubits > 1`.

# Returns
A validated `QubitSystem`.

# Throws
- `ArgumentError` if dimensions are inconsistent or Hamiltonians are not Hermitian.

# Example
```julia
# Single-qubit system driven on σx and σy
H0 = 0.5 * pauli_z()       # qubit at half the angular frequency
Hx = 0.5 * pauli_x()
Hy = 0.5 * pauli_y()
sys = qubit_system(1, H0, [Hx, Hy])
```
"""
function qubit_system(n_qubits::Int, H_drift::Matrix, H_controls::AbstractVector{<:Matrix};
                      connectivity::Union{Matrix{Bool},Nothing}=nothing)::QubitSystem
    if n_qubits < 1
        throw(ArgumentError("n_qubits must be ≥ 1, got $n_qubits"))
    end
    dim = 2^n_qubits
    Hd = ComplexF64.(H_drift)
    Hcs = [ComplexF64.(H) for H in H_controls]
    n_controls = length(Hcs)

    if size(Hd) != (dim, dim)
        throw(ArgumentError(
            "H_drift size $(size(Hd)) does not match expected ($dim, $dim) " *
            "for $n_qubits qubits"))
    end
    _check_hermitian_matrix(Hd, "H_drift")
    for (j, Hc) in enumerate(Hcs)
        if size(Hc) != (dim, dim)
            throw(ArgumentError(
                "H_controls[$j] size $(size(Hc)) does not match ($dim, $dim)"))
        end
        _check_hermitian_matrix(Hc, "H_controls[$j]")
    end

    # Default connectivity: fully connected
    conn = if connectivity === nothing
        if n_qubits == 1
            fill(false, 1, 1)
        else
            conn_mat = fill(true, n_qubits, n_qubits)
            for i in 1:n_qubits; conn_mat[i, i] = false; end
            conn_mat
        end
    else
        if size(connectivity) != (n_qubits, n_qubits)
            throw(ArgumentError(
                "connectivity matrix size $(size(connectivity)) must be " *
                "($n_qubits, $n_qubits)"))
        end
        connectivity
    end

    return QubitSystem(n_qubits, Hd, Hcs, dim, n_controls, conn)
end

# ============================================================================
# Internal helpers
# ============================================================================

"""
    _check_hermitian_matrix(H::Matrix{ComplexF64}, name::String; tol::Float64=1e-10)

Internal helper: throw `ArgumentError` if H is not square or not Hermitian within `tol`.
"""
function _check_hermitian_matrix(H::Matrix{ComplexF64}, name::String;
                                  tol::Float64=1e-10)
    m, n = size(H)
    if m != n
        throw(ArgumentError("$name must be square, got $m × $n"))
    end
    dev = maximum(abs.(H - H'))
    if dev > tol
        throw(ArgumentError(
            "$name is not Hermitian: max|H - H†| = $dev > tol = $tol"))
    end
end

# ============================================================================
# Utility: pretty printing
# ============================================================================

function Base.show(io::IO, sys::QuantumSystem)
    print(io, "QuantumSystem(dim=$(sys.dim), n_controls=$(sys.n_controls))")
end

function Base.show(io::IO, sys::SpinSystem)
    print(io, "SpinSystem(spins=$(sys.spins), dim=$(sys.dim), " *
              "n_controls=$(sys.n_controls))")
end

function Base.show(io::IO, sys::QubitSystem)
    print(io, "QubitSystem(n_qubits=$(sys.n_qubits), dim=$(sys.dim), " *
              "n_controls=$(sys.n_controls))")
end
