# Application/QuantumComputing/Gates/SingleQubitGates.jl
# Standard single-qubit gate matrices and rotation generators.
#
# All gates are returned as Matrix{ComplexF64} in the computational basis
# {|0⟩, |1⟩} (standard Dirac/Bloch-sphere convention).
#
# Usage:
#   U = X_gate()                  # Pauli X
#   U = Rx(π/2)                   # 90° rotation about x
#   tgt = unitary_target(H_gate())  # Hadamard target for GRAPE

using LinearAlgebra

# ── Pauli matrices ────────────────────────────────────────────────────────────

"""    X_gate() -> Matrix{ComplexF64}   Pauli X (NOT gate). """
X_gate()::Matrix{ComplexF64} = ComplexF64[0 1; 1 0]

"""    Y_gate() -> Matrix{ComplexF64}   Pauli Y. """
Y_gate()::Matrix{ComplexF64} = ComplexF64[0 -im; im 0]

"""    Z_gate() -> Matrix{ComplexF64}   Pauli Z. """
Z_gate()::Matrix{ComplexF64} = ComplexF64[1 0; 0 -1]

"""    I_gate(n=2) -> Matrix{ComplexF64}   n×n identity. """
I_gate(n::Int = 2)::Matrix{ComplexF64} = Matrix{ComplexF64}(I, n, n)

# ── Clifford single-qubit gates ───────────────────────────────────────────────

"""    H_gate() -> Matrix{ComplexF64}   Hadamard gate. """
H_gate()::Matrix{ComplexF64} = ComplexF64[1 1; 1 -1] ./ sqrt(2.0)

"""    S_gate() -> Matrix{ComplexF64}   Phase gate (√Z). """
S_gate()::Matrix{ComplexF64} = ComplexF64[1 0; 0 im]

"""    Sdg_gate() -> Matrix{ComplexF64}   S† (inverse phase gate). """
Sdg_gate()::Matrix{ComplexF64} = ComplexF64[1 0; 0 -im]

"""    T_gate() -> Matrix{ComplexF64}   π/8 gate (⁴√Z). """
T_gate()::Matrix{ComplexF64} = ComplexF64[1 0; 0 exp(im*π/4)]

"""    Tdg_gate() -> Matrix{ComplexF64}   T† (inverse T gate). """
Tdg_gate()::Matrix{ComplexF64} = ComplexF64[1 0; 0 exp(-im*π/4)]

"""    SX_gate() -> Matrix{ComplexF64}   √X gate (half NOT). """
SX_gate()::Matrix{ComplexF64} = ComplexF64[1+im 1-im; 1-im 1+im] ./ 2.0

# ── Rotation gates ────────────────────────────────────────────────────────────

"""
    Rx(θ) -> Matrix{ComplexF64}

Rotation by angle θ (radians) about the x-axis:

    Rx(θ) = cos(θ/2) I − i sin(θ/2) X
"""
function Rx(θ::Real)::Matrix{ComplexF64}
    c, s = cos(θ/2), sin(θ/2)
    ComplexF64[c -im*s; -im*s c]
end

"""
    Ry(θ) -> Matrix{ComplexF64}

Rotation by angle θ about the y-axis:

    Ry(θ) = cos(θ/2) I − i sin(θ/2) Y
"""
function Ry(θ::Real)::Matrix{ComplexF64}
    c, s = cos(θ/2), sin(θ/2)
    ComplexF64[c -s; s c]
end

"""
    Rz(θ) -> Matrix{ComplexF64}

Rotation by angle θ about the z-axis:

    Rz(θ) = exp(−iθ/2) |0⟩⟨0| + exp(iθ/2) |1⟩⟨1|
"""
function Rz(θ::Real)::Matrix{ComplexF64}
    ComplexF64[exp(-im*θ/2) 0; 0 exp(im*θ/2)]
end

"""
    Rn(θ, nx, ny, nz) -> Matrix{ComplexF64}

Rotation by angle θ about the unit axis n̂ = (nx, ny, nz):

    R_n(θ) = cos(θ/2) I − i sin(θ/2) (nx X + ny Y + nz Z)
"""
function Rn(θ::Real, nx::Real, ny::Real, nz::Real)::Matrix{ComplexF64}
    c = cos(θ/2)
    s = sin(θ/2)
    n = sqrt(nx^2 + ny^2 + nz^2)
    n < eps(Float64) && return I_gate()
    nx, ny, nz = nx/n, ny/n, nz/n
    ComplexF64[c - im*s*nz        -s*(ny + im*nx);
               s*(ny - im*nx)      c + im*s*nz]
end

"""
    U3(θ, φ, λ) -> Matrix{ComplexF64}

IBM/Qiskit U3 single-qubit gate:

    U3(θ,φ,λ) = [cos(θ/2)              −e^{iλ} sin(θ/2);
                  e^{iφ} sin(θ/2)        e^{i(φ+λ)} cos(θ/2)]
"""
function U3(θ::Real, φ::Real, λ::Real)::Matrix{ComplexF64}
    c, s = cos(θ/2), sin(θ/2)
    ComplexF64[c               -exp(im*λ)*s;
               exp(im*φ)*s      exp(im*(φ+λ))*c]
end

# ── Convenience: all Pauli/Clifford gates as named tuple ─────────────────────

"""
    single_qubit_gate_set() -> NamedTuple

Return a named tuple of all standard single-qubit gates:
`(I, X, Y, Z, H, S, Sdg, T, Tdg, SX)`.

Useful for building native gate sets and process tomography.

# Example
```julia
gs = single_qubit_gate_set()
println(gs.H)    # Hadamard
```
"""
function single_qubit_gate_set()
    (I  = I_gate(),
     X  = X_gate(),
     Y  = Y_gate(),
     Z  = Z_gate(),
     H  = H_gate(),
     S  = S_gate(),
     Sdg= Sdg_gate(),
     T  = T_gate(),
     Tdg= Tdg_gate(),
     SX = SX_gate())
end
