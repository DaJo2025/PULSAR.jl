# Application/QuantumComputing/Gates/TwoQubitGates.jl
# Standard two-qubit gate matrices.
#
# All gates are 4×4 Matrix{ComplexF64} in the computational basis
# {|00⟩, |01⟩, |10⟩, |11⟩} (qubit 1 is the most significant bit).

using LinearAlgebra

# ── CNOT / CX ─────────────────────────────────────────────────────────────────

"""
    CNOT_gate() -> Matrix{ComplexF64}

Controlled-NOT (CX) gate with qubit 1 as control, qubit 2 as target.

    |00⟩→|00⟩,  |01⟩→|01⟩,  |10⟩→|11⟩,  |11⟩→|10⟩
"""
function CNOT_gate()::Matrix{ComplexF64}
    ComplexF64[1 0 0 0;
               0 1 0 0;
               0 0 0 1;
               0 0 1 0]
end
const CX_gate = CNOT_gate

"""
    CZ_gate() -> Matrix{ComplexF64}

Controlled-Z (phase) gate.

    |00⟩→|00⟩,  |01⟩→|01⟩,  |10⟩→|10⟩,  |11⟩→−|11⟩
"""
function CZ_gate()::Matrix{ComplexF64}
    ComplexF64[1 0 0  0;
               0 1 0  0;
               0 0 1  0;
               0 0 0 -1]
end

"""
    CY_gate() -> Matrix{ComplexF64}

Controlled-Y gate (qubit 1 control, qubit 2 target).
"""
function CY_gate()::Matrix{ComplexF64}
    ComplexF64[1 0  0  0;
               0 1  0  0;
               0 0  0 -im;
               0 0  im 0]
end

# ── SWAP family ───────────────────────────────────────────────────────────────

"""
    SWAP_gate() -> Matrix{ComplexF64}

SWAP gate: exchanges the states of two qubits.
"""
function SWAP_gate()::Matrix{ComplexF64}
    ComplexF64[1 0 0 0;
               0 0 1 0;
               0 1 0 0;
               0 0 0 1]
end

"""
    iSWAP_gate() -> Matrix{ComplexF64}

iSWAP gate: SWAP with a phase of i on swapped components.
Native gate on superconducting platforms with fixed coupling.
"""
function iSWAP_gate()::Matrix{ComplexF64}
    ComplexF64[1  0   0  0;
               0  0  im  0;
               0  im  0  0;
               0  0   0  1]
end

"""
    SQISWAP_gate() -> Matrix{ComplexF64}

√iSWAP gate: square root of iSWAP, native on some superconducting chips.
"""
function SQISWAP_gate()::Matrix{ComplexF64}
    v = 1.0 / sqrt(2.0)
    ComplexF64[1    0       0   0;
               0    v    im*v   0;
               0  im*v     v   0;
               0    0       0   1]
end

# ── Mølmer-Sørensen (MS) gate ─────────────────────────────────────────────────

"""
    MS_gate(φ=π/4) -> Matrix{ComplexF64}

Mølmer-Sørensen entangling gate, native for trapped ions.

    MS(φ) = exp(−i φ (Xx ⊗ Xx + Xy ⊗ Xy + ...)  = exp(−iφ/2 (X⊗X + Y⊗Y))

The standard MS(π/4) maximally entangles two ions:

    |00⟩ → (|00⟩ − i|11⟩) / √2
    |01⟩ → (|01⟩ − i|10⟩) / √2
"""
function MS_gate(φ::Real = π/4)::Matrix{ComplexF64}
    c, s = cos(Float64(φ)), sin(Float64(φ))
    ComplexF64[ c      0      0   -im*s;
                0      c   -im*s    0;
                0   -im*s    c      0;
               -im*s   0      0      c]
end

# ── Controlled-rotation gates ─────────────────────────────────────────────────

"""
    CRx(θ) -> Matrix{ComplexF64}

Controlled Rx rotation.
"""
function CRx(θ::Real)::Matrix{ComplexF64}
    c, s = cos(θ/2), sin(θ/2)
    ComplexF64[1 0      0        0;
               0 1      0        0;
               0 0      c      -im*s;
               0 0    -im*s      c]
end

"""
    CRy(θ) -> Matrix{ComplexF64}

Controlled Ry rotation.
"""
function CRy(θ::Real)::Matrix{ComplexF64}
    c, s = cos(θ/2), sin(θ/2)
    ComplexF64[1 0  0  0;
               0 1  0  0;
               0 0  c -s;
               0 0  s  c]
end

"""
    CRz(θ) -> Matrix{ComplexF64}

Controlled Rz rotation.
"""
function CRz(θ::Real)::Matrix{ComplexF64}
    ComplexF64[1 0           0               0;
               0 1           0               0;
               0 0  exp(-im*θ/2)            0;
               0 0           0   exp(im*θ/2)]
end

# ── ZZ interaction gate ───────────────────────────────────────────────────────

"""
    ZZθ_gate(θ) -> Matrix{ComplexF64}

ZZ interaction gate:  exp(−iθ Z⊗Z / 2).
Common in superconducting, spin-qubit, and trapped-ion platforms.
"""
function ZZθ_gate(θ::Real)::Matrix{ComplexF64}
    e_p = exp(-im * θ / 2)
    e_m = exp( im * θ / 2)
    ComplexF64[e_p  0   0   0;
               0   e_m  0   0;
               0    0  e_m  0;
               0    0   0  e_p]
end

# ── Convenience ───────────────────────────────────────────────────────────────

"""
    two_qubit_gate_set() -> NamedTuple

Named tuple of standard two-qubit gates:
`(CNOT, CZ, CY, SWAP, iSWAP, SQISWAP, MS)`.
"""
function two_qubit_gate_set()
    (CNOT   = CNOT_gate(),
     CZ     = CZ_gate(),
     CY     = CY_gate(),
     SWAP   = SWAP_gate(),
     iSWAP  = iSWAP_gate(),
     SQISWAP= SQISWAP_gate(),
     MS     = MS_gate())
end
