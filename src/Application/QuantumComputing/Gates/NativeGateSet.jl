# Application/QuantumComputing/Gates/NativeGateSet.jl
# Per-platform native gate set definitions and gate decomposition utilities.
#
# A NativeGateSet stores the set of elementary gates that a hardware platform
# can implement, along with typical gate times, fidelities, and decomposition
# functions to express arbitrary unitaries in terms of native gates.

using LinearAlgebra

# ============================================================================
# NativeGateSet struct
# ============================================================================

"""
    NativeGateSet

Describes the native gate set of a quantum hardware platform.

# Fields
- `platform`        — Symbol: `:superconducting`, `:trapped_ion`, `:neutral_atom`,
  `:spin_qubit`, or `:nv_center`
- `single_qubit`    — Dict{String, Matrix{ComplexF64}}: named single-qubit gates
- `two_qubit`       — Dict{String, Matrix{ComplexF64}}: named two-qubit gates
- `gate_times_ns`   — Dict{String, Float64}: approximate gate durations (ns)
- `gate_fidelities` — Dict{String, Float64}: typical achieved gate fidelities
- `metadata`        — Dict{String,Any}: arbitrary platform metadata

# Example
```julia
gs = native_gate_set(:superconducting)
U  = gs.two_qubit["CZ"]            # CZ gate matrix
t  = gs.gate_times_ns["CZ"]        # ≈ 40 ns
```
"""
struct NativeGateSet
    platform        :: Symbol
    single_qubit    :: Dict{String, Matrix{ComplexF64}}
    two_qubit       :: Dict{String, Matrix{ComplexF64}}
    gate_times_ns   :: Dict{String, Float64}
    gate_fidelities :: Dict{String, Float64}
    metadata        :: Dict{String, Any}
end

# ============================================================================
# Platform-specific constructors
# ============================================================================

"""
    native_gate_set(platform::Symbol; metadata=Dict()) -> NativeGateSet

Return the typical native gate set for the specified hardware platform.

# Supported platforms
| Symbol              | Native 1Q            | Native 2Q           |
|:--------------------|:---------------------|:--------------------|
| `:superconducting`  | Rx, Ry, Rz, X, SX   | CZ, iSWAP, √iSWAP  |
| `:trapped_ion`      | Rx, Ry, Rz           | MS(π/4), ZZ(π/2)   |
| `:neutral_atom`     | Rx, Ry, Rz           | CZ (Rydberg)        |
| `:spin_qubit`       | Rx, Ry, Rz           | CNOT (via exchange) |
| `:nv_center`        | Rx, Ry, Rz           | (1Q only by default)|

# Example
```julia
gs = native_gate_set(:trapped_ion)
println(keys(gs.two_qubit))   # ["MS", "ZZ"]
```
"""
function native_gate_set(platform :: Symbol;
                          metadata :: Dict{String,Any} = Dict{String,Any}())::NativeGateSet
    sq1 = Dict{String, Matrix{ComplexF64}}(
        "X"  => X_gate(),
        "Y"  => Y_gate(),
        "Z"  => Z_gate(),
        "H"  => H_gate(),
        "S"  => S_gate(),
        "Sdg"=> Sdg_gate(),
        "T"  => T_gate(),
        "SX" => SX_gate(),
        "Rx(π/2)"  => Rx(π/2),
        "Ry(π/2)"  => Ry(π/2),
        "Rz(π/2)"  => Rz(π/2),
        "Rx(π)"    => Rx(π),
        "Ry(π)"    => Ry(π),
        "Rz(π)"    => Rz(π),
    )

    if platform == :superconducting
        tq = Dict{String, Matrix{ComplexF64}}(
            "CZ"      => CZ_gate(),
            "iSWAP"   => iSWAP_gate(),
            "SQISWAP" => SQISWAP_gate(),
        )
        times = Dict{String, Float64}(
            "X"       => 20.0,   "SX"      => 20.0,
            "Rz(π/2)" => 0.0,    "Rz(π)"   => 0.0,
            "CZ"      => 40.0,   "iSWAP"   => 60.0, "SQISWAP" => 60.0,
        )
        fids = Dict{String, Float64}(
            "X"  => 0.9995, "SX" => 0.9995,
            "CZ" => 0.995,  "iSWAP" => 0.993, "SQISWAP" => 0.993,
        )

    elseif platform == :trapped_ion
        tq = Dict{String, Matrix{ComplexF64}}(
            "MS"  => MS_gate(π/4),
            "ZZ"  => ZZθ_gate(π/2),
            "CNOT"=> CNOT_gate(),
        )
        times = Dict{String, Float64}(
            "Rx(π)"   => 10.0, "Rx(π/2)" => 5.0,
            "Ry(π)"   => 10.0, "Ry(π/2)" => 5.0,
            "MS"      => 100.0, "ZZ" => 100.0, "CNOT" => 120.0,
        )
        fids = Dict{String, Float64}(
            "Rx(π)"  => 0.9999, "Ry(π)"  => 0.9999,
            "MS"     => 0.999,  "CNOT"   => 0.999,
        )

    elseif platform == :neutral_atom
        tq = Dict{String, Matrix{ComplexF64}}(
            "CZ"   => CZ_gate(),
            "CNOT" => CNOT_gate(),
        )
        times = Dict{String, Float64}(
            "Rx(π)"  => 200.0, "Ry(π)"  => 200.0,
            "CZ"     => 500.0, "CNOT"   => 600.0,
        )
        fids = Dict{String, Float64}(
            "Rx(π)" => 0.999, "Ry(π)" => 0.999,
            "CZ"    => 0.997, "CNOT"  => 0.996,
        )

    elseif platform == :spin_qubit
        tq = Dict{String, Matrix{ComplexF64}}(
            "CNOT"  => CNOT_gate(),
            "CZ"    => CZ_gate(),
            "SWAP"  => SWAP_gate(),
        )
        times = Dict{String, Float64}(
            "Rx(π)" => 50.0, "Ry(π)" => 50.0,
            "CNOT"  => 200.0, "CZ"   => 200.0,
        )
        fids = Dict{String, Float64}(
            "Rx(π)" => 0.999, "Ry(π)" => 0.999,
            "CNOT"  => 0.990, "CZ"    => 0.990,
        )

    elseif platform == :nv_center
        tq = Dict{String, Matrix{ComplexF64}}()
        times = Dict{String, Float64}(
            "Rx(π)" => 50.0, "Ry(π)" => 50.0,
        )
        fids = Dict{String, Float64}(
            "Rx(π)" => 0.999, "Ry(π)" => 0.999,
        )

    else
        throw(ArgumentError(
            "Unknown platform ':$platform'. Valid: " *
            join(string.([":superconducting", ":trapped_ion",
                          ":neutral_atom", ":spin_qubit", ":nv_center"]), ", ")))
    end

    return NativeGateSet(platform, sq1, tq, times, fids, metadata)
end

# ============================================================================
# Decomposition utilities
# ============================================================================

"""
    zyz_decompose(U) -> (α, β, γ, δ)

Decompose an arbitrary SU(2) single-qubit unitary into ZYZ Euler angles:

    U = exp(iδ) Rz(α) Ry(β) Rz(γ)

Returns `(α, β, γ, δ)` as Float64 values (radians).

Reference: Nielsen & Chuang, Appendix A.
"""
function zyz_decompose(U::Matrix{ComplexF64})
    @assert size(U) == (2, 2) "zyz_decompose: U must be 2×2"
    # Extract global phase
    δ = angle(det(U)) / 2
    U_su2 = exp(-im * δ) .* U   # SU(2) part

    # ZYZ: U = [[cos(β/2)e^{-i(α+γ)/2}, -sin(β/2)e^{-i(α-γ)/2}],
    #           [sin(β/2)e^{i(α-γ)/2},   cos(β/2)e^{i(α+γ)/2}]]
    β = 2 * atan(abs(U_su2[2, 1]), abs(U_su2[1, 1]))
    αpγ = -2 * angle(U_su2[1, 1])    # −(α+γ) from top-left
    αmγ =  2 * angle(U_su2[2, 1])    # +(α−γ) from bottom-left

    α = (αpγ + αmγ) / 2
    γ = (αpγ - αmγ) / 2
    return (α, β, γ, δ)
end

"""
    zyz_sequence(U) -> Vector{NamedTuple}

Express an arbitrary SU(2) gate as a ZYZ rotation sequence suitable for
native `:Rz, :Ry, :Rz` compilation.

Returns a vector of `(gate, angle)` named tuples:
  `[(gate=:Rz, angle=α), (gate=:Ry, angle=β), (gate=:Rz, angle=γ)]`

# Example
```julia
seq = zyz_sequence(H_gate())
# seq[1].gate == :Rz, etc.
```
"""
function zyz_sequence(U::Matrix{ComplexF64})
    α, β, γ, _ = zyz_decompose(U)
    return [(gate=:Rz, angle=γ), (gate=:Ry, angle=β), (gate=:Rz, angle=α)]
end

"""
    gate_infidelity(U_achieved, U_target) -> Float64

Compute the gate infidelity 1 − F where

    F = |Tr(U_target† U_achieved)|² / dim²

# Example
```julia
ε = gate_infidelity(U_opt, CNOT_gate())
```
"""
function gate_infidelity(U_achieved :: Matrix{ComplexF64},
                          U_target   :: Matrix{ComplexF64})::Float64
    dim = size(U_target, 1)
    F   = abs2(tr(U_target' * U_achieved)) / Float64(dim^2)
    return 1.0 - F
end
