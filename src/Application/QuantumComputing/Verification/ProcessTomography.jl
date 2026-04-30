# Application/QuantumComputing/Verification/ProcessTomography.jl
# Quantum Process Tomography (QPT) for gate verification.
#
# QPT reconstructs the process matrix (χ-matrix or Choi matrix) of a quantum
# operation from a set of input states and measurement outcomes.
#
# For an N-qubit gate (dim = 2^N):
#   - Prepare 4^N input states (all combinations of {|0⟩, |1⟩, |+⟩, |+i⟩} per qubit)
#   - Apply the gate
#   - Measure all dim²-1 Pauli expectation values
#   - Reconstruct the process via linear inversion or MLE
#
# Provides:
#   QPTResult                  — struct: QPT reconstruction output
#   qpt_input_states           — generate standard QPT input states
#   qpt_reconstruct_linear     — linear inversion QPT
#   qpt_choi_matrix            — Choi matrix from propagator (exact, ideal)
#   process_fidelity           — F_process between reconstructed and target
#   average_gate_fidelity      — F_avg from process fidelity

using LinearAlgebra

# ============================================================================
# QPTResult
# ============================================================================

"""
    QPTResult

Results from quantum process tomography.

# Fields
- `chi_matrix`       — Matrix{ComplexF64}: process matrix in the Pauli basis (dim²×dim²)
- `choi_matrix`      — Matrix{ComplexF64}: Choi-Jamiołkowski matrix (dim²×dim²)
- `process_fidelity` — Float64: F_proc between reconstructed and target process
- `avg_gate_fidelity`— Float64: F_avg = (dim × F_proc + 1) / (dim + 1)
- `is_cptp`          — Bool: whether the reconstructed channel is CPTP (within tolerance)
- `dim`              — Int: Hilbert space dimension
"""
struct QPTResult
    chi_matrix        :: Matrix{ComplexF64}
    choi_matrix       :: Matrix{ComplexF64}
    process_fidelity  :: Float64
    avg_gate_fidelity :: Float64
    is_cptp           :: Bool
    dim               :: Int
end

# ============================================================================
# Input state generators
# ============================================================================

"""
    qpt_input_states(n_qubits::Int) -> Vector{Vector{ComplexF64}}

Generate the standard set of 4^n_qubits QPT input states for n qubits.

For each qubit, the four states are:
  |+z⟩ = |0⟩,  |−z⟩ = |1⟩,  |+x⟩ = (|0⟩+|1⟩)/√2,  |+y⟩ = (|0⟩+i|1⟩)/√2

The full multi-qubit set is the tensor product of single-qubit sets.

# Returns
`Vector{Vector{ComplexF64}}` of length 4^n_qubits, each of length 2^n_qubits.
"""
function qpt_input_states(n_qubits :: Int)::Vector{Vector{ComplexF64}}
    # Single-qubit QPT states
    sq_states = [
        ComplexF64[1.0, 0.0],                     # |0⟩
        ComplexF64[0.0, 1.0],                     # |1⟩
        ComplexF64[1.0, 1.0]   ./ sqrt(2.0),      # |+x⟩
        ComplexF64[1.0, im]    ./ sqrt(2.0),       # |+y⟩
    ]

    # Build tensor product for n_qubits
    states = [ComplexF64[1.0]]
    for _ in 1:n_qubits
        new_states = Vector{ComplexF64}[]
        for ψ_prev in states, ψ_new in sq_states
            push!(new_states, kron(ψ_prev, ψ_new))
        end
        states = new_states
    end
    return states
end

# ============================================================================
# Choi matrix (exact, from propagator)
# ============================================================================

"""
    qpt_choi_matrix(U) -> Matrix{ComplexF64}

Compute the Choi-Jamiołkowski matrix of an ideal unitary gate U.

The Choi matrix is:

    Λ = (I ⊗ U) |Φ⟩⟨Φ| (I ⊗ U†)

where |Φ⟩ = Σ_i |i⟩|i⟩ / √dim is the maximally entangled state.

For a unitary U, the Choi matrix is a pure state: Λ = |φ_U⟩⟨φ_U|.

# Arguments
- `U` — dim×dim unitary matrix

# Returns
dim²×dim² Choi matrix.
"""
function qpt_choi_matrix(U :: Matrix{ComplexF64})::Matrix{ComplexF64}
    dim = size(U, 1)
    # Build |Φ⟩ = Σ_i |i⟩⊗|i⟩ / √dim (column-stacked)
    Phi = zeros(ComplexF64, dim^2)
    for i in 1:dim
        ei = zeros(ComplexF64, dim); ei[i] = 1.0
        Phi[(i-1)*dim+1 : i*dim] .= ei ./ sqrt(Float64(dim))
    end
    IU = kron(Matrix{ComplexF64}(I, dim, dim), U)
    phi_out = IU * Phi
    return phi_out * phi_out'
end

# ============================================================================
# Linear inversion QPT
# ============================================================================

"""
    qpt_reconstruct_linear(U_propagated, n_qubits; U_target=nothing) -> QPTResult

Reconstruct the process matrix via linear inversion QPT from the set of
output states obtained by propagating each QPT input state through the process.

In the ideal (noiseless) case the input is the set of output states
`{U |ψ_k⟩}` for all QPT input states `|ψ_k⟩`.

# Arguments
- `U_propagated` — Matrix{ComplexF64}: the gate/propagator whose process to reconstruct
- `n_qubits`     — Int: number of qubits
- `U_target`     — optional ideal target gate; used to compute process fidelity

# Returns
[`QPTResult`](@ref)

# Example
```julia
# Characterise an optimised X gate
U_opt = result_x_gate_propagator
qpt   = qpt_reconstruct_linear(U_opt, 1; U_target=X_gate())
@printf("Process fidelity: %.4f\\n", qpt.process_fidelity)
```
"""
function qpt_reconstruct_linear(U_propagated :: Matrix{ComplexF64},
                                  n_qubits     :: Int;
                                  U_target     :: Union{Matrix{ComplexF64}, Nothing} = nothing
                                  )::QPTResult
    dim    = 2^n_qubits

    # Choi matrix of the propagator (exact)
    choi = qpt_choi_matrix(U_propagated)

    # Convert Choi to χ-matrix (process matrix in the normalised Pauli basis).
    # For a process ε, the Choi matrix Λ and the χ-matrix are related by:
    #   Λ_{(i,k),(j,l)} = Σ_{μν} χ_{μν} (E_μ)_{ij} (E_ν*)_{kl}
    # Inversion: χ = (T† T)^{-1} T† vec(Λ) where T is the transfer matrix.
    # For a unitary, a simpler route: χ_{μν} = Tr[E_μ† U] Tr[E_ν† U]* / dim
    B     = _pauli_basis(dim)
    n_ops = length(B)
    chi   = zeros(ComplexF64, n_ops, n_ops)

    # c_μ = Tr[E_μ† U] / √dim  (expansion coefficients of U in Pauli basis)
    c = ComplexF64[tr(B[μ]' * U_propagated) / sqrt(Float64(dim)) for μ in 1:n_ops]
    for μ in 1:n_ops, ν in 1:n_ops
        chi[μ, ν] = c[μ] * conj(c[ν])
    end

    # Process fidelity against target
    F_proc = 0.0
    F_avg  = 0.0
    if !isnothing(U_target)
        choi_target = qpt_choi_matrix(U_target)
        F_proc = real(tr(choi_target' * choi)) / Float64(dim^2)
        F_avg  = process_fidelity_to_avg(F_proc, dim)
    end

    # CPTP check: Choi matrix should be PSD with trace = dim
    evals   = real.(eigvals(choi))
    is_cptp = all(evals .>= -1e-6) && abs(real(tr(choi)) - Float64(dim)) < 1e-4

    return QPTResult(chi, choi, F_proc, F_avg, is_cptp, dim)
end

# ============================================================================
# Fidelity functions
# ============================================================================

"""
    process_fidelity(choi1, choi2) -> Float64

Compute the process fidelity between two quantum processes given by their
Choi matrices:

    F_proc = Tr(Choi_1† Choi_2) / dim²
"""
function process_fidelity(choi1 :: Matrix{ComplexF64},
                            choi2 :: Matrix{ComplexF64})::Float64
    dim2 = size(choi1, 1)   # dim^2
    dim  = round(Int, sqrt(dim2))
    return real(tr(choi1' * choi2)) / Float64(dim^2)
end

"""
    process_fidelity_to_avg(F_proc, dim) -> Float64

Convert process fidelity F_proc to average gate fidelity F_avg using the
Horodecki relation:

    F_avg = (dim × F_proc + 1) / (dim + 1)
"""
function process_fidelity_to_avg(F_proc :: Float64, dim :: Int)::Float64
    return (Float64(dim) * F_proc + 1.0) / Float64(dim + 1)
end

"""
    average_gate_fidelity(U_achieved, U_target) -> Float64

Compute the average gate fidelity between an achieved unitary and the target:

    F_avg = (dim × |Tr(U_target† U)|² / dim² + 1) / (dim + 1)
          = (|Tr(U_target† U)|² / dim + 1) / (dim + 1)
"""
function average_gate_fidelity(U_achieved :: Matrix{ComplexF64},
                                 U_target   :: Matrix{ComplexF64})::Float64
    dim    = size(U_target, 1)
    F_proc = abs2(tr(U_target' * U_achieved)) / Float64(dim^2)
    return process_fidelity_to_avg(F_proc, dim)
end

# ============================================================================
# Internal: Pauli basis
# ============================================================================

"""
    _pauli_basis(dim) -> Vector{Matrix{ComplexF64}}

Return the normalised Pauli operator basis for a dim×dim Hilbert space.
For dim = 2^n, this is the n-qubit Pauli group {I,X,Y,Z}^⊗n / √dim.
"""
function _pauli_basis(dim :: Int)::Vector{Matrix{ComplexF64}}
    n_qubits = round(Int, log2(dim))
    @assert 2^n_qubits == dim "dim must be a power of 2"

    I2 = ComplexF64[1 0; 0 1]
    X  = ComplexF64[0 1; 1 0]
    Y  = ComplexF64[0 -im; im 0]
    Z  = ComplexF64[1 0; 0 -1]
    paulis1 = [I2, X, Y, Z]

    ops = [ComplexF64[1.0+0im][:, :]]   # start with 1×1 identity
    for _ in 1:n_qubits
        new_ops = Matrix{ComplexF64}[]
        for prev in ops, p in paulis1
            push!(new_ops, kron(prev, p))
        end
        ops = new_ops
    end
    # Normalise: E_k / √dim
    return [op ./ sqrt(Float64(dim)) for op in ops]
end

# ============================================================================
# Quick summary printout
# ============================================================================

"""
    print_qpt_summary(result::QPTResult)

Print a human-readable summary of a [`QPTResult`](@ref).
"""
function print_qpt_summary(result :: QPTResult)
    @printf("  QPT Summary\n")
    @printf("  %-22s : %d\n", "Hilbert dim",        result.dim)
    @printf("  %-22s : %.6f\n", "Process fidelity",  result.process_fidelity)
    @printf("  %-22s : %.6f\n", "Avg gate fidelity", result.avg_gate_fidelity)
    @printf("  %-22s : %s\n",   "CPTP",               result.is_cptp ? "yes" : "no")
    # Dominant χ-matrix component
    chi = result.chi_matrix
    max_idx = argmax(abs.(chi))
    @printf("  %-22s : [%d,%d]  |χ| = %.4f\n",
            "Dominant χ component", max_idx[1], max_idx[2],
            abs(chi[max_idx]))
end
