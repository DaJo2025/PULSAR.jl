# Application/QuantumComputing/Verification/RandomizedBenchmarking.jl
# Randomized Benchmarking (RB) protocol for characterising gate error rates.
#
# Standard RB (Emerson et al. 2005, Magesan et al. 2011) measures the average
# error per Clifford gate by preparing random Clifford sequences of increasing
# length m and measuring the survival probability:
#
#   p(m) = A × r_g^m + B
#
# where r_g = 1 − EPC is the error per Clifford (EPC = error per Clifford gate).
#
# Provides:
#   RBResult                   — struct: fitted RB decay parameters
#   rb_sequence                — generate a random Clifford sequence
#   rb_survival_probability    — simulate p(m) for ideal/noisy gates
#   fit_rb_decay               — fit A r^m + B to measured p(m) data
#   estimate_epc               — estimate error per Clifford from RB data
#   interleaved_rb             — interleaved RB for single-gate characterisation

using LinearAlgebra
using Random
using Statistics

# ============================================================================
# Single-qubit Clifford group (24 elements)
# ============================================================================

"""
    _single_qubit_clifford_group() -> Vector{Matrix{ComplexF64}}

Return all 24 single-qubit Clifford matrices.
"""
function _single_qubit_clifford_group()::Vector{Matrix{ComplexF64}}
    # Generate from generators H and S by BFS closure
    H  = H_gate()
    S  = S_gate()
    I2 = I_gate(2)
    group = Matrix{ComplexF64}[I2]
    queue = Matrix{ComplexF64}[I2]
    while !isempty(queue)
        U = popfirst!(queue)
        for g in (H, S)
            for V in (U * g, g * U)
                # Check if V is already in group (up to global phase)
                is_new = true
                for W in group
                    if norm(V - W) < 1e-9 || norm(V + W) < 1e-9 ||
                       norm(V - im*W) < 1e-9 || norm(V + im*W) < 1e-9
                        is_new = false
                        break
                    end
                end
                if is_new
                    push!(group, V)
                    push!(queue, V)
                end
            end
        end
        length(group) >= 24 && break
    end
    return group
end

# Cache the Clifford group on first use
const _CLIFFORD1 = Ref{Union{Nothing, Vector{Matrix{ComplexF64}}}}(nothing)

function _get_clifford1()::Vector{Matrix{ComplexF64}}
    if isnothing(_CLIFFORD1[])
        _CLIFFORD1[] = _single_qubit_clifford_group()
    end
    return _CLIFFORD1[]
end

# ============================================================================
# RBResult
# ============================================================================

"""
    RBResult

Results from a randomized benchmarking experiment.

# Fields
- `m_values`   — Vector{Int}: Clifford sequence lengths used
- `p_survival` — Vector{Float64}: measured/simulated survival probabilities
- `A`          — Float64: fitted amplitude parameter
- `B`          — Float64: fitted offset parameter
- `r_g`        — Float64: fitted per-Clifford fidelity decay rate (1 − EPC)
- `epc`        — Float64: error per Clifford = 1 − r_g
- `epg`        — Float64: error per gate ≈ EPC / (d²−1) × d²  (d = 2 for qubit)
- `r_squared`  — Float64: goodness-of-fit R²
"""
struct RBResult
    m_values   :: Vector{Int}
    p_survival :: Vector{Float64}
    A          :: Float64
    B          :: Float64
    r_g        :: Float64
    epc        :: Float64
    epg        :: Float64
    r_squared  :: Float64
end

# ============================================================================
# Sequence generation
# ============================================================================

"""
    rb_sequence(m; n_qubits=1, rng=GLOBAL_RNG) -> Vector{Matrix{ComplexF64}}

Generate a random Clifford gate sequence of length m+1, where the last gate
is the inverse of the product of the first m gates (ensuring the sequence
returns to the identity when applied to |0⟩).

# Arguments
- `m`        — Int: number of random Clifford gates (not counting the recovery gate)
- `n_qubits` — Int: number of qubits (currently 1 or 2 supported)
- `rng`      — AbstractRNG

# Returns
Vector of m+1 gate matrices; applying them in order should give ≈ identity.

# Example
```julia
seq = rb_sequence(10)
U_total = foldl(*, seq)
@assert norm(U_total - I_gate(2)) < 1e-10
```
"""
function rb_sequence(m        :: Int;
                      n_qubits :: Int         = 1,
                      rng      :: AbstractRNG  = Random.GLOBAL_RNG
                      )::Vector{Matrix{ComplexF64}}
    @assert m >= 1 "sequence length must be ≥ 1"
    @assert n_qubits == 1 "multi-qubit RB: only n_qubits=1 currently supported"

    cliffords = _get_clifford1()
    n_cliff   = length(cliffords)
    seq       = Matrix{ComplexF64}[]
    U_cumul   = Matrix{ComplexF64}(I, 2, 2)

    for _ in 1:m
        idx = rand(rng, 1:n_cliff)
        C   = cliffords[idx]
        push!(seq, C)
        U_cumul = C * U_cumul
    end

    # Recovery gate: inverse of U_cumul
    U_inv = U_cumul'   # Clifford unitaries are unitary so U† = U^{-1}
    push!(seq, U_inv)

    return seq
end

# ============================================================================
# Survival probability simulation
# ============================================================================

"""
    rb_survival_probability(m_values, gate_fn;
                            n_qubits=1, n_sequences=50, rng=GLOBAL_RNG)
    -> Vector{Float64}

Simulate the RB survival probability p(m) for a set of sequence lengths.

`gate_fn(U_ideal) -> Matrix{ComplexF64}` maps an ideal Clifford gate to its
noisy implementation.  For ideal (noiseless) simulation use `gate_fn = identity`.
For simulating a uniform gate error pass a function that applies a fixed
noise channel.

# Arguments
- `m_values`    — Vector{Int}: sequence lengths to simulate
- `gate_fn`     — Function: `U_ideal -> U_noisy` gate noise model
- `n_qubits`    — Int (default 1)
- `n_sequences` — Int: number of random sequences per length (default 50)
- `rng`         — AbstractRNG

# Returns
`Vector{Float64}` mean survival probabilities, one per entry in `m_values`.
"""
function rb_survival_probability(m_values    :: AbstractVector{Int},
                                  gate_fn     :: Function;
                                  n_qubits    :: Int        = 1,
                                  n_sequences :: Int        = 50,
                                  rng         :: AbstractRNG = Random.GLOBAL_RNG
                                  )::Vector{Float64}
    dim   = 2^n_qubits
    psi0  = zeros(ComplexF64, dim)
    psi0[1] = 1.0    # |0…0⟩ initial state
    P0    = psi0 * psi0'   # |0⟩⟨0| projector

    p_mean = Float64[]

    for m in m_values
        p_vals = Float64[]
        for _ in 1:n_sequences
            seq = rb_sequence(m; n_qubits=n_qubits, rng=rng)
            U   = Matrix{ComplexF64}(I, dim, dim)
            for C in seq
                U = gate_fn(C) * U
            end
            psi_f = U * psi0
            push!(p_vals, abs2(dot(psi0, psi_f)))
        end
        push!(p_mean, mean(p_vals))
    end
    return p_mean
end

# ============================================================================
# RB decay fitting (Levenberg-Marquardt-style via simple gradient descent)
# ============================================================================

"""
    fit_rb_decay(m_values, p_survival; maxiter=1000) -> (A, r, B)

Fit the RB decay model p(m) = A × r^m + B using iterative least squares.

# Returns
Tuple `(A, r, B)` of fitted parameters.
"""
function fit_rb_decay(m_values   :: AbstractVector{Int},
                       p_survival :: AbstractVector{Float64};
                       maxiter    :: Int = 1000)
    # Initial guess
    n = length(m_values)
    A = 0.5;  r = 0.99;  B = 0.5

    # Simple gradient descent on sum-of-squares residuals
    lr = 1e-4
    for _ in 1:maxiter
        pred = [A * r^m + B for m in m_values]
        res  = pred .- p_survival
        dA   = 2.0 * sum(res[i] * r^m_values[i]       for i in 1:n)
        dr   = 2.0 * sum(res[i] * A * m_values[i] * r^(m_values[i]-1) for i in 1:n)
        dB   = 2.0 * sum(res)
        A -= lr * dA;  r -= lr * dr;  B -= lr * dB
        r  = clamp(r, 0.0, 1.0)
        A  = max(A, 0.0)
        B  = clamp(B, 0.0, 1.0)
    end
    return A, r, B
end

"""
    estimate_epc(m_values, p_survival; n_qubits=1, maxiter=1000) -> RBResult

Fit an RB decay curve and return an [`RBResult`](@ref) with the estimated
error per Clifford (EPC) and error per gate (EPG).

# Arguments
- `m_values`    — Vector{Int} Clifford sequence lengths
- `p_survival`  — Vector{Float64} measured survival probabilities
- `n_qubits`    — Int (default 1; used to compute EPG from EPC)
- `maxiter`     — Int fitting iterations (default 1000)

# Returns
[`RBResult`](@ref)

# Example
```julia
m_vals = [1, 2, 5, 10, 20, 50, 100]
p_data = rb_survival_probability(m_vals, identity)
result = estimate_epc(m_vals, p_data)
@printf("EPC = %.4f\\n", result.epc)
```
"""
function estimate_epc(m_values    :: AbstractVector{Int},
                       p_survival  :: AbstractVector{Float64};
                       n_qubits    :: Int = 1,
                       maxiter     :: Int = 1000)::RBResult
    A, r, B = fit_rb_decay(m_values, p_survival; maxiter=maxiter)

    epc = 1.0 - r
    d   = 2^n_qubits
    # EPG = EPC × d² / (d²−1) for a uniform Clifford compilation
    epg = epc * Float64(d^2) / Float64(d^2 - 1)

    # R² goodness-of-fit
    p_pred = [A * r^m + B for m in m_values]
    ss_res = sum((p_pred[i] - p_survival[i])^2 for i in eachindex(p_survival))
    ss_tot = sum((p_survival[i] - mean(p_survival))^2 for i in eachindex(p_survival))
    r2 = ss_tot > 1e-15 ? 1.0 - ss_res / ss_tot : 1.0

    return RBResult(collect(m_values), collect(p_survival),
                    A, B, r, epc, epg, r2)
end

# ============================================================================
# Interleaved RB
# ============================================================================

"""
    interleaved_rb(U_gate, m_values;
                   gate_fn=identity, n_sequences=50, n_qubits=1, rng=GLOBAL_RNG)
    -> (RBResult, RBResult, Float64)

Perform interleaved randomized benchmarking to characterise the error rate of
a specific gate `U_gate`.

In interleaved RB, every random Clifford gate is followed by the gate under
test.  Comparing the decay rates of the standard and interleaved experiments
gives the error rate of the specific gate.

# Returns
Tuple `(rb_ref, rb_interleaved, epc_gate)` where
- `rb_ref`           — [`RBResult`](@ref) from standard RB (reference)
- `rb_interleaved`   — [`RBResult`](@ref) from interleaved RB
- `epc_gate`         — estimated error per gate

# Example
```julia
U_opt = result.controls  # optimal X gate from GRAPE
# Gate function: apply noise channel after each gate
noise_fn(U) = noisy_gate(U)
rb_ref, rb_int, epc = interleaved_rb(X_gate(), [1,2,5,10,20,50];
                                      gate_fn=noise_fn)
@printf("X gate EPC = %.4e\\n", epc)
```
"""
function interleaved_rb(U_gate      :: Matrix{ComplexF64},
                         m_values    :: AbstractVector{Int};
                         gate_fn     :: Function    = identity,
                         n_sequences :: Int         = 50,
                         n_qubits    :: Int         = 1,
                         rng         :: AbstractRNG  = Random.GLOBAL_RNG)
    # Standard RB (reference)
    p_ref = rb_survival_probability(m_values, gate_fn;
                                     n_qubits=n_qubits,
                                     n_sequences=n_sequences, rng=rng)
    rb_ref = estimate_epc(m_values, p_ref; n_qubits=n_qubits)

    # Interleaved RB: insert U_gate after each random Clifford
    cliffords = _get_clifford1()
    n_cliff   = length(cliffords)
    dim       = 2^n_qubits
    psi0      = zeros(ComplexF64, dim); psi0[1] = 1.0

    p_int = Float64[]
    for m in m_values
        p_vals = Float64[]
        for _ in 1:n_sequences
            seq     = rb_sequence(m; n_qubits=n_qubits, rng=rng)
            U       = Matrix{ComplexF64}(I, dim, dim)
            for k in 1:m
                U = gate_fn(seq[k]) * U        # random Clifford
                U = gate_fn(U_gate) * U         # interleaved gate
            end
            U = gate_fn(seq[end]) * U           # recovery gate
            push!(p_vals, abs2(dot(psi0, U * psi0)))
        end
        push!(p_int, mean(p_vals))
    end
    rb_int = estimate_epc(m_values, p_int; n_qubits=n_qubits)

    # Gate EPC: r_int / r_ref per Magesan et al. Eq. (4)
    epc_gate = (1.0 - rb_int.r_g / rb_ref.r_g) * Float64(2^n_qubits - 1) /
               Float64(2^n_qubits)

    return rb_ref, rb_int, max(epc_gate, 0.0)
end
