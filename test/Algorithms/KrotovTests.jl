# test/Algorithms/KrotovTests.jl
# =========================================
# Tests for the real Krotov method (src/Optimization/Gradient/QOC/Krotov.jl).
# Covers:
#   1. First-order state-transfer convergence + strict monotonicity
#   2. Second-order state-transfer (σ > 0) convergence
#   3. σ = 0 equivalence between second-order and first-order
#   4. Rollback path — small λ_a with enforce_monotonic=true must still
#      converge monotonically, since bad steps are rolled back.
#   5. check_invariants=true does not raise on the canonical problem.
#   6. Unitary-target driver runs (documented gradient-degeneracy caveat
#      applies — we exercise the code path with a guess that has non-zero
#      Hilbert-Schmidt overlap with the target).

using Test
using Pulsar
using LinearAlgebra
using Random

const _KσX = ComplexF64[0 1; 1 0]
const _KσY = ComplexF64[0 -im; im 0]
const _KI2 = Matrix{ComplexF64}(I, 2, 2)

# Standard 1-qubit driven system:  H_drift = 0,  H_controls = {σx/2, σy/2}
function _qubit_system(; init = ComplexF64[1, 0])
    H_drift = zeros(ComplexF64, 2, 2)
    H_ctrl  = [_KσX / 2, _KσY / 2]
    return QuantumSystem(H_drift, H_ctrl, 2, 2,
                         Dict{String,Any}("init_state" => ComplexF64.(init)))
end

@testset "Real Krotov Method" begin

    # -----------------------------------------------------------------------
    @testset "First-order state transfer |0⟩ → |1⟩" begin
        sys  = _qubit_system()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 40
        amps = 0.1 .* randn(MersenneTwister(17), N, 2)
        ctrl = ControlSequence(amps, dt, N)

        _, F, st = krotov_optimize(sys, tgt, ctrl;
                                    λ_a = 5.0, max_iter = 500, tol = 1e-10,
                                    check_invariants = true, verbose = false)

        @test F > 0.999
        @test st.converged
        # Strict monotone ascent must hold when enforce_monotonic=true (default)
        @test all(diff(st.history) .>= -1e-10)
    end

    # -----------------------------------------------------------------------
    @testset "Second-order Krotov (σ > 0) — converges" begin
        sys  = _qubit_system()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 40
        amps = 0.1 .* randn(MersenneTwister(19), N, 2)
        ctrl = ControlSequence(amps, dt, N)

        _, F, st = krotov_second_order_optimize(sys, tgt, ctrl;
                                                 λ_a = 5.0, σ = 0.1,
                                                 max_iter = 500, tol = 1e-10,
                                                 check_invariants = true)
        @test F > 0.999
        @test all(diff(st.history) .>= -1e-10)
    end

    # -----------------------------------------------------------------------
    @testset "σ = 0 reduces to first-order" begin
        # With identical seeds + λ_a the σ=0 second-order driver must match
        # first-order bit-for-bit on the accepted fidelity history.
        sys  = _qubit_system()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 20
        Random.seed!(42)
        amps = 0.1 .* randn(N, 2)
        ctrl = ControlSequence(amps, dt, N)

        _, F1, st1 = krotov_optimize(sys, tgt, ctrl;
                                      λ_a = 5.0, max_iter = 50, tol = -Inf)
        _, F2, st2 = krotov_second_order_optimize(sys, tgt, ctrl;
                                                   λ_a = 5.0, σ = 0.0,
                                                   max_iter = 50, tol = -Inf)

        @test isapprox(F1, F2; atol = 1e-12)
        @test length(st1.history) == length(st2.history)
        @test isapprox(st1.history, st2.history; atol = 1e-12)
    end

    # -----------------------------------------------------------------------
    @testset "Rollback path — small λ_a still monotonic with enforcement" begin
        sys  = _qubit_system()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 40
        amps = 0.1 .* randn(MersenneTwister(23), N, 2)
        ctrl = ControlSequence(amps, dt, N)

        # λ_a=0.1 is too small for monotonicity in 1 shot; the driver must
        # rollback + double λ_a and still reach a good F without raising.
        _, F, st = krotov_optimize(sys, tgt, ctrl;
                                    λ_a = 0.1, max_iter = 500, tol = 1e-10,
                                    enforce_monotonic = true,
                                    check_invariants  = true)

        @test F > 0.99
        @test all(diff(st.history) .>= -1e-10)
    end

    # -----------------------------------------------------------------------
    @testset "Unitary-target code path runs with nonzero overlap guess" begin
        # The real-trace gate functional has documented zero-gradient fixed
        # points; we simply verify the unitary branch executes end-to-end
        # without error and that monotonicity is respected when steps are
        # accepted.  A Hadamard target gives a guess with nonzero
        # Hilbert-Schmidt overlap against near-identity controls.
        H_gate = ComplexF64[1 1; 1 -1] / sqrt(2)
        sys    = _qubit_system()
        tgt    = unitary_target(H_gate)
        dt, N  = 0.05, 40
        amps   = 0.3 .* randn(MersenneTwister(5), N, 2)
        ctrl   = ControlSequence(amps, dt, N)

        _, F, st = krotov_optimize(sys, tgt, ctrl;
                                    λ_a = 5.0, max_iter = 300, tol = 1e-10,
                                    check_invariants = true)

        @test isfinite(F)
        @test length(st.history) ≥ 1
        @test all(diff(st.history) .>= -1e-10)
    end

end  # @testset "Real Krotov Method"
