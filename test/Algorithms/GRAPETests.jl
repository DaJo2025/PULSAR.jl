# test/Algorithms/GRAPETests.jl
# ================================
# Unit and regression tests for the GRAPE (GRadient Ascent Pulse Engineering)
# optimizer.  Every sub-testset isolates one property of the algorithm so that
# a failure points immediately at the underlying bug.

using Test
using PULSAR
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Shared Pauli matrices
# ---------------------------------------------------------------------------
const _gσ_x = ComplexF64[0 1; 1 0]
const _gσ_y = ComplexF64[0 -im; im 0]
const _gσ_z = ComplexF64[1 0; 0 -1]
const _gI2  = ComplexF64[1 0; 0 1]

# ---------------------------------------------------------------------------
# Helper: build a resonant single-qubit system (no drift)
# ---------------------------------------------------------------------------
function _qubit_sys_resonant()
    H_drift = zeros(ComplexF64, 2, 2)
    # H_ctrl is scaled to a 1 MHz Rabi drive so that u ∈ O(1) over the
    # 10 µs / 100-step pulse spans a full π rotation — the regime where
    # GRAPE's gradient is informative and the optimizer can actually find
    # the X-gate solution from a small random initial guess.
    H_ctrl  = [2π * 1e6 * _gσ_x, 2π * 1e6 * _gσ_y]
    return quantum_system(H_drift, H_ctrl)
end

# ---------------------------------------------------------------------------
# Helper: build a two-qubit system with ZZ coupling
# ---------------------------------------------------------------------------
function _two_qubit_sys()
    I2 = _gI2
    ZI = kron(_gσ_z, I2)
    IZ = kron(I2, _gσ_z)
    ZZ = kron(_gσ_z, _gσ_z)
    XI = kron(_gσ_x, I2)
    YI = kron(_gσ_y, I2)
    IX = kron(I2, _gσ_x)
    IY = kron(I2, _gσ_y)

    # ZZ raised to 50 kHz so the entangling phase is reachable in the
    # 6-µs window allotted by the test, and the per-qubit X/Y drives are
    # scaled to a 1 MHz Rabi rate so a unit control performs a π flip
    # within the pulse window.
    H_drift = 2π * (80e3 * ZI + 60e3 * IZ + 50e3 * ZZ)
    H_ctrl  = [2π * 1e6 * XI, 2π * 1e6 * YI, 2π * 1e6 * IX, 2π * 1e6 * IY]
    return quantum_system(H_drift, H_ctrl)
end

# ---------------------------------------------------------------------------
@testset "GRAPE Algorithm" begin

    # -----------------------------------------------------------------------
    @testset "Simple qubit convergence – state transfer |0⟩ → |1⟩" begin

        sys    = _qubit_sys_resonant()
        ψ1     = ComplexF64[0, 1]
        target = state_target(ψ1)

        rng    = MersenneTwister(42)
        n_ts   = 100
        dt     = 1e-7          # 100 ns total → 10 µs
        u_init = 0.1 * randn(rng, Float64, 2, n_ts)

        config = GRAPEConfig(
            max_iter        = 500,
            convergence_tol = 1e-9,
            adapt_step_size = true,
            verbose         = false,
        )

        result = grape_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > 0.99
        @test result.converged || result.n_iterations == 500
        # Fidelity history is monotonically non-decreasing (with line search)
        diffs = diff(result.fidelity_history)
        @test all(d >= -1e-8 for d in diffs)

    end

    # -----------------------------------------------------------------------
    @testset "Fidelity monotonicity with backtracking line search" begin

        sys    = _qubit_sys_resonant()
        target = state_target(ComplexF64[0, 1])

        rng    = MersenneTwister(7)
        n_ts   = 50
        dt     = 1e-7
        u_init = 0.05 * randn(rng, Float64, 2, n_ts)

        config = GRAPEConfig(
            max_iter        = 200,
            adapt_step_size = true,   # enables backtracking
            verbose         = false,
        )

        result = grape_optimize(sys, target, u_init, dt; config=config)

        # With backtracking, the fidelity should never decrease by more than
        # numerical noise.
        hist = result.fidelity_history
        for k in 2:length(hist)
            @test hist[k] >= hist[k-1] - 1e-8
        end

    end

    # -----------------------------------------------------------------------
    @testset "Gradient vanishes at optimum" begin

        sys    = _qubit_sys_resonant()
        target = state_target(ComplexF64[0, 1])

        rng    = MersenneTwister(11)
        u_init = 0.1 * randn(rng, Float64, 2, 80)
        dt     = 1e-7

        config = GRAPEConfig(
            max_iter        = 600,
            convergence_tol = 1e-10,
            verbose         = false,
        )

        result = grape_optimize(sys, target, u_init, dt; config=config)

        # At convergence, gradient norm should be very small.
        # We allow a relaxed threshold because convergence criterion
        # may have been triggered by fidelity plateau, not gradient magnitude.
        @test result.gradient_norm_history[end] < 1e-3

    end

    # -----------------------------------------------------------------------
    @testset "Reproducibility – same seed same result" begin

        sys    = _qubit_sys_resonant()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        function run_once()
            rng    = MersenneTwister(2024)
            u_init = 0.05 * randn(rng, Float64, 2, 60)
            config = GRAPEConfig(max_iter=100, verbose=false)
            grape_optimize(sys, target, u_init, dt; config=config)
        end

        r1 = run_once()
        r2 = run_once()

        @test r1.fidelity == r2.fidelity
        @test norm(r1.controls - r2.controls) == 0.0
        @test r1.n_iterations == r2.n_iterations

    end

    # -----------------------------------------------------------------------
    @testset "Parallel gradient matches serial gradient" begin

        sys    = _qubit_sys_resonant()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(88)
        u      = 0.05 * randn(rng, Float64, 2, 80)

        g_serial   = grape_gradient(sys, target, u, dt; parallel=false)
        g_parallel = grape_gradient(sys, target, u, dt; parallel=true)

        @test norm(g_serial - g_parallel) / (norm(g_serial) + 1e-30) < 1e-12

    end

    # -----------------------------------------------------------------------
    @testset "Gate fidelity target – X gate on single qubit" begin

        sys    = _qubit_sys_resonant()
        target = gate_target(_gσ_x)   # optimize X (NOT) gate

        rng    = MersenneTwister(55)
        u_init = 0.1 * randn(rng, Float64, 2, 100)
        dt     = 1e-7

        config = GRAPEConfig(max_iter=600, convergence_tol=1e-9, verbose=false)
        result = grape_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > 0.99

    end

    # -----------------------------------------------------------------------
    @testset "2-qubit CNOT gate" begin

        sys = _two_qubit_sys()

        CNOT = ComplexF64[1 0 0 0;
                          0 1 0 0;
                          0 0 0 1;
                          0 0 1 0]
        target = gate_target(CNOT)

        rng    = MersenneTwister(33)
        n_ts   = 120
        dt     = 5e-8
        u_init = 0.02 * randn(rng, Float64, 4, n_ts)

        config = GRAPEConfig(
            max_iter        = 1000,
            convergence_tol = 1e-8,
            adapt_step_size = true,
            verbose         = false,
        )

        result = grape_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > 0.95

    end

    # -----------------------------------------------------------------------
    @testset "Fidelity improves from random initial guess" begin

        # Even a poorly chosen initial point should strictly improve.
        sys    = _qubit_sys_resonant()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(999)
        u_init = 0.001 * randn(rng, Float64, 2, 30)  # very small amplitudes
        F0     = evaluate_fidelity(sys, target, u_init, dt)

        config = GRAPEConfig(max_iter=50, verbose=false)
        result = grape_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > F0

    end

    # -----------------------------------------------------------------------
    @testset "Step size adaptation" begin

        # Test that the step-size returned by the line search is finite
        # and strictly positive.
        sys    = _qubit_sys_resonant()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(64)
        u_init = 0.1 * randn(rng, Float64, 2, 60)

        config = GRAPEConfig(
            max_iter        = 100,
            adapt_step_size = true,
            verbose         = false,
        )

        result = grape_optimize(sys, target, u_init, dt; config=config)

        @test all(isfinite.(result.step_size_history))
        @test all(result.step_size_history .> 0)

    end

    # -----------------------------------------------------------------------
    @testset "Long pulse sequence stability" begin

        # 500 timesteps with small dt — checks numerics do not degrade.
        sys    = _qubit_sys_resonant()
        target = state_target(ComplexF64[0, 1])
        dt     = 2e-8   # 20 ns per step → 10 µs total

        rng    = MersenneTwister(123)
        u_init = 0.05 * randn(rng, Float64, 2, 500)

        config = GRAPEConfig(max_iter=200, verbose=false)
        result = grape_optimize(sys, target, u_init, dt; config=config)

        # Main check: all propagators stay unitary throughout (indirectly
        # verified by finite fidelity without NaN).
        @test isfinite(result.fidelity)
        @test result.fidelity >= 0.0
        @test result.fidelity <= 1.0 + 1e-10

    end

end  # GRAPE Algorithm
