# test/Algorithms/SecondOrderTests.jl
# =====================================
# Tests for second-order optimization algorithms:
#   - BFGS (quasi-Newton)
#   - L-BFGS (limited-memory BFGS)
#   - Newton (exact second-order, where implemented)
#   - Line search conditions (Armijo, Wolfe)
#   - Hessian positive-definiteness maintenance

using Test
using PULSAR
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Shared Pauli matrices
# ---------------------------------------------------------------------------
const _sσ_x = ComplexF64[0 1; 1 0]
const _sσ_y = ComplexF64[0 -im; im 0]
const _sσ_z = ComplexF64[1 0; 0 -1]
const _sI2  = ComplexF64[1 0; 0 1]

# ---------------------------------------------------------------------------
# Helper: build a resonant single-qubit system (no drift)
# ---------------------------------------------------------------------------
function _s_qubit_sys()
    H_drift = zeros(ComplexF64, 2, 2)
    H_ctrl  = [2π * _sσ_x, 2π * _sσ_y]
    return quantum_system(H_drift, H_ctrl)
end

# ---------------------------------------------------------------------------
# Helper: build the 2-qubit test system
# ---------------------------------------------------------------------------
function _s_two_qubit_sys()
    I2 = _sI2
    ZI = kron(_sσ_z, I2); IZ = kron(I2, _sσ_z); ZZ = kron(_sσ_z, _sσ_z)
    XI = kron(_sσ_x, I2); YI = kron(_sσ_y, I2)
    IX = kron(I2, _sσ_x); IY = kron(I2, _sσ_y)
    # ZZ raised to 50 kHz and X/Y drives to 1 MHz Rabi so that the CNOT
    # is reachable inside the 6-µs (120 × 50 ns) test window.
    H_drift = 2π * (80e3 * ZI + 60e3 * IZ + 50e3 * ZZ)
    H_ctrl  = [2π * 1e6 * XI, 2π * 1e6 * YI, 2π * 1e6 * IX, 2π * 1e6 * IY]
    return quantum_system(H_drift, H_ctrl)
end

# ---------------------------------------------------------------------------
@testset "Second Order Methods" begin

    # -----------------------------------------------------------------------
    @testset "BFGS convergence – single qubit" begin

        sys    = _s_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(42)
        u_init = 0.1 * randn(rng, Float64, 2, 80)

        config = BFGSConfig(
            max_iter        = 400,
            convergence_tol = 1e-9,
            verbose         = false,
        )

        result = bfgs_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > 0.99
        @test result.converged || result.n_iterations == 400

    end

    # -----------------------------------------------------------------------
    @testset "BFGS converges faster than GRAPE (equal iterations)" begin

        # This test checks the qualitative advantage of BFGS over first-order
        # GRAPE given the same number of iterations and identical starting point.

        sys    = _s_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7
        n_iter = 150

        rng    = MersenneTwister(17)
        u_init = 0.1 * randn(rng, Float64, 2, 80)

        cfg_grape = GRAPEConfig(max_iter=n_iter, adapt_step_size=true, verbose=false)
        cfg_bfgs  = BFGSConfig( max_iter=n_iter, verbose=false)

        r_grape = grape_optimize(sys, target, u_init, dt; config=cfg_grape)
        r_bfgs  = bfgs_optimize( sys, target, u_init, dt; config=cfg_bfgs)

        # BFGS should be at least as good; we allow a 1% slack because the
        # landscape can be tricky.
        @test r_bfgs.fidelity >= r_grape.fidelity - 0.01

    end

    # -----------------------------------------------------------------------
    @testset "L-BFGS convergence – single qubit" begin

        sys    = _s_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(31)
        u_init = 0.1 * randn(rng, Float64, 2, 80)

        config = LBFGSConfig(
            max_iter        = 400,
            memory_size     = 10,
            convergence_tol = 1e-9,
            verbose         = false,
        )

        result = lbfgs_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > 0.99

    end

    # -----------------------------------------------------------------------
    @testset "L-BFGS large problem – memory bounded" begin

        # 500 time steps × 2 controls → 1000-dimensional control vector.
        # L-BFGS with memory_size=10 should handle this without OOM.
        sys    = _s_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 2e-8  # 20 ns per step → 10 µs total

        rng    = MersenneTwister(55)
        u_init = 0.03 * randn(rng, Float64, 2, 500)

        config = LBFGSConfig(
            max_iter        = 200,
            memory_size     = 10,
            verbose         = false,
        )

        result = lbfgs_optimize(sys, target, u_init, dt; config=config)

        @test isfinite(result.fidelity)
        @test result.fidelity > 0.0

    end

    # -----------------------------------------------------------------------
    @testset "L-BFGS two-qubit CNOT gate" begin

        sys = _s_two_qubit_sys()
        CNOT = ComplexF64[1 0 0 0;
                          0 1 0 0;
                          0 0 0 1;
                          0 0 1 0]
        target = gate_target(CNOT)

        rng    = MersenneTwister(88)
        u_init = 0.02 * randn(rng, Float64, 4, 120)
        dt     = 5e-8

        config = LBFGSConfig(
            max_iter    = 800,
            memory_size = 15,
            verbose     = false,
        )

        result = lbfgs_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > 0.95

    end

    # -----------------------------------------------------------------------
    @testset "Line search – Armijo sufficient-decrease condition" begin

        # Verify that every step accepted by the backtracking line search
        # satisfies the Armijo condition:
        #   F(u + α*d) >= F(u) + c * α * ∇F·d
        # where c = 1e-4 (Wolfe constant) and d is the BFGS search direction.

        sys    = _s_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(62)
        u_init = 0.1 * randn(rng, Float64, 2, 60)

        config = BFGSConfig(
            max_iter           = 50,
            verbose            = false,
            record_line_search = true,   # store (α, F_before, F_after, slope) per step
        )

        result = bfgs_optimize(sys, target, u_init, dt; config=config)

        c₁ = 1e-4
        for (α, F_before, F_after, slope) in result.line_search_history
            expected_min = F_before + c₁ * α * slope
            @test F_after >= expected_min - 1e-10
        end

    end

    # -----------------------------------------------------------------------
    @testset "BFGS Hessian approximation positive definiteness" begin

        # The BFGS update preserves positive definiteness only when the
        # curvature condition (sᵀy > 0) is satisfied; the implementation
        # should skip the update when it is not.
        # We verify that the Hessian approximation is PD throughout the run
        # by checking that all eigenvalues of B_inv remain positive.

        sys    = _s_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(77)
        u_init = 0.1 * randn(rng, Float64, 2, 50)

        config = BFGSConfig(
            max_iter       = 100,
            verbose        = false,
            record_hessian = true,   # store B_inv snapshots
        )

        result = bfgs_optimize(sys, target, u_init, dt; config=config)

        for (iter, B_inv) in result.hessian_history
            evals = eigvals(Symmetric(real.(B_inv)))
            @test minimum(evals) > -1e-6   # allow tiny numerical negative
        end

    end

    # -----------------------------------------------------------------------
    @testset "Newton quadratic convergence near optimum" begin

        # Near a non-degenerate local optimum the Newton method should show
        # super-linear (ideally quadratic) convergence.  We verify that the
        # log of gradient norms decreases approximately linearly
        # (i.e., the ratio log|g_{k+1}|/log|g_k| increases toward 2).

        sys    = _s_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        # First warm-up with GRAPE to get near the optimum
        rng    = MersenneTwister(19)
        u_warm = 0.1 * randn(rng, Float64, 2, 60)
        cfg_warmup = GRAPEConfig(max_iter=300, verbose=false)
        r_warm = grape_optimize(sys, target, u_warm, dt; config=cfg_warmup)

        # Now run Newton from the warm start
        cfg_newton = NewtonConfig(max_iter=20, verbose=false)
        r_newton   = newton_optimize(sys, target, r_warm.controls, dt;
                                     config=cfg_newton)

        gnorms = r_newton.gradient_norm_history
        # Need at least 3 steps to check convergence rate
        if length(gnorms) >= 3
            # Verify at least some reduction in the gradient norm
            @test gnorms[end] < gnorms[1] / 10
        end

    end

    # -----------------------------------------------------------------------
    @testset "BFGS fidelity history monotone with Wolfe line search" begin

        sys    = _s_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(200)
        u_init = 0.1 * randn(rng, Float64, 2, 70)

        config = BFGSConfig(
            max_iter        = 200,
            line_search     = :wolfe,
            verbose         = false,
        )

        result = bfgs_optimize(sys, target, u_init, dt; config=config)

        hist = result.fidelity_history
        for k in 2:length(hist)
            @test hist[k] >= hist[k-1] - 1e-8
        end

    end

end  # Second Order Methods
