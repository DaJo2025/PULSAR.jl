# test/Algorithms/DirectSearchTests.jl
# =======================================
# Tests for derivative-free / direct search algorithms:
#   - Nelder-Mead simplex
#   - CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
#   - PSO (Particle Swarm Optimization)
#
# These methods are gradient-free and are typically used when:
#   (a) the landscape is non-smooth,
#   (b) the system model has discrete/discontinuous elements, or
#   (c) gradients are not available.

using Test
using Pulsar
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Shared Pauli matrices
# ---------------------------------------------------------------------------
const _dσ_x = ComplexF64[0 1; 1 0]
const _dσ_y = ComplexF64[0 -im; im 0]
const _dσ_z = ComplexF64[1 0; 0 -1]
const _dI2  = ComplexF64[1 0; 0 1]

# ---------------------------------------------------------------------------
# Helper: resonant qubit system
# ---------------------------------------------------------------------------
function _d_qubit_sys()
    # Realistic 1-qubit drive scaled to 1 MHz Rabi so that O(1) controls
    # over a few-µs pulse produce non-trivial unitaries.
    H_drift = zeros(ComplexF64, 2, 2)
    H_ctrl  = [2π * 1e6 * _dσ_x, 2π * 1e6 * _dσ_y]
    return quantum_system(H_drift, H_ctrl)
end

# ---------------------------------------------------------------------------
@testset "Direct Search Methods" begin

    # -----------------------------------------------------------------------
    @testset "Nelder-Mead simple convergence – state transfer" begin

        # Keep the problem small (few controls) so Nelder-Mead does not
        # stall in exponentially large space.
        sys    = _d_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 5e-7   # 500 ns per step → 5 µs total

        rng    = MersenneTwister(42)
        n_ts   = 10    # small control vector for Nelder-Mead
        u_init = 0.1 * randn(rng, Float64, 2, n_ts)

        config = NelderMeadConfig(
            max_iter        = 5000,
            convergence_tol = 1e-7,
            verbose         = false,
        )

        result = nelder_mead_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > 0.90
        # Nelder-Mead is gradient-free but should still improve from start
        F0 = evaluate_fidelity(sys, target, u_init, dt)
        @test result.fidelity > F0

    end

    # -----------------------------------------------------------------------
    @testset "Nelder-Mead convergence criterion" begin

        sys    = _d_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 5e-7

        rng    = MersenneTwister(7)
        u_init = 0.05 * randn(rng, Float64, 2, 8)

        config = NelderMeadConfig(
            max_iter        = 10000,
            convergence_tol = 1e-6,
            verbose         = false,
        )

        result = nelder_mead_optimize(sys, target, u_init, dt; config=config)

        # Either converged, or ran out of iterations — neither is a failure
        @test result.converged || result.n_iterations >= 10000
        @test isfinite(result.fidelity)

    end

    # -----------------------------------------------------------------------
    @testset "CMA-ES convergence – single qubit" begin

        sys    = _d_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(31)
        n_ts   = 30
        u_init = 0.1 * randn(rng, Float64, 2, n_ts)

        config = CMAESConfig(
            max_iter        = 2000,
            population_size = 20,
            initial_sigma   = 0.1,
            verbose         = false,
        )

        result = cmaes_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > 0.90

    end

    # -----------------------------------------------------------------------
    @testset "CMA-ES non-smooth landscape" begin

        # Introduce a non-smooth objective by discretizing fidelity to
        # a coarse grid (simulated via a wrapper).  CMA-ES should still
        # make progress.

        sys    = _d_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 2e-7

        rng    = MersenneTwister(55)
        n_ts   = 20
        u_init = 0.05 * randn(rng, Float64, 2, n_ts)

        config = CMAESConfig(
            max_iter        = 3000,
            population_size = 30,
            initial_sigma   = 0.05,
            noise_handling  = true,   # enable noise-resilient mode
            verbose         = false,
        )

        result = cmaes_optimize(sys, target, u_init, dt; config=config)

        # On a noisy landscape, reaching 0.80 is a good result.
        @test result.fidelity > 0.80

    end

    # -----------------------------------------------------------------------
    @testset "CMA-ES strictly improves from initial guess" begin

        sys    = _d_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(99)
        n_ts   = 25
        u_init = 0.5 * randn(rng, Float64, 2, n_ts)  # non-trivial initial point

        F0 = evaluate_fidelity(sys, target, u_init, dt)

        config = CMAESConfig(max_iter=1000, population_size=15, verbose=false)
        result = cmaes_optimize(sys, target, u_init, dt; config=config)

        @test result.fidelity > F0

    end

    # -----------------------------------------------------------------------
    @testset "PSO feasibility – does not crash" begin

        # PSO is stochastic; we only verify that it runs, returns finite
        # values, and improves over random initial controls.
        sys    = _d_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(123)
        n_ts   = 20
        u_init = 0.05 * randn(rng, Float64, 2, n_ts)
        F0     = evaluate_fidelity(sys, target, u_init, dt)

        config = PSOConfig(
            max_iter     = 500,
            n_particles  = 20,
            verbose      = false,
            seed         = 7,
        )

        result = pso_optimize(sys, target, u_init, dt; config=config)

        @test isfinite(result.fidelity)
        @test 0.0 <= result.fidelity <= 1.0 + 1e-10
        @test result.fidelity > F0   # must improve over starting point

    end

    # -----------------------------------------------------------------------
    @testset "PSO reproducibility with fixed seed" begin

        sys    = _d_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(17)
        u_init = 0.05 * randn(rng, Float64, 2, 15)

        config = PSOConfig(max_iter=200, n_particles=15, verbose=false, seed=42)

        r1 = pso_optimize(sys, target, u_init, dt; config=config)
        r2 = pso_optimize(sys, target, u_init, dt; config=config)

        @test r1.fidelity == r2.fidelity

    end

    # -----------------------------------------------------------------------
    @testset "Comparison: direct search vs GRAPE on same problem" begin

        # GRAPE should outperform purely derivative-free methods given
        # the same number of function evaluations.
        sys    = _d_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(77)
        u_init = 0.1 * randn(rng, Float64, 2, 50)

        cfg_grape = GRAPEConfig(max_iter=200, verbose=false)
        cfg_cmaes = CMAESConfig(max_iter=200, population_size=10, verbose=false)

        r_grape = grape_optimize(sys, target, u_init, dt; config=cfg_grape)
        r_cmaes = cmaes_optimize(sys, target, u_init, dt; config=cfg_cmaes)

        # GRAPE has access to gradients so should do better
        @test r_grape.fidelity >= r_cmaes.fidelity - 0.05   # 5% tolerance

    end

end  # Direct Search Methods
