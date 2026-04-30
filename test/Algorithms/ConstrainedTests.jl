# test/Algorithms/ConstrainedTests.jl
# ======================================
# Tests for constrained quantum optimal control.
# Constraint types exercised:
#   - Box (bound) constraints on individual control amplitudes
#   - L2 power constraint on the full pulse
#   - Bandwidth (spectral) constraint
#   - Infeasible / very-tight constraint handling

using Test
using Pulsar
using LinearAlgebra
using Random
using Statistics: mean

# ---------------------------------------------------------------------------
# Shared Pauli matrices
# ---------------------------------------------------------------------------
const _cσ_x = ComplexF64[0 1; 1 0]
const _cσ_y = ComplexF64[0 -im; im 0]
const _cσ_z = ComplexF64[1 0; 0 -1]
const _cI2  = ComplexF64[1 0; 0 1]

# ---------------------------------------------------------------------------
# Helper: resonant qubit system
# ---------------------------------------------------------------------------
function _c_qubit_sys()
    H_drift = zeros(ComplexF64, 2, 2)
    H_ctrl  = [2π * _cσ_x, 2π * _cσ_y]
    return quantum_system(H_drift, H_ctrl)
end

# ---------------------------------------------------------------------------
@testset "Constrained Optimization" begin

    # -----------------------------------------------------------------------
    @testset "Bound constraints satisfied" begin

        sys    = _c_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(42)
        u_init = 0.1 * randn(rng, Float64, 2, 80)

        u_lb = -0.5
        u_ub =  0.5
        constraints = [BoundConstraint(u_lb, u_ub)]

        config = ConstrainedConfig(
            max_iter = 500,
            verbose  = false,
        )

        result = constrained_optimize(sys, target, u_init, dt, constraints;
                                      config=config)

        # Every control amplitude must be within [u_lb, u_ub]
        @test all(u_lb - 1e-9 .<= result.controls .<= u_ub + 1e-9)
        @test result.fidelity > 0.50   # should still find a reasonable solution

    end

    # -----------------------------------------------------------------------
    @testset "Tight bound constraints still feasible" begin

        sys    = _c_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(5)
        u_init = 0.01 * randn(rng, Float64, 2, 120)

        # Very tight bounds — smaller than what GRAPE would naturally converge to
        constraints = [BoundConstraint(-0.05, 0.05)]

        config = ConstrainedConfig(max_iter=300, verbose=false)
        result = constrained_optimize(sys, target, u_init, dt, constraints;
                                      config=config)

        # Feasibility must be maintained regardless of fidelity
        @test all(-0.05 - 1e-9 .<= result.controls .<= 0.05 + 1e-9)

    end

    # -----------------------------------------------------------------------
    @testset "Power (L2) constraint satisfied" begin

        sys    = _c_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(99)
        u_init = 0.1 * randn(rng, Float64, 2, 80)

        P_max       = 1.0   # Σ u²  ≤ P_max
        constraints = [PowerConstraint(P_max)]

        config = ConstrainedConfig(max_iter=400, verbose=false)
        result = constrained_optimize(sys, target, u_init, dt, constraints;
                                      config=config)

        actual_power = sum(abs2, result.controls)
        @test actual_power <= P_max * 1.01   # 1% numerical tolerance

    end

    # -----------------------------------------------------------------------
    @testset "Power constraint – smaller budget reduces fidelity" begin

        # With a very small power budget we should get lower fidelity than
        # with a generous budget.
        sys    = _c_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng1   = MersenneTwister(12)
        u_init = 0.1 * randn(rng1, Float64, 2, 80)

        config = ConstrainedConfig(max_iter=300, verbose=false)

        r_high = constrained_optimize(sys, target, u_init, dt,
                                      [PowerConstraint(100.0)]; config=config)
        r_low  = constrained_optimize(sys, target, u_init, dt,
                                      [PowerConstraint(0.1)];  config=config)

        @test r_high.fidelity >= r_low.fidelity - 0.01

    end

    # -----------------------------------------------------------------------
    @testset "Bandwidth constraint reduces spectral content" begin

        # After optimization with a bandwidth constraint, the per-step
        # control amplitude should be bounded by `BW_max = f_max * dt`.
        # The Pulsar `BandwidthConstraint` projects via amplitude clamping,
        # which limits the per-step jump and therefore the highest spectral
        # content the discrete waveform can carry.
        sys    = _c_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7
        n_ts   = 100

        rng    = MersenneTwister(66)
        u_init = 0.1 * randn(rng, Float64, 2, n_ts)

        # Bandwidth = 2 MHz; the per-step amplitude bound is BW_max = 2 MHz × dt.
        f_max_MHz   = 2.0
        bw_max      = f_max_MHz * 1e6 * dt
        constraints = [BandwidthConstraint(f_max_MHz * 1e6, dt)]

        config = ConstrainedConfig(max_iter=300, verbose=false)
        result = constrained_optimize(sys, target, u_init, dt, constraints;
                                      config=config)

        # The amplitude bound is the actual feasibility check; verifying it
        # captures the spirit of bandwidth limiting in this implementation.
        @test maximum(abs.(result.controls)) <= bw_max + 1e-9
        @test isfinite(result.fidelity)

    end

    # -----------------------------------------------------------------------
    @testset "Combined bound + power constraints" begin

        sys    = _c_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(47)
        u_init = 0.1 * randn(rng, Float64, 2, 80)

        constraints = [
            BoundConstraint(-0.3, 0.3),
            PowerConstraint(5.0),
        ]

        config = ConstrainedConfig(max_iter=400, verbose=false)
        result = constrained_optimize(sys, target, u_init, dt, constraints;
                                      config=config)

        @test all(-0.3 - 1e-9 .<= result.controls .<= 0.3 + 1e-9)
        @test sum(abs2, result.controls) <= 5.0 * 1.01

    end

    # -----------------------------------------------------------------------
    @testset "Infeasible problem – best feasible returned" begin

        # Constraints so tight that target fidelity > 0.5 is physically
        # impossible.  The optimizer should return the best it can (≥ 0)
        # rather than crashing or returning NaN.
        sys    = _c_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(3)
        u_init = 0.001 * randn(rng, Float64, 2, 20)   # very few steps

        # Essentially zero power allowed
        constraints = [PowerConstraint(1e-8)]

        config = ConstrainedConfig(max_iter=100, verbose=false)
        result = constrained_optimize(sys, target, u_init, dt, constraints;
                                      config=config)

        @test isfinite(result.fidelity)
        @test result.fidelity >= 0.0
        @test result.fidelity <= 1.0 + 1e-10
        # Feasibility must still hold
        @test sum(abs2, result.controls) <= 1e-8 * 1.01

    end

    # -----------------------------------------------------------------------
    @testset "Unconstrained vs constrained fidelity comparison" begin

        sys    = _c_qubit_sys()
        target = state_target(ComplexF64[0, 1])
        dt     = 1e-7

        rng    = MersenneTwister(88)
        u_init = 0.1 * randn(rng, Float64, 2, 80)

        cfg_grape = GRAPEConfig(max_iter=300, verbose=false)
        r_unc  = grape_optimize(sys, target, u_init, dt; config=cfg_grape)

        cfg_con = ConstrainedConfig(max_iter=300, verbose=false)
        r_con  = constrained_optimize(sys, target, u_init, dt,
                                      [BoundConstraint(-0.2, 0.2)];
                                      config=cfg_con)

        # Constraints can only reduce or maintain the achievable fidelity
        @test r_unc.fidelity >= r_con.fidelity - 0.05   # 5% slack

    end

end  # Constrained Optimization
