# test/Algorithms/KrotovTheme7Tests.jl
# ====================================
# Theme 7 — Krotov upgrades:
#   * `update_shapes` — per-control update-shape vector (Krotov.jl / QuTiP)
#   * `chi_constructor` — pluggable boundary seeding for χ(T)
#   * `σ_adaptive` — second-order σ doubles with λ_a on rollback
#
# Each new kwarg is exercised on the canonical 1-qubit |0⟩→|1⟩ state-transfer
# fixture used by the baseline KrotovTests.jl.

using Test
using PULSAR
using LinearAlgebra
using Random

const _T7σX = ComplexF64[0 1; 1 0]
const _T7σY = ComplexF64[0 -im; im 0]

function _qubit_system_t7(; init = ComplexF64[1, 0])
    H_drift = zeros(ComplexF64, 2, 2)
    H_ctrl  = [_T7σX / 2, _T7σY / 2]
    return QuantumSystem(H_drift, H_ctrl, 2, 2,
                         Dict{String,Any}("init_state" => ComplexF64.(init)))
end

@testset "Theme 7 — Krotov upgrades" begin

    @testset "update_shapes — single Function broadcasts to all controls" begin
        sys  = _qubit_system_t7()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 40
        amps = 0.1 .* randn(MersenneTwister(101), N, 2)
        ctrl = ControlSequence(amps, dt, N)

        # Identical-ones shape vs. shape=nothing must match (same default 1.0).
        _, F_default, _ = krotov_optimize(sys, tgt, ctrl;
                                           λ_a = 5.0, max_iter = 100,
                                           tol = 1e-12, verbose = false)
        _, F_broadcast, _ = krotov_optimize(sys, tgt, ctrl;
                                             λ_a = 5.0, max_iter = 100,
                                             tol = 1e-12, verbose = false,
                                             update_shapes = (k, N) -> 1.0)
        @test F_default ≈ F_broadcast atol = 1e-10
    end

    @testset "update_shapes — per-control vector" begin
        sys  = _qubit_system_t7()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 40
        amps = 0.1 .* randn(MersenneTwister(102), N, 2)
        ctrl = ControlSequence(amps, dt, N)

        # Different per-control shapes — first control on, second control off.
        # Effect: only the σx/2 control is updated; σy/2 stays at the guess.
        s_on  = (k, N) -> 1.0
        s_off = (k, N) -> 0.0
        ctrl_opt, F, st = krotov_optimize(sys, tgt, ctrl;
                                          λ_a = 1.0, max_iter = 200,
                                          tol = 1e-12, verbose = false,
                                          update_shapes = [s_on, s_off])
        # σy column unchanged by Krotov (zero shape):
        @test ctrl_opt.amplitudes[:, 2] ≈ amps[:, 2] atol = 1e-12
        # σx column did move:
        @test maximum(abs.(ctrl_opt.amplitudes[:, 1] .- amps[:, 1])) > 1e-3
        # Per-control vector path is exercised: history must be monotonic
        # ascending with σ-shape gating one control off.  Hard convergence is
        # not possible with σx alone from the symmetric |0⟩ initial state
        # (the linear real-overlap functional has near-zero imaginary
        # χ-overlap when only one quadrature control is active), so we only
        # require monotonic progress, not absolute convergence.
        @test all(diff(st.history) .>= -1e-12)
        @test F > st.history[1]
    end

    @testset "update_shapes — bad length errors cleanly" begin
        sys  = _qubit_system_t7()
        tgt  = state_target(ComplexF64[0, 1])
        ctrl = ControlSequence(0.1 .* randn(MersenneTwister(103), 20, 2), 0.05, 20)
        @test_throws ArgumentError krotov_optimize(sys, tgt, ctrl;
            λ_a = 5.0, max_iter = 5, verbose = false,
            update_shapes = [(k, N) -> 1.0])    # only 1 fn for 2 controls
    end

    @testset "chi_constructor — passes default boundary through" begin
        sys  = _qubit_system_t7()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 40
        amps = 0.1 .* randn(MersenneTwister(104), N, 2)
        ctrl = ControlSequence(amps, dt, N)

        _, F_def, _ = krotov_optimize(sys, tgt, ctrl;
                                       λ_a = 5.0, max_iter = 100,
                                       tol = 1e-12, verbose = false)

        # An identity chi_constructor must reproduce the default behaviour.
        identity_chi = (target, _) -> target.target_state
        _, F_id, _ = krotov_optimize(sys, tgt, ctrl;
                                      λ_a = 5.0, max_iter = 100,
                                      tol = 1e-12, verbose = false,
                                      chi_constructor = identity_chi)
        @test F_def ≈ F_id atol = 1e-12
    end

    @testset "chi_constructor — bad return errors cleanly" begin
        sys  = _qubit_system_t7()
        tgt  = state_target(ComplexF64[0, 1])
        ctrl = ControlSequence(0.1 .* randn(MersenneTwister(105), 20, 2), 0.05, 20)
        bad_chi = (target, _) -> ComplexF64[1, 0, 0]    # wrong dim (3 vs 2)
        @test_throws ArgumentError krotov_optimize(sys, tgt, ctrl;
            λ_a = 5.0, max_iter = 1, verbose = false,
            chi_constructor = bad_chi)
    end

    @testset "σ_adaptive — second-order Krotov still converges" begin
        sys  = _qubit_system_t7()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 40
        amps = 0.1 .* randn(MersenneTwister(106), N, 2)
        ctrl = ControlSequence(amps, dt, N)

        # σ_adaptive only changes behaviour when a rollback fires; on the
        # canonical fixture with λ_a=5 there are no rollbacks, so the
        # σ_adaptive=true and =false runs must agree.
        _, F_static,  _ = krotov_second_order_optimize(sys, tgt, ctrl;
                              λ_a = 5.0, σ = 0.1, σ_adaptive = false,
                              max_iter = 100, tol = 1e-12, verbose = false)
        _, F_adapt,   _ = krotov_second_order_optimize(sys, tgt, ctrl;
                              λ_a = 5.0, σ = 0.1, σ_adaptive = true,
                              max_iter = 100, tol = 1e-12, verbose = false)
        @test F_adapt ≈ F_static atol = 1e-6
        @test F_adapt > 0.999
    end

    @testset "σ_adaptive — agressive λ_a triggers rollback path" begin
        sys  = _qubit_system_t7()
        tgt  = state_target(ComplexF64[0, 1])
        dt, N = 0.05, 40
        amps = 0.5 .* randn(MersenneTwister(107), N, 2)
        ctrl = ControlSequence(amps, dt, N)

        # λ_a=0.1 is small enough that early steps overshoot — rollback
        # logic must keep the run monotonic and σ_adaptive must not break it.
        _, F, st = krotov_second_order_optimize(sys, tgt, ctrl;
                       λ_a = 0.1, σ = 0.1, σ_adaptive = true,
                       max_iter = 300, tol = 1e-12, verbose = false,
                       enforce_monotonic = true)
        @test all(diff(st.history) .>= -1e-10)
        @test F > 0.99
    end
end
