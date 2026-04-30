# test/Physics/PulseCompositionTests.jl
# =====================================
# Theme 9 — hardware-aware pulse composition.
#
# Verifies that a `PulseComposition(prefix, suffix, dead)` wrapping an
# optimised segment is mathematically equivalent to running the optimiser on
# the un-composited problem with shifted boundary states.

using Test
using Pulsar
using LinearAlgebra

const _T9σX = ComplexF64[0 1; 1 0]
const _T9σY = ComplexF64[0 -im; im 0]
const _T9σZ = ComplexF64[1 0; 0 -1]

@testset "Theme 9 — hardware-aware pulse composition" begin

    @testset "compose_hard_pulse_propagator — single 90°_x pulse" begin
        # Single 90°_x on a qubit with operators [σx/2, σy/2] should rotate
        # |0⟩ → (|0⟩ − i|1⟩)/√2 (i.e. exp(−i π/2 · σx/2)).
        Ox = _T9σX / 2; Oy = _T9σY / 2
        segs = [CompositePulseSegment(90.0, 0.0)]
        U = compose_hard_pulse_propagator(segs, [Ox, Oy]; rf_hz = 1.0e3)
        ψ0 = ComplexF64[1, 0]
        ψ1 = U * ψ0
        ψref = ComplexF64[cos(π/4), -im * sin(π/4)]
        @test ψ1 ≈ ψref atol = 1e-10
    end

    @testset "compose_hard_pulse_propagator — BB1(180°) is unitary and ≈ X" begin
        # BB1(180°) is a broadband 180°-equivalent composite — applied to
        # |0⟩ it should land near |1⟩ up to a global phase.
        Ox = _T9σX / 2; Oy = _T9σY / 2
        segs = bb1(180.0)
        U = compose_hard_pulse_propagator(segs, [Ox, Oy]; rf_hz = 1.0e3)
        @test U * adjoint(U) ≈ Matrix{ComplexF64}(I, 2, 2) atol = 1e-10
        ψ1 = U * ComplexF64[1, 0]
        @test abs2(dot(ComplexF64[0, 1], ψ1)) ≈ 1.0 atol = 1e-8
    end

    @testset "dead_time_propagator — H_drift = ω σz/2" begin
        ω = 2π * 1000.0
        H = ω .* _T9σZ ./ 2
        t = 250e-6                                # quarter-period free precession
        U = dead_time_propagator(H, t)
        @test U * adjoint(U) ≈ Matrix{ComplexF64}(I, 2, 2) atol = 1e-10
        # Expected: exp(−i ω t / 2) on |0⟩, exp(+i ω t / 2) on |1⟩
        @test U[1, 1] ≈ exp(-im * ω * t / 2) atol = 1e-10
        @test U[2, 2] ≈ exp( im * ω * t / 2) atol = 1e-10
        @test U[1, 2] ≈ 0.0 atol = 1e-12
    end

    @testset "dead_time_propagator — t = 0 returns identity" begin
        H = randn(ComplexF64, 3, 3); H = (H + H') / 2
        @test dead_time_propagator(H, 0.0) ≈ Matrix{ComplexF64}(I, 3, 3) atol = 1e-12
        @test_throws ArgumentError dead_time_propagator(H, -1e-6)
    end

    @testset "PulseComposition — empty composition is identity-like" begin
        comp = PulseComposition()                  # all nothing
        ψi = ComplexF64[1, 0]
        ψt = ComplexF64[0, 1]
        ψi_eff, ψt_eff = compose_effective_boundary(comp, ψi, ψt)
        @test ψi_eff == ψi                         # passthrough (no rotation)
        @test ψt_eff == ψt
    end

    @testset "compose_effective_boundary — boundary-shift identity (state)" begin
        # The boundary-shift identity: for any unitary U_pre, U_post and any
        # "middle" unitary U_mid,
        #   ⟨ψ_t | U_post · U_mid · U_pre | ψ_i⟩
        #     = ⟨U_post† ψ_t | U_mid | U_pre ψ_i⟩
        ψi = normalize!(randn(ComplexF64, 2))
        ψt = normalize!(randn(ComplexF64, 2))
        U_pre  = exp(-im * 0.37 .* _T9σX)
        U_dead = exp(-im * 0.21 .* _T9σZ)
        U_suf  = exp(-im * 0.53 .* _T9σY)
        U_mid  = exp(-im * 1.10 .* _T9σX)

        comp = PulseComposition(prefix = U_pre, suffix = U_suf, dead = U_dead)
        ψi_eff, ψt_eff = compose_effective_boundary(comp, ψi, ψt)

        lhs = dot(ψt, U_suf * U_dead * U_mid * U_pre * ψi)
        rhs = dot(ψt_eff, U_mid * ψi_eff)
        @test lhs ≈ rhs atol = 1e-12

        # Squared-overlap fidelity invariance:
        @test abs2(lhs) ≈ abs2(rhs) atol = 1e-12
    end

    @testset "compose_effective_boundary — density-matrix dispatch" begin
        # For ρ = |ψ⟩⟨ψ|, U_pre · ρ · U_pre† = (U_pre ψ)(U_pre ψ)†.
        ψ  = normalize!(randn(ComplexF64, 2))
        ρ  = ψ * ψ'
        U_pre = exp(-im * 0.42 .* _T9σY)

        comp = PulseComposition(prefix = U_pre)
        ρi_eff, _ = compose_effective_boundary(comp, ρ, ρ)
        ψ_eff = U_pre * ψ
        @test ρi_eff ≈ ψ_eff * ψ_eff' atol = 1e-12
    end

    @testset "compose_effective_boundary — vector overload" begin
        ψi1 = ComplexF64[1, 0];  ψi2 = ComplexF64[0, 1]
        ψt1 = ComplexF64[0, 1];  ψt2 = ComplexF64[1, 0]
        U_pre = exp(-im * 0.31 .* _T9σX)
        comp  = PulseComposition(prefix = U_pre)

        outs_i, outs_t = compose_effective_boundary(comp,
                            [ψi1, ψi2], [ψt1, ψt2])
        @test outs_i[1] ≈ U_pre * ψi1 atol = 1e-12
        @test outs_i[2] ≈ U_pre * ψi2 atol = 1e-12
        @test outs_t[1] ≈ ψt1 atol = 1e-12      # no suffix → unchanged
        @test outs_t[2] ≈ ψt2 atol = 1e-12

        @test_throws ArgumentError compose_effective_boundary(
            comp, [ψi1], [ψt1, ψt2])
    end

    @testset "compose_hard_pulse_propagator — argument validation" begin
        Ox = _T9σX / 2; Oy = _T9σY / 2
        @test_throws ArgumentError compose_hard_pulse_propagator(
            CompositePulseSegment[], [Ox, Oy]; rf_hz = 1.0e3)
        @test_throws ArgumentError compose_hard_pulse_propagator(
            [CompositePulseSegment(90.0, 0.0)], [Ox, Oy]; rf_hz = -10.0)
        @test_throws ArgumentError compose_hard_pulse_propagator(
            [CompositePulseSegment(90.0, 0.0)], [Ox, Oy];
            rf_hz = 1.0e3, x_index = 5)
    end

    @testset "End-to-end equivalence vs explicit propagator chain" begin
        # Pick a random middle propagator U_mid and a random initial state ψ_i.
        # Verify that for ANY ψ_t, the boundary-shifted overlap matches the
        # full prefix · middle · suffix chain.  (Sweep over multiple targets.)
        U_pre  = exp(-im * 0.7 .* _T9σX)
        U_dead = exp(-im * 0.3 .* _T9σZ)
        U_suf  = exp(-im * 0.4 .* _T9σY)
        U_mid  = exp(-im * 0.9 .* (_T9σX + _T9σZ))
        comp   = PulseComposition(prefix = U_pre, suffix = U_suf, dead = U_dead)

        ψi = ComplexF64[1, 0]
        ψi_eff, _ = compose_effective_boundary(comp, ψi, ComplexF64[0, 1])

        for ψt in (ComplexF64[0, 1], ComplexF64[1, 1] ./ sqrt(2),
                   ComplexF64[1, im] ./ sqrt(2))
            _, ψt_eff = compose_effective_boundary(comp, ψi, ψt)
            full      = abs2(dot(ψt, U_suf * U_dead * U_mid * U_pre * ψi))
            shifted   = abs2(dot(ψt_eff, U_mid * ψi_eff))
            @test full ≈ shifted atol = 1e-12
        end
    end
end
