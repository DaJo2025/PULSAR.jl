# test/Computation/PropagatorRegistryTests.jl
# ============================================
# Theme 1 — exercises the AbstractPropagator dispatch hierarchy.
#
#   • EigenPropagator must agree with the legacy 2-arg compute_propagator.
#   • PadePropagator must agree to ~1e-12 on a Hermitian H (where eigen
#     is exact) and remain unitary.
#   • Reserved backends (Chebyshev / Newton / Magnus) must throw a clear
#     ErrorException — they are skeletons, not silent no-ops.

using Test
using PULSAR
using LinearAlgebra
using Random

@testset "PropagatorRegistry (Theme 1)" begin

    rng = MersenneTwister(20260424)

    # ── Build a small Hermitian H ────────────────────────────────────────
    A = randn(rng, ComplexF64, 4, 4)
    H = Matrix{ComplexF64}((A + A') / 2)        # Hermitian
    dt = 0.123

    @testset "EigenPropagator delegates to legacy 2-arg form" begin
        U_eig    = compute_propagator(H, dt, EigenPropagator())
        U_legacy = compute_propagator(H, dt)
        @test U_eig == U_legacy                 # bit-exact (same code path)
    end

    @testset "PadePropagator agrees with eigen on Hermitian" begin
        U_legacy = compute_propagator(H, dt)
        U_pade   = compute_propagator(H, dt, PadePropagator())
        @test maximum(abs.(U_pade .- U_legacy)) < 1e-10
        @test norm(U_pade * U_pade' - I) < 1e-10
    end

    @testset "PadePropagator handles dt = 0" begin
        U = compute_propagator(H, 0.0, PadePropagator())
        @test U == Matrix{ComplexF64}(I, size(H)...)
    end

    @testset "Reserved backends throw guarded errors" begin
        @test_throws ErrorException compute_propagator(
            H, dt, ChebyshevPropagator(20, (-2.0, 2.0)))
        @test_throws ErrorException compute_propagator(
            H, dt, NewtonPropagator(20))
        @test_throws ErrorException compute_propagator(
            H, dt, MagnusPropagator(4))
    end

    @testset "Constructor validation" begin
        @test_throws ArgumentError ChebyshevPropagator(0, (-1.0, 1.0))
        @test_throws ArgumentError ChebyshevPropagator(20, (1.0, -1.0))
        @test_throws ArgumentError NewtonPropagator(0)
        @test_throws ArgumentError MagnusPropagator(3)   # only 2/4/6 allowed
    end
end
