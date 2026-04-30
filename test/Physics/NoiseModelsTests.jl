# test/Physics/NoiseModelsTests.jl
# ================================
# Theme 5 — exercises the AbstractNoiseModel hierarchy.
#
#   • Each concrete type produces the right number of samples.
#   • Quadrature weights sum to 1 (normalised).
#   • CompositeNoise is the Cartesian product of its components.
#   • Constructor validation rejects malformed inputs.

using Test
using Pulsar
using LinearAlgebra
using Random

@testset "AbstractNoiseModel (Theme 5)" begin

    rng = MersenneTwister(20260424)
    σz  = ComplexF64[1 0; 0 -1]
    σm  = ComplexF64[0 0; 1 0]

    # ── ParametricDrift ─────────────────────────────────────────────────
    @testset "ParametricDrift" begin
        delta_freq = (s) -> (2π * s) .* σz / 2
        nm = ParametricDrift(delta_freq, :gaussian, 100e3; n_samples = 5)
        @test n_samples(nm) == 5
        samples = sample_ensemble(nm; rng = rng)
        @test length(samples) == 5
        @test all(s -> s.delta_drift !== nothing, samples)
        @test sum(s.weight for s in samples) ≈ 1.0

        # Explicit samples override distribution
        nm2 = ParametricDrift(delta_freq, :custom, 0.0;
                                n_samples = 3,
                                samples   = [-1.0, 0.0, 1.0])
        @test n_samples(nm2) == 3
        samples2 = sample_ensemble(nm2; rng = rng)
        @test length(samples2) == 3

        @test_throws ArgumentError ParametricDrift(delta_freq, :weird, 1.0)
        @test_throws ArgumentError ParametricDrift(delta_freq, :gaussian, 1.0;
                                                     n_samples = 0)
    end

    # ── PowderOrientation ───────────────────────────────────────────────
    @testset "PowderOrientation" begin
        nm = PowderOrientation([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)], [1.0, 3.0])
        s  = sample_ensemble(nm)
        @test length(s) == 2
        @test s[1].weight ≈ 0.25
        @test s[2].weight ≈ 0.75
        @test all(x -> x.euler !== nothing, s)
        @test_throws ArgumentError PowderOrientation([], Float64[])
        @test_throws ArgumentError PowderOrientation([(0.0,0.0,0.0)], [1.0, 2.0])
        @test_throws ArgumentError PowderOrientation([(0.0,0.0,0.0)], [0.0])
    end

    # ── DriveCalibration ────────────────────────────────────────────────
    @testset "DriveCalibration{:exact}" begin
        nm = DriveCalibration([0.95, 1.0, 1.05])
        s  = sample_ensemble(nm)
        @test length(s) == 3
        @test all(x -> x.drive_factors !== nothing, s)
        @test s[2].drive_factors[1] == 1.0
        @test sum(x.weight for x in s) ≈ 1.0
        @test_throws ArgumentError DriveCalibration(Float64[])
        @test_throws ArgumentError DriveCalibration([1.0]; distribution = :weird)
    end

    @testset "DriveCalibration{:gaussian} requires explicit draw" begin
        nm = DriveCalibration([1.0]; distribution = :gaussian)
        @test_throws ErrorException sample_ensemble(nm)
    end

    # ── MarkovianDissipation ───────────────────────────────────────────
    @testset "MarkovianDissipation" begin
        nm = MarkovianDissipation([σm], [1e3])
        s  = sample_ensemble(nm)
        @test length(s) == 1
        @test s[1].jump_ops !== nothing
        @test s[1].decay_rates == [1e3]
        @test_throws ArgumentError MarkovianDissipation([σm], [-1.0])
        @test_throws ArgumentError MarkovianDissipation([σm], [1e3, 1e2])
    end

    # ── ColoredNoiseSpectrum ───────────────────────────────────────────
    @testset "ColoredNoiseSpectrum" begin
        nm = ColoredNoiseSpectrum(ω -> 1.0/(1+ω^2), σz, [0.1, 1.0, 10.0])
        s  = sample_ensemble(nm)
        @test length(s) == 1
        @test s[1].weight == 1.0
        @test nm.psd_fn(1.0) ≈ 0.5
        @test_throws ArgumentError ColoredNoiseSpectrum(ω -> 1.0,
                                                          ComplexF64[1 2; 3 4; 5 6],
                                                          [1.0])
        @test_throws ArgumentError ColoredNoiseSpectrum(ω -> 1.0, σz, Float64[])
    end

    # ── CompositeNoise: Cartesian product ───────────────────────────────
    @testset "CompositeNoise (Cartesian product)" begin
        delta_freq = (s) -> (2π * s) .* σz / 2
        a = ParametricDrift(delta_freq, :gaussian, 100e3; n_samples = 4)
        b = PowderOrientation([(0.0,0.0,0.0), (1.0,2.0,3.0)], [1.0, 1.0])
        c = DriveCalibration([0.95, 1.0, 1.05])
        comp = CompositeNoise([a, b, c])
        @test n_samples(comp) == 4 * 2 * 3
        s = sample_ensemble(comp; rng = MersenneTwister(1))
        @test length(s) == 4 * 2 * 3
        @test isapprox(sum(x.weight for x in s), 1.0; atol = 1e-12)
        # Each merged sample carries fields from every component
        @test all(x -> x.delta_drift !== nothing &&
                       x.euler !== nothing &&
                       x.drive_factors !== nothing, s)
        @test_throws ArgumentError CompositeNoise(AbstractNoiseModel[])
    end
end
