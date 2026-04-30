# test/Physics/PenaltyTests.jl
# =============================
# Unit tests for the AbstractPenalty subtypes added under Theme 3 of the
# QOC framework extension plan (Quandary / Spinach / qopt parity additions).
#
# Each penalty is exercised with:
#   1. A hand-computed reference value on a small known waveform.
#   2. A finite-difference cross-check against the analytical gradient.

using Test
using PULSAR
using Random

# ---------------------------------------------------------------------------
# Central-difference gradient — used to validate analytical gradients
# ---------------------------------------------------------------------------
function fd_gradient(pen::AbstractPenalty, w::Matrix{Float64}; h::Float64 = 1e-6)
    G = zeros(size(w))
    for idx in eachindex(w)
        w_p = copy(w); w_p[idx] += h
        w_m = copy(w); w_m[idx] -= h
        G[idx] = (pen(w_p) - pen(w_m)) / (2h)
    end
    return G
end

@testset "AbstractPenalty additions (Theme 3)" begin

    rng = MersenneTwister(20260424)

    # ── TotalEnergyBudget ─────────────────────────────────────────────────
    @testset "TotalEnergyBudget" begin
        # 1 ctrl × 4 ts, uniform unit dt: E = 1+4+9+16 = 30
        w = reshape(Float64[1.0, 2.0, 3.0, 4.0], 1, 4)

        # below-budget → zero penalty + zero gradient
        pen_under = TotalEnergyBudget(2.0, 100.0)
        @test pen_under(w) == 0.0
        @test all(iszero, gradient(pen_under, w))

        # over-budget → analytical formula matches by hand
        E_max = 25.0
        pen_over = TotalEnergyBudget(0.5, E_max)
        over = 30.0 - E_max          # 5
        @test pen_over(w) ≈ 0.5 * over^2
        # ∂P/∂w[j,k] = weight × 4 × over × w × dt;  dt = 1
        G_expected = 0.5 * 4.0 * over .* w
        @test gradient(pen_over, w) ≈ G_expected

        # vector dt
        dt = Float64[0.5, 0.5, 1.0, 2.0]
        E_v = sum(w[1, k]^2 * dt[k] for k in 1:4)        # 0.5+2+9+32 = 43.5
        pen_v = TotalEnergyBudget(1.0, 30.0; dt = dt)
        over_v = E_v - 30.0
        @test pen_v(w) ≈ over_v^2
        G_v = gradient(pen_v, w)
        for k in 1:4
            @test G_v[1, k] ≈ 4.0 * over_v * w[1, k] * dt[k]
        end

        # FD cross-check on a random waveform
        wr = randn(rng, 2, 6)
        pen_r = TotalEnergyBudget(0.7, 1.5)
        @test isapprox(gradient(pen_r, wr), fd_gradient(pen_r, wr); atol = 1e-6)
    end

    # ── MirrorSymmetryPenalty ─────────────────────────────────────────────
    @testset "MirrorSymmetryPenalty" begin
        # palindromic waveform → zero penalty
        w_sym = reshape(Float64[1, 2, 3, 2, 1], 1, 5)
        pen   = MirrorSymmetryPenalty(1.0)
        @test pen(w_sym) ≈ 0.0
        @test all(iszero, gradient(pen, w_sym))

        # known asymmetric waveform: w = [1,0,0,0,−1] (n_t=5, midpoint at 3)
        # canonical sum: Σ (w_k − w_{6−k})² = (1+1)² + 0² + 0² + 0² + (−1−1)² = 8
        w = reshape(Float64[1.0, 0, 0, 0, -1.0], 1, 5)
        @test pen(w) ≈ 8.0

        # FD cross-check: random 3-channel × 7-step waveform (odd → midpoint exercised)
        wr = randn(rng, 3, 7)
        pen_r = MirrorSymmetryPenalty(0.3)
        @test isapprox(gradient(pen_r, wr), fd_gradient(pen_r, wr); atol = 1e-6)

        # also even-length grid (no midpoint)
        we = randn(rng, 2, 8)
        @test isapprox(gradient(pen_r, we), fd_gradient(pen_r, we); atol = 1e-6)
    end

    # ── AsymmetryPenalty ──────────────────────────────────────────────────
    @testset "AsymmetryPenalty" begin
        # antisymmetric waveform → zero penalty
        w_anti = reshape(Float64[1, 2, 0, -2, -1], 1, 5)
        pen    = AsymmetryPenalty(1.0)
        @test pen(w_anti) ≈ 0.0
        @test all(iszero, gradient(pen, w_anti))

        # constant waveform on 5-step grid: w = [1,1,1,1,1]
        # canonical: Σ (w_k + w_{6−k})² = 5 × (1+1)² = 20
        w = ones(1, 5)
        @test pen(w) ≈ 20.0

        # midpoint behaviour: pure midpoint pulse w = [0,0,3,0,0]
        # canonical: only k=3 contributes (3+3)² = 36
        w_mid = reshape(Float64[0, 0, 3.0, 0, 0], 1, 5)
        @test pen(w_mid) ≈ 36.0

        # FD cross-check, odd + even
        for n_t in (7, 8)
            wr = randn(rng, 2, n_t)
            pen_r = AsymmetryPenalty(0.4)
            @test isapprox(gradient(pen_r, wr), fd_gradient(pen_r, wr); atol = 1e-6)
        end
    end

    # ── CrossCouplingPenalty ──────────────────────────────────────────────
    @testset "CrossCouplingPenalty" begin
        # single-channel pulse → zero (no pairs)
        w_one = reshape(Float64[1, 2, 3], 1, 3)
        pen   = CrossCouplingPenalty(1.0)
        @test pen(w_one) ≈ 0.0
        @test all(iszero, gradient(pen, w_one))

        # two-channel known case
        # w = [1 2 3; 4 5 6]
        # P = Σ_k (w[1,k] w[2,k])² = (1·4)² + (2·5)² + (3·6)² = 16+100+324 = 440
        w = Float64[1 2 3; 4 5 6]
        @test pen(w) ≈ 440.0

        # FD cross-check on random 4-channel waveform
        wr = randn(rng, 4, 5)
        pen_r = CrossCouplingPenalty(0.2)
        @test isapprox(gradient(pen_r, wr), fd_gradient(pen_r, wr); atol = 1e-6)
    end

    # ── InterpolatedTikhonov ──────────────────────────────────────────────
    @testset "InterpolatedTikhonov" begin
        w_ref = randn(rng, 2, 5)
        pen   = InterpolatedTikhonov(0.5, w_ref)

        # at reference → zero penalty
        @test pen(w_ref) ≈ 0.0
        @test all(iszero, gradient(pen, w_ref))

        # canonical: weight × Σ |w − w_ref|²
        w = w_ref .+ 0.1
        expected = 0.5 * 0.01 * length(w_ref)
        @test pen(w) ≈ expected

        # ∂/∂w = 2 weight (w − w_ref)
        @test gradient(pen, w) ≈ 2 * 0.5 .* (w .- w_ref)

        # FD cross-check
        wr = randn(rng, 2, 5)
        @test isapprox(gradient(pen, wr), fd_gradient(pen, wr); atol = 1e-7)

        # dimension mismatch should error
        @test_throws DimensionMismatch pen(zeros(3, 5))
        @test_throws DimensionMismatch gradient(pen, zeros(2, 6))
    end

    # ── value_and_gradient consistency ────────────────────────────────────
    @testset "value_and_gradient consistency" begin
        w = randn(rng, 2, 6)
        pens = AbstractPenalty[
            TotalEnergyBudget(1.0, 0.5),
            MirrorSymmetryPenalty(0.7),
            AsymmetryPenalty(0.3),
            CrossCouplingPenalty(0.2),
            InterpolatedTikhonov(0.5, randn(rng, 2, 6)),
        ]
        for p in pens
            v, g = value_and_gradient(p, w)
            @test v ≈ p(w)
            @test g ≈ gradient(p, w)
        end
    end

    # ── make_penalty_fns / make_penalty_grad_fns interop ─────────────────
    @testset "Helper closures interop" begin
        w = randn(rng, 2, 4)
        pens = AbstractPenalty[
            MirrorSymmetryPenalty(0.5),
            CrossCouplingPenalty(0.1),
        ]
        fns  = make_penalty_fns(pens)
        gfns = make_penalty_grad_fns(pens)
        @test length(fns) == length(pens)
        @test length(gfns) == length(pens)
        for (p, f, gf) in zip(pens, fns, gfns)
            @test f(w) ≈ p(w)
            @test gf(w) ≈ gradient(p, w)
        end
    end
end
