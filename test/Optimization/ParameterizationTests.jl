# test/Optimization/ParameterizationTests.jl
# ===========================================
# Theme 2 — exercises the AbstractControlParameterization hierarchy.
#
#   • PiecewiseConstant: forward = identity, inverse = identity, J = I.
#   • TanhParam / TanhSqParam / LogisticParam: bounds are respected, the
#     analytical Jacobian agrees with central FD on the diagonal, and
#     waveform → θ → waveform is a round-trip up to atanh-clamping.
#   • Reserved parameterisations (BSpline/Hermite/Fourier/Chebyshev/
#     Slepian/CRAB) throw ErrorException from each entry point.

using Test
using PULSAR
using LinearAlgebra
using Random

# Diagonal entry of analytical Jacobian vs central-difference at a single index
function _check_diagonal_jacobian(θ::Vector{Float64}, p, n_ctrl::Int, n_t::Int;
                                   h::Float64 = 1e-6, atol::Float64 = 1e-7)
    J = waveform_jacobian(θ, p, n_ctrl, n_t)
    J isa Diagonal || error("expected diagonal Jacobian for $(typeof(p))")
    for i in 1:length(θ)
        θp = copy(θ); θp[i] += h
        θm = copy(θ); θm[i] -= h
        fd = (vec(to_waveform(θp, p, n_ctrl, n_t)) .-
              vec(to_waveform(θm, p, n_ctrl, n_t))) ./ (2h)
        @test isapprox(fd[i], J[i, i]; atol = atol)
        # Off-diagonal entries should be (numerically) zero
        @test all(abs.(fd[1:end .!= i]) .< atol)
    end
end

@testset "AbstractControlParameterization (Theme 2)" begin

    rng = MersenneTwister(20260424)
    n_ctrl, n_t = 2, 6
    θ = randn(rng, n_ctrl * n_t)

    # ── PiecewiseConstant ────────────────────────────────────────────────
    @testset "PiecewiseConstant" begin
        p = PiecewiseConstant()
        w = to_waveform(θ, p, n_ctrl, n_t)
        @test vec(w) == θ
        @test from_waveform(w, p) == θ
        @test waveform_jacobian(θ, p, n_ctrl, n_t) ==
              Diagonal(ones(length(θ)))
        @test_throws DimensionMismatch to_waveform(θ[1:end-1], p, n_ctrl, n_t)
    end

    # ── TanhParam ────────────────────────────────────────────────────────
    @testset "TanhParam" begin
        u_max = 2.5
        p = TanhParam(u_max)
        w = to_waveform(θ, p, n_ctrl, n_t)
        @test all(abs.(w) .<= u_max + 1e-12)
        @test from_waveform(w, p) ≈ θ atol = 1e-9
        _check_diagonal_jacobian(θ, p, n_ctrl, n_t)
        @test_throws ArgumentError TanhParam(0)
        @test_throws ArgumentError TanhParam(-1.0)
    end

    # ── TanhSqParam ──────────────────────────────────────────────────────
    @testset "TanhSqParam" begin
        u_max = 1.7
        p = TanhSqParam(u_max)
        w = to_waveform(θ, p, n_ctrl, n_t)
        @test all(0 .<= w .<= u_max + 1e-12)
        # tanh² is even — inverse maps the magnitude only.
        @test from_waveform(w, p) ≈ abs.(θ) atol = 1e-8
        _check_diagonal_jacobian(θ, p, n_ctrl, n_t)
        @test_throws ArgumentError TanhSqParam(0)
    end

    # ── LogisticParam ────────────────────────────────────────────────────
    @testset "LogisticParam (β = 2 ⇒ TanhParam algebra)" begin
        u_max = 3.0
        p_log  = LogisticParam(u_max; beta = 2.0)
        p_tanh = TanhParam(u_max)
        w_log  = to_waveform(θ, p_log,  n_ctrl, n_t)
        w_tanh = to_waveform(θ, p_tanh, n_ctrl, n_t)
        @test w_log ≈ w_tanh atol = 1e-12
        @test from_waveform(w_log, p_log) ≈ θ atol = 1e-9
        _check_diagonal_jacobian(θ, p_log, n_ctrl, n_t)
        @test_throws ArgumentError LogisticParam(1.0; beta = 0.0)
        @test_throws ArgumentError LogisticParam(-1.0)
    end

    # ── Reserved parameterisations: must throw ───────────────────────────
    @testset "Reserved parameterisations are guarded" begin
        reserved = [
            BSplineParam(4),
            HermiteParam(5),
            FourierParam(3),
            ChebyshevParam(8),
            SlepianParam(4, 0.1),
            CRABRandomParam(:fourier, 3),
        ]
        for r in reserved
            @test_throws ErrorException to_waveform(θ, r, n_ctrl, n_t)
            @test_throws ErrorException from_waveform(zeros(n_ctrl, n_t), r)
            @test_throws ErrorException waveform_jacobian(θ, r, n_ctrl, n_t)
        end
        @test_throws ArgumentError BSplineParam(0)
        @test_throws ArgumentError BSplineParam(4, 5)             # unsupported order
        @test_throws ArgumentError CRABRandomParam(:weird, 3)
    end

    # ── PhaseOnlyParam ───────────────────────────────────────────────────
    @testset "PhaseOnlyParam" begin
        rng_p = MersenneTwister(20260429)

        @testset "constructor validation" begin
            @test_throws ArgumentError PhaseOnlyParam(0.0, [(1, 2)])
            @test_throws ArgumentError PhaseOnlyParam(-1.0, [(1, 2)])
            @test_throws ArgumentError PhaseOnlyParam(1.0, Tuple{Int,Int}[])
            @test_throws ArgumentError PhaseOnlyParam(1.0, [(1, 1)])     # cx == cy
            @test_throws ArgumentError PhaseOnlyParam(1.0, [(0, 1)])     # idx < 1
            @test_throws ArgumentError PhaseOnlyParam(1.0, [(1, 2), (2, 3)])  # reuse
        end

        @testset "single-pair forward / round-trip" begin
            A = 1.5
            p = PhaseOnlyParam(A, [(1, 2)])
            n_ctrl_pp, n_t_pp = 2, 8
            n_p = 1; n_free = 0
            θ_pp = randn(rng_p, (n_p + n_free) * n_t_pp)
            w = to_waveform(θ_pp, p, n_ctrl_pp, n_t_pp)
            @test size(w) == (n_ctrl_pp, n_t_pp)
            # Constant amplitude: √(Cx²+Cy²) = A everywhere
            @test all(isapprox.(sqrt.(w[1, :].^2 .+ w[2, :].^2), A; atol = 1e-12))
            # Round-trip: from_waveform recovers the phase mod 2π
            θ_back = from_waveform(w, p)
            @test all(isapprox.(mod.(θ_back .- θ_pp .+ π, 2π) .- π, 0.0; atol = 1e-10))
            # Hand-computed reference at φ = π/3
            θ_one = [π/3]
            w_one = to_waveform(θ_one, p, 2, 1)
            @test isapprox(w_one[1, 1], A * cos(π/3); atol = 1e-12)
            @test isapprox(w_one[2, 1], A * sin(π/3); atol = 1e-12)
        end

        @testset "multi-pair (heteronuclear)" begin
            A = 2.0
            p = PhaseOnlyParam(A, [(1, 2), (3, 4)])
            n_ctrl_pp, n_t_pp = 4, 5
            n_p = 2; n_free = 0
            θ_pp = randn(rng_p, (n_p + n_free) * n_t_pp)
            w = to_waveform(θ_pp, p, n_ctrl_pp, n_t_pp)
            @test all(isapprox.(sqrt.(w[1, :].^2 .+ w[2, :].^2), A; atol = 1e-12))
            @test all(isapprox.(sqrt.(w[3, :].^2 .+ w[4, :].^2), A; atol = 1e-12))
            # Phases must round-trip independently
            θ_back = from_waveform(w, p)
            @test all(isapprox.(mod.(θ_back .- θ_pp .+ π, 2π) .- π, 0.0; atol = 1e-10))
        end

        @testset "mixed: paired + free channels" begin
            A = 0.7
            p = PhaseOnlyParam(A, [(1, 2)])
            n_ctrl_pp, n_t_pp = 5, 6
            n_p = 1; n_free = 3   # channels 3, 4, 5 are free Cartesian
            θ_pp = randn(rng_p, (n_p + n_free) * n_t_pp)
            w = to_waveform(θ_pp, p, n_ctrl_pp, n_t_pp)
            # Pair (1,2) has constant amplitude
            @test all(isapprox.(sqrt.(w[1, :].^2 .+ w[2, :].^2), A; atol = 1e-12))
            # Free channels 3,4,5 survive verbatim through round-trip
            θ_back = from_waveform(w, p)
            Θ_back = reshape(θ_back, n_p + n_free, n_t_pp)
            Θ_pp   = reshape(θ_pp,   n_p + n_free, n_t_pp)
            @test isapprox(Θ_back[2, :], Θ_pp[2, :]; atol = 1e-12)
            @test isapprox(Θ_back[3, :], Θ_pp[3, :]; atol = 1e-12)
            @test isapprox(Θ_back[4, :], Θ_pp[4, :]; atol = 1e-12)
        end

        @testset "apply_jacobian_transpose! FD agreement" begin
            A = 1.2
            p = PhaseOnlyParam(A, [(1, 2)])
            n_ctrl_pp, n_t_pp = 2, 4
            n_p = 1; n_free = 0
            n_θ = (n_p + n_free) * n_t_pp
            θ_pp = randn(rng_p, n_θ)

            # Random scalar objective f(θ) = sum(W .* w(θ))
            W_obj = randn(rng_p, n_ctrl_pp, n_t_pp)
            f(θ_) = sum(W_obj .* to_waveform(θ_, p, n_ctrl_pp, n_t_pp))

            # Analytical: g_w = W_obj (since ∂f/∂w[i,j] = W_obj[i,j])
            g_w = copy(W_obj)
            g_θ_an = zeros(n_θ)
            apply_jacobian_transpose!(g_θ_an, g_w, θ_pp, p, n_ctrl_pp, n_t_pp)

            # Central FD vs analytical
            h = 1e-7
            for i in 1:n_θ
                θp = copy(θ_pp); θp[i] += h
                θm = copy(θ_pp); θm[i] -= h
                fd = (f(θp) - f(θm)) / (2h)
                @test isapprox(fd, g_θ_an[i]; atol = 1e-6)
            end
        end

        @testset "apply_jacobian_transpose! agrees with explicit Jacobian" begin
            A = 0.9
            p = PhaseOnlyParam(A, [(1, 2), (3, 4)])
            n_ctrl_pp, n_t_pp = 4, 3
            n_p = 2; n_free = 0
            θ_pp = randn(rng_p, (n_p + n_free) * n_t_pp)
            g_w  = randn(rng_p, n_ctrl_pp, n_t_pp)

            J = waveform_jacobian(θ_pp, p, n_ctrl_pp, n_t_pp)
            g_θ_explicit = transpose(J) * vec(g_w)

            g_θ_fast = zeros(length(θ_pp))
            apply_jacobian_transpose!(g_θ_fast, g_w, θ_pp, p, n_ctrl_pp, n_t_pp)

            @test isapprox(g_θ_fast, g_θ_explicit; atol = 1e-12)
        end

        @testset "PiecewiseConstant fast-path identity" begin
            n_ctrl_pp, n_t_pp = 3, 5
            θ_pp = randn(rng_p, n_ctrl_pp * n_t_pp)
            g_w  = randn(rng_p, n_ctrl_pp, n_t_pp)
            g_θ_fast = zeros(length(θ_pp))
            apply_jacobian_transpose!(g_θ_fast, g_w, θ_pp, PiecewiseConstant(),
                                       n_ctrl_pp, n_t_pp)
            @test g_θ_fast == vec(g_w)
        end

        @testset "Diagonal default fall-through (TanhParam)" begin
            u_max = 2.0
            p_t   = TanhParam(u_max)
            n_ctrl_pp, n_t_pp = 2, 4
            θ_pp = randn(rng_p, n_ctrl_pp * n_t_pp)
            g_w  = randn(rng_p, n_ctrl_pp, n_t_pp)

            # Default goes through waveform_jacobian (Diagonal)
            g_θ_default = zeros(length(θ_pp))
            apply_jacobian_transpose!(g_θ_default, g_w, θ_pp, p_t,
                                       n_ctrl_pp, n_t_pp)

            # Hand-computed: J is Diagonal, so g_θ[i] = J[i,i] · vec(g_w)[i]
            J = waveform_jacobian(θ_pp, p_t, n_ctrl_pp, n_t_pp)
            g_θ_ref = diag(J) .* vec(g_w)
            @test isapprox(g_θ_default, g_θ_ref; atol = 1e-12)
        end
    end
end
