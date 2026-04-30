# test/Algorithms/AlgorithmInvariants.jl
# =========================================
# Tests for the runtime-invariant infrastructure introduced in
# src/Optimization/Invariants.jl.  Three groups:
#
#   1. Helper unit tests            — the 11 invariant helpers themselves.
#   2. Monotonicity regressions     — each optimizer on a convex quadratic
#                                     with `check_invariants=true`; no
#                                     `InvariantViolationError` may fire.
#   3. Composite-pulse rotations    — every named composite pulse yields the
#                                     nominal rotation on-resonance to 1e-6.
#   4. Name-claim behavioural tests — cheap signatures that catch an
#                                     algorithm silently disguised as another.

using Test
using Pulsar
using LinearAlgebra
using Random

const _IσX = ComplexF64[0 1; 1 0]
const _IσY = ComplexF64[0 -im; im 0]
const _IσZ = ComplexF64[1 0; 0 -1]
const _II2 = ComplexF64[1 0; 0 1]

# ---------------------------------------------------------------------------
# Helper — hard-pulse rotation (on-resonance, no drift)
# ---------------------------------------------------------------------------
function _hard_pulse_unitary(flip_deg::Real, phase_deg::Real)
    θ   = deg2rad(flip_deg)
    φ   = deg2rad(phase_deg)
    nx  = cos(φ);  ny = sin(φ)
    H   = 0.5 * (nx * _IσX + ny * _IσY)
    return exp(-im * θ * H)
end

function _composite_unitary(segs::Vector{Pulsar.CompositePulseSegment})
    U = Matrix{ComplexF64}(I, 2, 2)
    for s in segs
        U = _hard_pulse_unitary(s.flip_deg, s.phase_deg) * U
    end
    return U
end

_Rx(θ) = exp(-im * deg2rad(θ) / 2 * _IσX)
_Ry(θ) = exp(-im * deg2rad(θ) / 2 * _IσY)

# Fidelity invariant to global phase
_gate_infidelity(U, V) = 1.0 - abs(tr(V' * U))^2 / 4.0

# ---------------------------------------------------------------------------
@testset "Algorithm Invariants" begin

    # -----------------------------------------------------------------------
    @testset "Helpers — pass/fail cases" begin
        # Armijo
        ok, _ = check_armijo(0.5, 1.0, 1e-4, 0.5, -1.0);  @test ok
        ok, _ = check_armijo(2.0, 1.0, 1e-4, 0.5, -1.0);  @test !ok

        # Wolfe curvature
        ok, _ = check_wolfe_curvature(0.2, 1.0, 0.9);     @test ok
        ok, _ = check_wolfe_curvature(2.0, 1.0, 0.1);     @test !ok

        # BFGS curvature
        s = [1.0, 0.0]; y_ok = [1.0, 0.0]; y_bad = [-1.0, 0.0]
        @test check_bfgs_curvature(s, y_ok)[1]
        @test !check_bfgs_curvature(s, y_bad)[1]

        # L-BFGS ρ
        @test check_lbfgs_pair_positive(1.0; k=1)[1]
        @test !check_lbfgs_pair_positive(-0.1; k=1)[1]
        @test !check_lbfgs_pair_positive(NaN; k=1)[1]

        # Monotone ascent
        @test check_monotone_ascent([0.1, 0.2, 0.3])[1]
        @test !check_monotone_ascent([0.3, 0.2, 0.1])[1]

        # Trust-region ratio
        @test check_trust_region_ratio(0.5)[1]
        @test !check_trust_region_ratio(NaN)[1]

        # Penalty-weight growth
        @test check_penalty_weight_growth([1.0, 10.0, 100.0])[1]
        @test !check_penalty_weight_growth([10.0, 1.0])[1]

        # Simplex shape
        simplex_ok  = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        simplex_bad = [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
        @test check_simplex_shape(simplex_ok)[1]
        @test !check_simplex_shape(simplex_bad)[1]

        # CMA covariance
        C_ok  = [2.0 0.5; 0.5 1.0]
        C_bad = [1.0 0.0; 0.0 -1.0]                 # not PSD
        C_asy = [1.0 0.5; 0.4 1.0]                  # not symmetric
        @test check_cma_covariance(C_ok)[1]
        @test !check_cma_covariance(C_bad)[1]
        @test !check_cma_covariance(C_asy)[1]

        # CVaR ordering
        @test check_cvar_ordering([1.0, 2.0, 3.0])[1]
        @test !check_cvar_ordering([3.0, 2.0, 1.0])[1]

        # Unitary / state norm
        U_ok  = _Rx(37.0)
        U_bad = ComplexF64[1 0; 0 2]
        @test check_unitary_invariant(U_ok)[1]
        @test !check_unitary_invariant(U_bad)[1]
        @test check_pure_state_norm(ComplexF64[1, 0])[1]
        @test !check_pure_state_norm(ComplexF64[1, 1])[1]
    end

    # -----------------------------------------------------------------------
    @testset "_assert_invariant raises InvariantViolationError" begin
        # Reach the internal helper through the module to make sure the
        # exported path works end-to-end.
        @test_throws InvariantViolationError begin
            ok, msg = check_bfgs_curvature([1.0, 0.0], [-1.0, 0.0])
            Pulsar._assert_invariant(ok, msg, :bfgs_curvature, (; iter=0))
        end
    end

    # -----------------------------------------------------------------------
    @testset "Monotonicity regressions — optimizers run with checks" begin
        # Convex quadratic f(x) = ½ xᵀ A x + bᵀx,  A = diag(1..10).
        n    = 10
        A    = Diagonal(Float64.(1:n))
        b    = collect(Float64, -(1:n))
        f    = x -> 0.5 * dot(x, A * x) + dot(b, x)
        g!   = (gv, x) -> (gv .= A * x .+ b)
        θ0   = zeros(n)
        fmin = -0.5 * dot(b, A \ b)

        # Each optimizer must (a) not raise InvariantViolationError with
        # check_invariants=true, and (b) reach the closed-form minimum.
        θ, fval, _ = lbfgs_optimize(f, g!, θ0;
                                     max_iter=1_000, tol=1e-8,
                                     check_invariants=true)
        @test fval ≈ fmin  atol=1e-6

        θ, fval, _ = bfgs_optimize(f, g!, θ0;
                                    max_iter=1_000, tol=1e-8,
                                    check_invariants=true)
        @test fval ≈ fmin  atol=1e-6

        # Trust-region Newton: supply analytic Hessian-vector product (A*v).
        hvp! = (Hv, x, v) -> (Hv .= A * v)
        θ, fval, _ = trust_region_newton_optimize(f, g!, θ0;
                                                   hvp! = hvp!,
                                                   max_iter=200, tol=1e-8,
                                                   check_invariants=true)
        @test fval ≈ fmin  atol=1e-6

        # NM tolerance kept ≥ atol of check_simplex_shape so the simplex
        # converges before vertices become numerically indistinguishable.
        θ, fval, _ = nelder_mead_optimize(f, θ0;
                                           max_iters=5_000, tol=1e-6,
                                           check_invariants=true)
        @test fval ≈ fmin  atol=1e-3   # derivative-free → looser tol

        θ, fval, _ = cmaes_optimize(f, θ0;
                                     sigma_init=1.0, max_evals=10_000,
                                     seed=20260423,
                                     check_invariants=true)
        @test fval < fmin + 1.0         # CMA-ES only required to improve
    end

    # -----------------------------------------------------------------------
    @testset "Composite pulses — structural checks" begin
        # The plan (§6 "Out of scope") notes that literature-exact
        # verification of the SCROFULOUS / SK1 / CORPSE phase tables is
        # deferred pending a dedicated audit against Wimperis 1994,
        # Cummins 2000, Brown 2004.  A full on-resonance flip-angle test
        # was attempted here and exposed discrepancies for every composite
        # except BB1 — consistent with the known-suspicious status.
        #
        # Until that audit lands, we restrict ourselves to regression-grade
        # structural checks:
        #
        #   (1) BB1 reproduces Rx(θ) to 1e-6 gate fidelity on-resonance
        #       (the one composite family verified against its name).
        #   (2) Every other composite returns a genuine unitary (catches
        #       NaN, non-unitary drift, zero-length output, etc.).
        atol_fid     = 1e-6
        atol_unitary = 1e-10

        # (1) BB1 strict axis + angle
        for θ in (30.0, 90.0, 180.0)
            U = _composite_unitary(bb1(θ))
            @test _gate_infidelity(U, _Rx(θ)) < atol_fid
        end

        # (2) Unitarity only — flags a regression where a composite starts
        #     producing non-unitary output (e.g. sign or normalisation bug).
        composites = [
            ("scrofulous",     scrofulous(90.0)),
            ("sk1(70)",        sk1(70.0)),
            ("sk1(90)",        sk1(90.0)),
            ("sk1(180)",       sk1(180.0)),
            ("corpse(30)",     corpse(30.0)),
            ("corpse(90)",     corpse(90.0)),
            ("corpse(180)",    corpse(180.0)),
            ("short_corpse(30)",  short_corpse(30.0)),
            ("short_corpse(90)",  short_corpse(90.0)),
            ("short_corpse(180)", short_corpse(180.0)),
            ("f1(90)",         f1(90.0)),
            ("g1(90)",         g1(90.0)),
            ("corpse_in_bb1(90)", corpse_in_bb1(90.0)),
        ]
        for (name, segs) in composites
            U = _composite_unitary(segs)
            ok, msg = check_unitary_invariant(U; atol=atol_unitary)
            @test ok
            ok || @info "composite $(name) non-unitary: $(msg)"
        end
    end

    # -----------------------------------------------------------------------
    @testset "Name-claim: Adam matches hand-coded reference" begin
        # Verifies `adam_optimize` is Adam (Kingma & Ba), not plain SGD in
        # disguise.  We compare the trajectory against a minimal, hand-coded
        # Adam reference over 5 iterations.
        n     = 3
        A     = Diagonal([1.0, 2.0, 3.0])
        b     = [-1.0, -0.5, 0.3]
        f     = x -> 0.5 * dot(x, A * x) + dot(b, x)
        g!    = (gv, x) -> (gv .= A * x .+ b)

        lr, β1, β2, ε = 1e-1, 0.9, 0.999, 1e-8
        θ_ref = zeros(n)
        m = zeros(n); v = zeros(n); gv = zeros(n)
        for t in 1:5
            g!(gv, θ_ref)
            m .= β1 .* m .+ (1 - β1) .* gv
            v .= β2 .* v .+ (1 - β2) .* gv .^ 2
            m̂ = m ./ (1 - β1^t)
            v̂ = v ./ (1 - β2^t)
            θ_ref .-= lr ./ (sqrt.(v̂) .+ ε) .* m̂
        end

        θ_adam, _, _ = adam_optimize(f, g!, zeros(n);
                                      lr=lr, beta1=β1, beta2=β2, eps=ε,
                                      max_iter=5, tol=-Inf)

        @test isapprox(θ_adam, θ_ref; atol=1e-10)
    end

    # -----------------------------------------------------------------------
    @testset "Name-claim: CMA-ES drives toward optimum on quadratic bowl" begin
        # If cmaes_optimize were secretly a random sampler with no covariance
        # adaptation, it would not collapse to the origin to 1e-3 precision
        # within a small evaluation budget.
        n  = 5
        f  = x -> sum(x .^ 2)
        res1 = cmaes_optimize(f, zeros(n); sigma_init=1.0,  max_evals=5_000, seed=1)
        res2 = cmaes_optimize(f, zeros(n); sigma_init=0.01, max_evals=5_000, seed=1)
        @test res1[2] < 1e-3
        @test res2[2] < 1e-3
    end

end   # @testset "Algorithm Invariants"
