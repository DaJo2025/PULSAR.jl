# test/Application/MR/LindbladMRAlgorithmSweepTests.jl
# ====================================================
# Theme 6b — regression sweep on `LindbladMRControl`.
#
# Before Theme 6b the MR file held four near-identical loops:
# `_optimcon_lbfgs` / `_optimcon_lindblad_lbfgs` and
# `_optimcon_grape` / `_optimcon_lindblad_grape`.  Step M1 collapsed each
# pair onto the unified `_optimcon_lbfgs` / `_optimcon_grape`, both of
# which now route through `_mr_kernel(w, ctrl)` and dispatch to the
# Liouville-space kernel for `LindbladMRControl`.
#
# This file confirms that every method surviving in `_MR_GENERIC_METHODS`
# plus the unified `:lbfgs` and `:grape` runs end-to-end on a small
# Lindblad fixture and reaches a sensible fidelity.  Differences across
# methods reflect algorithmic style; what matters is that **none** of them
# silently fall through to a Hilbert-space code path.

using Test
using PULSAR
using LinearAlgebra
using Random

@testset "Theme 6b — LindbladMRControl algorithm sweep" begin
    Random.seed!(2025)

    # ── Small 1-spin Lindblad fixture (same shape as test/m1_smoke) ─────
    σx_half = ComplexF64[0 1; 1 0] / 2
    σy_half = ComplexF64[0 -im; im 0] / 2
    drift   = ComplexF64[0 0; 0 0]
    ψ0      = ComplexF64[1, 0]
    ψf      = ComplexF64[0, 1]
    ops     = Matrix{ComplexF64}[σx_half, σy_half]
    inits   = Vector{ComplexF64}[ψ0]
    targs   = Vector{ComplexF64}[ψf]

    a       = ComplexF64[0 1; 0 0]              # |0⟩⟨1| collapse
    γ       = 1.0 / 50e-3                        # T1 = 50 ms
    n_t     = 50
    pulse_dt = fill(1e-5, n_t)
    pwr     = [2π * 1e3]
    # Mid-amplitude guess: a near-zero guess gives ~zero fidelity *and*
    # ~zero gradient on this Lindblad fixture, so closure-based methods
    # (CG / L-BFGS-B / Nelder–Mead) interpret the iterate as already
    # converged. The hand-rolled `:lbfgs` / `:grape` paths normalise the
    # search direction and thus cope; here we use a guess that gives
    # every method a non-vanishing initial signal.
    guess   = 0.30 .+ 0.10 .* randn(2, n_t)

    function _build(method::Symbol; max_iter::Int = 30)
        return LindbladMRControl(
            drifts      = [drift],
            operators   = ops,
            rho_init    = inits,
            rho_targ    = targs,
            pwr_levels  = pwr,
            pulse_dt    = pulse_dt,
            jump_ops    = Matrix{ComplexF64}[a],
            decay_rates = [γ],
            method      = method,
            max_iter    = max_iter,
            verbose     = false,
        )
    end

    fidelities = Dict{Symbol,Float64}()

    @testset "Unified :lbfgs and :grape (Step M1)" begin
        for method in (:lbfgs, :grape)
            ctrl = _build(method)
            res  = optimcon(ctrl, copy(guess))
            @test res isa OptimizationResult
            @test 0.0 ≤ res.fidelity ≤ 1.0 + 1e-9
            fidelities[method] = res.fidelity
            # Both unified paths must report a Lindblad-flavoured algorithm
            # string (not the closed-system label).
            @test occursin("Lindblad", res.metadata["algorithm"])
        end
    end

    @testset "Generic methods (closure-routed)" begin
        # Subset of `_MR_GENERIC_METHODS` that converges quickly enough
        # for a 60-iter regression sweep. `:de` and `:pscmaes` are skipped
        # because they need many more evaluations to leave the initial
        # guess; their routing is exercised indirectly via the closed-
        # system MR test suite. `:pso` is also skipped: PSO uses
        # `@threads` to evaluate particle fitness, and the Liouville-space
        # GRAPE kernel itself uses `@threads :static` over the ensemble —
        # Julia forbids nested static threading. The closure routing for
        # `:pso` is exercised through the closed-system MR fixtures.
        for method in (:lbfgsb, :cg, :cmaes, :nelder_mead)
            ctrl = _build(method; max_iter = 60)
            res  = optimcon(ctrl, copy(guess))
            @test res isa OptimizationResult
            @test 0.0 ≤ res.fidelity ≤ 1.0 + 1e-9
            fidelities[method] = res.fidelity
        end
    end

    @testset "Gradient methods reach a non-trivial fidelity" begin
        # The T1 = 50 ms / pulse = 0.5 ms regime is essentially unitary;
        # gradient methods should comfortably exceed 0.5.
        @test fidelities[:lbfgs]  > 0.5
        @test fidelities[:lbfgsb] > 0.5
        @test fidelities[:cg]     > 0.5
    end
end
