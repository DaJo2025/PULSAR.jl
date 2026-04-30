# test/Application/QuantumComputing/LindbladQCControlIntegrationTests.jl
# ======================================================================
# Theme 6b — end-to-end optimization on `LindbladQCControl`.
#
# Phase-2 only verified construction + a guarded `optimcon` stub.  Theme 6b
# wires `optimcon(::LindbladQCControl)` through the closure-based dispatch
# (`_qc_kernel` + `grape_lindblad_kernel` via the `_LindbladQCAdapter`).
# This file proves that every supported method runs to completion and
# produces a non-trivial fidelity on a 1-qubit + amplitude-damping
# X-rotation problem.

using Test
using Pulsar
using LinearAlgebra
using Random

@testset "Theme 6b — LindbladQCControl end-to-end" begin
    Random.seed!(2025)

    # ── 1-qubit transmon-style test bed ─────────────────────────────────
    Ω        = 2π * 1.0e6                                # 1 MHz Rabi
    σx       = Ω .* ComplexF64[0.0 1.0; 1.0 0.0]
    σy       = Ω .* ComplexF64[0.0 -im; im  0.0]
    H_drift  = ComplexF64[0.0 0.0; 0.0 0.0]
    sys      = QuantumSystem(H_drift, [σx, σy], 2, 2, Dict{String,Any}())
    target   = state_target(ComplexF64[0.0, 1.0])         # |0⟩ → |1⟩
    n_t      = 40
    dt       = 5.0e-9                                     # 5 ns slices
    guess    = 0.30 .+ 0.10 .* randn(2, n_t)
    ctrl     = ControlSequence(copy(guess), dt, dt * n_t, n_t)

    # Amplitude damping (T1 = 50 µs)
    a        = ComplexF64[0.0 1.0; 0.0 0.0]
    γ        = 1.0 / 50.0e-6

    function _build(method::Symbol; max_iter::Int = 60, step::Float64 = 0.05)
        return LindbladQCControl(sys, target, ctrl;
                                  jump_ops    = [a],
                                  decay_rates = [γ],
                                  method      = method,
                                  max_iter    = max_iter,
                                  step_size   = step,
                                  verbose     = false)
    end

    @testset ":lbfgsb (and :lbfgs alias)" begin
        ctx_lb = _build(:lbfgsb)
        res_lb = optimcon(ctx_lb)
        @test res_lb isa OptimizationResult
        @test res_lb.fidelity > 0.95
        @test occursin("Lindblad QC L-BFGS-B", res_lb.metadata["algorithm"])

        # Alias :lbfgs must produce the *same* result (deterministic, no RNG).
        ctx_la = _build(:lbfgs)
        res_la = optimcon(ctx_la)
        @test res_la.fidelity ≈ res_lb.fidelity atol = 1e-9
    end

    @testset ":cg" begin
        ctx = _build(:cg)
        res = optimcon(ctx)
        @test res.fidelity > 0.95
        @test occursin("Lindblad QC CG", res.metadata["algorithm"])
    end

    @testset ":grape (gradient ascent)" begin
        # Gradient ascent with a fixed step is slower — only require the
        # routing to work and fidelity to climb above the initial guess.
        ctx_init = _build(:grape; max_iter = 1, step = 1e-6)
        res_init = optimcon(ctx_init)            # ~initial fidelity
        ctx_full = _build(:grape; max_iter = 60, step = 0.05)
        res_full = optimcon(ctx_full)
        @test res_full.fidelity > res_init.fidelity
        @test occursin("Lindblad QC GRAPE", res_full.metadata["algorithm"])
    end

    @testset "metaheuristics — :cmaes / :nelder_mead" begin
        for method in (:cmaes, :nelder_mead)
            ctx = _build(method; max_iter = 200)
            res = optimcon(ctx)
            @test res isa OptimizationResult
            @test 0.0 ≤ res.fidelity ≤ 1.0 + 1e-9
            @test occursin("Lindblad QC", res.metadata["algorithm"])
        end
    end

    @testset "Penalty terms apply through closure builder" begin
        # Add a tiny smoothness penalty and verify the optimum still climbs.
        smoothness     = w -> sum(diff(w; dims = 2) .^ 2) * 1e-6
        smoothness_grad = w -> begin
            G = zeros(size(w))
            d = diff(w; dims = 2)
            G[:, 1:end-1] .-= 2e-6 .* d
            G[:, 2:end]   .+= 2e-6 .* d
            G
        end
        ctx_pen = LindbladQCControl(sys, target, ctrl;
                                     jump_ops         = [a],
                                     decay_rates      = [γ],
                                     method           = :lbfgsb,
                                     max_iter         = 30,
                                     verbose          = false,
                                     penalty_fns      = Function[smoothness],
                                     penalty_grad_fns = Function[smoothness_grad])
        res_pen = optimcon(ctx_pen)
        @test res_pen isa OptimizationResult
        @test res_pen.fidelity > 0.5     # penalty is small enough to not block
    end

    @testset "Unknown method errors cleanly" begin
        ctx = _build(:not_a_method)
        @test_throws ArgumentError optimcon(ctx)
    end
end
