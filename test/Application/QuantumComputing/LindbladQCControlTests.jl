# test/Application/QuantumComputing/LindbladQCControlTests.jl
# =============================================================
# Theme 6 (Phase 2 skeleton) — exercises the LindbladQCControl
# struct and its keyword constructor.
#
#   • Subtype relation: LindbladQCControl <: AbstractOptimizationContext.
#   • Constructor pre-computes Liouvillians of the right dimension
#     (N² × N²) and vectorised initial / target states (length N²).
#   • Validation: jump_ops / decay_rates length mismatch and negative
#     rates are rejected at construction time.
#   • State-target only: unitary targets are rejected with a clear
#     ArgumentError pending the kernel lift (Phase 2b).
#   • optimcon stub throws a guarded ErrorException — callers cannot
#     accidentally land on a silent Hilbert-space code path.

using Test
using PULSAR
using LinearAlgebra

@testset "Theme 6 — LindbladQCControl skeleton" begin

    # 1-qubit transmon test bed
    # Minimal 1-qubit drive Hamiltonian (rad/s); construct directly to avoid
    # depending on platform constructors here.
    σx        = ComplexF64[0.0 1.0; 1.0 0.0]
    σy        = ComplexF64[0.0 -im; im  0.0]
    H_drift   = ComplexF64[0.0 0.0; 0.0 0.0]
    sys       = QuantumSystem(H_drift, [σx, σy], 2, 2, Dict{String,Any}())
    target    = state_target(ComplexF64[0.0, 1.0])    # |0⟩ → |1⟩
    ctrl      = ControlSequence(0.01 .* randn(2, 50), 5e-9, 5e-7, 50)

    # Single Lindblad jump operator (amplitude damping a = |0⟩⟨1|)
    a       = ComplexF64[0.0 1.0; 0.0 0.0]
    γ       = 1.0 / 50e-6           # T1 = 50 µs

    @testset "Type hierarchy" begin
        @test LindbladQCControl <: AbstractOptimizationContext
    end

    @testset "Constructor — happy path" begin
        ctx = LindbladQCControl(sys, target, ctrl;
                                 jump_ops    = [a],
                                 decay_rates = [γ])
        N  = 2
        N2 = 4
        @test ctx isa LindbladQCControl
        @test ctx._hilbert_dim   == N
        @test ctx._liouville_dim == N2
        @test size(ctx._L_drift) == (N2, N2)
        @test length(ctx._L_controls) == sys.n_controls
        @test all(size(L) == (N2, N2) for L in ctx._L_controls)
        @test length(ctx._sigma_init) == N2
        @test length(ctx._sigma_targ) == N2
        # vec(ρ) → ρ has trace 1
        ρ_init = reshape(ctx._sigma_init, N, N)
        ρ_targ = reshape(ctx._sigma_targ, N, N)
        @test tr(ρ_init) ≈ 1.0 atol = 1e-12
        @test tr(ρ_targ) ≈ 1.0 atol = 1e-12
        @test ctx.method        === :lbfgs
        @test ctx.decay_rates    == [γ]
    end

    @testset "Constructor — closed-system path" begin
        ctx = LindbladQCControl(sys, target, ctrl)
        @test isempty(ctx.jump_ops)
        @test isempty(ctx.decay_rates)
        # Drift Liouvillian must still be 4×4 even with no dissipation
        @test size(ctx._L_drift) == (4, 4)
    end

    @testset "Constructor — validation" begin
        # Length mismatch
        @test_throws ArgumentError LindbladQCControl(sys, target, ctrl;
                                                       jump_ops    = [a],
                                                       decay_rates = Float64[])
        # Negative rate
        @test_throws ArgumentError LindbladQCControl(sys, target, ctrl;
                                                       jump_ops    = [a],
                                                       decay_rates = [-1.0])
        # Unitary target rejected
        U  = ComplexF64[0 1; 1 0]
        ut = unitary_target(U)
        @test_throws ArgumentError LindbladQCControl(sys, ut, ctrl;
                                                       jump_ops    = [a],
                                                       decay_rates = [γ])
    end

    @testset "rho_init keyword (custom initial)" begin
        # |+⟩ initial → vec(ρ) entries equal to 0.5
        psi_plus = ComplexF64[1.0, 1.0] ./ sqrt(2)
        ctx = LindbladQCControl(sys, target, ctrl;
                                 jump_ops    = [a],
                                 decay_rates = [γ],
                                 rho_init    = psi_plus)
        ρ_plus = reshape(ctx._sigma_init, 2, 2)
        @test tr(ρ_plus) ≈ 1.0 atol = 1e-12
        @test all(abs(z - 0.5) < 1e-12 for z in ctx._sigma_init)
    end

    @testset "optimcon — closure-based dispatch (Theme 6b)" begin
        # Theme 6b lifts the Phase-2 stub: optimcon now routes through
        # `_qc_kernel` + `grape_lindblad_kernel` via the `_LindbladQCAdapter`.
        ctx = LindbladQCControl(sys, target, ctrl;
                                 jump_ops    = [a],
                                 decay_rates = [γ],
                                 method      = :lbfgsb,
                                 max_iter    = 5,
                                 verbose     = false)
        res = optimcon(ctx)
        @test res isa OptimizationResult
        @test 0.0 ≤ res.fidelity ≤ 1.0 + 1e-9
        @test haskey(res.metadata, "algorithm")
        @test occursin("Lindblad QC", res.metadata["algorithm"])
        @test res.metadata["liouville_dim"] == 4

        # Unknown method must still error.
        ctx_bad = LindbladQCControl(sys, target, ctrl;
                                     jump_ops    = [a],
                                     decay_rates = [γ],
                                     method      = :not_a_method)
        @test_throws ArgumentError optimcon(ctx_bad)
    end
end
