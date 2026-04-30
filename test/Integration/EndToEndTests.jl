# ============================================================
# Pulsar.jl — End-to-End Integration Tests
# Pulse Design Library for Spin Control Algorithms and Rollout
# ============================================================

using Test, LinearAlgebra, Random

@testset "End-to-End Integration Tests" begin

    # ── Helpers ────────────────────────────────────────────────
    function simple_qubit(; n_ts=80, dt=1e-7, amp=0.1, seed=42)
        # Realistic 1-qubit fixture: H_ctrl scaled to 1 MHz Rabi so that
        # u ∈ O(1) drives full π flips over the 8-µs / 80-step pulse.
        # Target a Pauli-X gate so that small random initial controls do
        # NOT trivially yield F = 1 (state targets evaluate
        # |⟨ψ|U|ψ⟩|², which is 1 for U = I).
        σ_x = ComplexF64[0 1; 1 0]
        σ_y = ComplexF64[0 -1im; 1im 0]
        H_ctrl = [2π * 1e6 * σ_x, 2π * 1e6 * σ_y]
        system = quantum_system(zeros(ComplexF64, 2, 2), H_ctrl)
        target = unitary_target(σ_x)
        rng    = MersenneTwister(seed)
        u      = amp .* (2 .* rand(rng, 2, n_ts) .- 1)
        cs     = ControlSequence(u, dt, dt*n_ts, n_ts)
        return system, target, cs
    end

    # ── Basic GRAPE workflow ───────────────────────────────────
    @testset "GRAPE improves from random initialisation" begin
        system, target, controls = simple_qubit()
        f0 = compute_fidelity(system, controls, target)
        result = grape_optimize(system, target, controls;
                                 config=GRAPEConfig(max_iter=300, verbose=false))
        @test result.fidelity > f0       # Must improve
        @test result.fidelity > 0.5      # Must get somewhere useful
        @test !any(isnan, result.controls)
        @test result.n_iterations <= 300
        @test result isa OptimizationResult
    end

    @testset "GRAPE OptimizationResult fields" begin
        system, target, controls = simple_qubit()
        result = grape_optimize(system, target, controls;
                                 config=GRAPEConfig(max_iter=50, verbose=false))
        @test length(result.fidelity_history) >= 1
        @test result.total_time >= 0
        @test result.n_fidelity_evaluations >= 1
        @test result.termination_reason isa String
        @test result.converged isa Bool
    end

    # ── Problem library ────────────────────────────────────────
    @testset "Hadamard problem from library" begin
        sys, tgt, ctrl = hadamard_gate_problem(n_timesteps=40)
        validate_all(sys, tgt, ctrl)   # Pre-flight checks
        result = grape_optimize(sys, tgt, ctrl;
                                 config=GRAPEConfig(max_iter=300, verbose=false))
        @test result.fidelity > 0.7
    end

    @testset "State transfer |0⟩→|1⟩ from library" begin
        sys, tgt, ctrl = state_transfer_0_to_1()
        result = grape_optimize(sys, tgt, ctrl;
                                 config=GRAPEConfig(max_iter=300, verbose=false))
        @test result.fidelity > 0.7
    end

    # ── Algorithm selection ────────────────────────────────────
    @testset "recommend_optimizer returns valid recommendation" begin
        system, target, _ = simple_qubit()
        rec = recommend_optimizer(system, target, 50)
        @test rec isa AlgorithmRecommendation
        @test rec.method in (:grape, :lbfgs, :bfgs, :cmaes, :nelder_mead,
                              :trust_region, :constrained_grape, :robust_grape)
        @test rec.expected_iterations > 0
        @test length(rec.reasoning) > 5
    end

    @testset "auto_optimize runs without error" begin
        sys, tgt, ctrl = hadamard_gate_problem(n_timesteps=30)
        result = auto_optimize(sys, tgt, ctrl; verbose=false)
        @test result isa OptimizationResult
        @test !any(isnan, result.controls)
    end

    # ── L-BFGS ────────────────────────────────────────────────
    @testset "L-BFGS runs and improves" begin
        system, target, controls = simple_qubit()
        f0 = compute_fidelity(system, controls, target)
        result = lbfgs_optimize(system, target, controls;
                                 config=LBFGSConfig(max_iter=100, verbose=false))
        @test result.fidelity > f0
        @test !any(isnan, result.controls)
    end

    # ── Constrained optimization ───────────────────────────────
    @testset "Bound constraint satisfied after optimization" begin
        system, target, controls = simple_qubit(; amp=0.5)
        constraint_bound = 0.3
        constraints = [BoundConstraint(-constraint_bound, constraint_bound, Int[])]
        result = constrained_optimize(system, target, controls, constraints;
                                       config=ConstrainedConfig(max_iter=100, verbose=false))
        @test all(result.controls .>= -constraint_bound - 1e-6)
        @test all(result.controls .<= constraint_bound + 1e-6)
    end

    # ── Checkpoint and resume ──────────────────────────────────
    @testset "Checkpoint save and load round-trip" begin
        controls_mat = rand(MersenneTwister(1), 2, 30)
        chk = Checkpoint(
            controls_mat, 0.875, 2, 30;
            iteration             = 100,
            fidelity_history      = collect(range(0.5, 0.875; length=100)),
            gradient_norm_history = collect(range(0.2, 0.01; length=100)),
            optimizer_state       = Dict{String,Any}("method" => "grape"),
            system_dim            = 2,
            metadata              = Dict{String,Any}("dt" => 1e-8),
        )
        filepath = tempname() * ".jls"
        save_checkpoint(filepath, chk)
        @test isfile(filepath)
        loaded = load_checkpoint(filepath)
        @test loaded.iteration == 100
        @test abs(loaded.F_opt - 0.875) < 1e-10
        @test size(loaded.w_opt) == (2, 30)
        @test loaded.n_controls == 2
        @test loaded.n_timesteps == 30
        rm(filepath)
    end

    # ── Parameter validation ───────────────────────────────────
    @testset "validate_system catches non-Hermitian H_drift" begin
        H_bad = ComplexF64[1.0+0im 2.0; 3.0 4.0]   # Not Hermitian
        @test_throws ArgumentError quantum_system(H_bad, [])
    end

    @testset "validate_controls catches zero dt" begin
        σ_x = ComplexF64[0 1; 1 0]
        system = quantum_system(zeros(ComplexF64,2,2), [σ_x])
        bad_cs = ControlSequence(zeros(1, 10), 0.0, 0.0, 10)
        @test_throws ArgumentError validate_controls(bad_cs)
    end

    @testset "validate_target catches non-normalised state" begin
        system, target, _ = simple_qubit()
        bad_tgt = state_target(ComplexF64[2.0, 0.0])  # norm = 2
        @test_throws ArgumentError validate_target(bad_tgt, system)
    end

    # ── Fidelity sanity ────────────────────────────────────────
    @testset "Fidelity == 1 for trivial case (zero drift, zero controls)" begin
        σ_x = ComplexF64[0 1; 1 0]
        σ_y = ComplexF64[0 -1im; 1im 0]
        system = quantum_system(zeros(ComplexF64,2,2), [σ_x, σ_y])
        # Identity gate target
        tgt   = unitary_target(Matrix{ComplexF64}(I, 2, 2))
        u_zero = zeros(2, 10)
        cs    = ControlSequence(u_zero, 1e-9, 1e-8, 10)
        F = compute_fidelity(system, cs, tgt)
        @test F ≈ 1.0 atol=1e-10
    end

    # ── Sensitivity analysis ───────────────────────────────────
    @testset "Sensitivity result has correct shape" begin
        system, target, controls = simple_qubit()
        result = grape_optimize(system, target, controls;
                                 config=GRAPEConfig(max_iter=50, verbose=false))
        sens = compute_sensitivity(result, system, target)
        @test size(sens.control_sensitivities) == (system.n_controls, controls.n_timesteps)
        @test size(sens.normalized_sensitivities) == size(sens.control_sensitivities)
        @test maximum(sens.normalized_sensitivities) <= 1.0 + 1e-10
        @test minimum(sens.normalized_sensitivities) >= 0.0 - 1e-10
    end

    # ── Performance monitor ────────────────────────────────────
    @testset "PerformanceMonitor records data" begin
        mon = PerformanceMonitor()
        record_iteration!(mon, 1, 0.5, 0.1, 0.01, :cpu)
        record_iteration!(mon, 2, 0.7, 0.05, 0.01, :cpu)
        summary = get_summary(mon)
        @test summary["n_iterations"] == 2
        @test summary["best_fidelity"] ≈ 0.7
        @test summary["final_fidelity"] ≈ 0.7
    end

    # ── Uncertainty quantification ─────────────────────────────
    @testset "Uncertainty estimate has correct types" begin
        system, target, controls = simple_qubit()
        result = grape_optimize(system, target, controls;
                                 config=GRAPEConfig(max_iter=50, verbose=false))
        uq = estimate_uncertainty(result, system, target;
                                   config=UQConfig(method=:hessian, n_samples=10))
        @test uq isa UncertaintyResult
        @test uq.optimal_fidelity ≈ result.fidelity atol=1e-8
        @test all(uq.control_uncertainty .>= 0)
        @test uq.fidelity_ci_lower <= uq.fidelity_ci_upper
    end

end
