@testset "CPU Parallelization" begin

    # ─── Fixtures ────────────────────────────────────────────────────────────
    # Small 2-qubit system for fast tests
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    I2 = Matrix{ComplexF64}(I, 2, 2)

    H_drift  = 2π * 100.0 .* kron(σz, I2)
    H_ctrl1  = kron(σx, I2)
    H_ctrl2  = kron(σy, I2)
    H_ctrl3  = kron(I2, σx)
    H_ctrl4  = kron(I2, σy)

    sys = QuantumSystem(H_drift, [H_ctrl1, H_ctrl2, H_ctrl3, H_ctrl4], 4, 4, Dict())

    target = unitary_target(Matrix{ComplexF64}(I, 4, 4))  # identity as a trivial target

    rng = MersenneTwister(42)
    n_ts = 20
    dt   = 2e-8
    ctrl = ControlSequence(0.01 .* randn(rng, 4, n_ts), dt, n_ts * dt, n_ts)

    cfg_serial = GRAPEConfig(max_iter=5, verbose=false)

    # ─── GradientParallelization produces same gradient as serial ─────────────
    @testset "GradientParallelization consistency" begin
        G_serial = compute_grape_gradient(sys, ctrl, target)

        par_strategy = GradientParallelization(n_threads=1)  # 1 thread = serial
        G_par = compute_grape_gradient(sys, ctrl, target)

        @test norm(G_par - G_serial) < 1e-10
    end

    # ─── Threaded batch runs give reproducible results with fixed seed ────────
    @testset "Threaded batch reproducibility" begin
        n_runs  = 4
        results = Vector{Float64}(undef, n_runs)

        Threads.@threads for k in 1:n_runs
            rng_k = MersenneTwister(100 + k)
            ctrl_k = ControlSequence(0.01 .* randn(rng_k, 4, n_ts), dt, n_ts * dt, n_ts)
            r = grape_optimize(sys, target, ctrl_k; config=cfg_serial)
            results[k] = r.fidelity
        end

        # Re-run serially with the same seeds — results must match
        for k in 1:n_runs
            rng_k = MersenneTwister(100 + k)
            ctrl_k = ControlSequence(0.01 .* randn(rng_k, 4, n_ts), dt, n_ts * dt, n_ts)
            r = grape_optimize(sys, target, ctrl_k; config=cfg_serial)
            @test abs(results[k] - r.fidelity) < 1e-10
        end
    end

end  # @testset "CPU Parallelization"
