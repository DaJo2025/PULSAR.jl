# Regression tests for tGRAPE (time-optimal GRAPE).
#
# The original gate-target Q recursion was buggy — it satisfied
#     Q[k] = (Q[k+1] · U_k)†
# instead of the documented invariant
#     Q[k] = (U_{n_t} … U_{k+1})†
# producing an FD/analytic mismatch of ~3% in the amplitude block.
# The invariant Q[k]† · U_k · P[k] = U_total is the diagnostic.

using Test
using PULSAR
using LinearAlgebra
using Random

const _tgrape_step_hamiltonian! = PULSAR._tgrape_step_hamiltonian!
const _tgrape_gate_value_and_grad! = PULSAR._tgrape_gate_value_and_grad!
const _tgrape_state_value_and_grad! = PULSAR._tgrape_state_value_and_grad!

@testset "tGRAPE — gradient correctness" begin

    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = qubit_system(1, (2π * 0.3 / 2) .* σz, [σx, σy])

    rng = MersenneTwister(2026)
    n_c, n_t = 2, 6

    @testset "gate-target FD agreement" begin
        # Build a Haar-random target unitary
        A = randn(rng, ComplexF64, 2, 2)
        Q_, R_ = qr(A)
        D = Diagonal(sign.(diag(Matrix(R_))))
        U_target = Matrix(Q_) * D
        @test norm(U_target * U_target' - I) < 1e-12

        w      = randn(rng, n_c, n_t) .* 0.5
        dt_vec = fill(0.005, n_t)

        g_w  = Matrix{Float64}(undef, n_c, n_t)
        g_dt = Vector{Float64}(undef, n_t)
        F0   = _tgrape_gate_value_and_grad!(g_w, g_dt, sys, w, dt_vec, U_target)

        function gate_fid(w_, dt_)
            U_total = Matrix{ComplexF64}(I, 2, 2)
            Hk      = Matrix{ComplexF64}(undef, 2, 2)
            for k in 1:n_t
                _tgrape_step_hamiltonian!(Hk, sys, w_, k)
                U_total = PULSAR.compute_propagator(Hk, dt_[k]) * U_total
            end
            return abs2(tr(U_target' * U_total) / 2)
        end

        # F0 sanity
        @test abs(gate_fid(w, dt_vec) - F0) < 1e-12

        # Amplitude block: O(dt²) first-order error; with dt=0.005 expect ≤ 1e-4
        h = 1e-6
        amp_max = 0.0
        for c in 1:n_c, k in 1:n_t
            wp = copy(w); wp[c,k] += h
            wm = copy(w); wm[c,k] -= h
            fd = (gate_fid(wp, dt_vec) - gate_fid(wm, dt_vec)) / (2h)
            amp_max = max(amp_max, abs(fd - g_w[c,k]))
        end
        @test amp_max < 5e-5  # first-order GRAPE error scales as O(dt²)

        # Time block: derivative is exact, expect ≤ 1e-7
        dt_max = 0.0
        for k in 1:n_t
            dp = copy(dt_vec); dp[k] += 1e-7
            dm = copy(dt_vec); dm[k] -= 1e-7
            fd = (gate_fid(w, dp) - gate_fid(w, dm)) / (2e-7)
            dt_max = max(dt_max, abs(fd - g_dt[k]))
        end
        @test dt_max < 1e-7
    end

    @testset "state-target FD agreement" begin
        ψ_init   = ComplexF64[1.0, 0.0]
        ψ_target = ComplexF64[0.0, 1.0]
        w        = randn(rng, n_c, n_t) .* 0.5
        dt_vec   = fill(0.005, n_t)

        g_w  = Matrix{Float64}(undef, n_c, n_t)
        g_dt = Vector{Float64}(undef, n_t)
        F0   = _tgrape_state_value_and_grad!(g_w, g_dt, sys, w, dt_vec, ψ_init, ψ_target)

        function state_fid(w_, dt_)
            ψ  = copy(ψ_init)
            Hk = Matrix{ComplexF64}(undef, 2, 2)
            for k in 1:n_t
                _tgrape_step_hamiltonian!(Hk, sys, w_, k)
                ψ = PULSAR.compute_propagator(Hk, dt_[k]) * ψ
            end
            return abs2(dot(ψ_target, ψ))
        end

        @test abs(state_fid(w, dt_vec) - F0) < 1e-12

        h = 1e-6
        amp_max = 0.0
        for c in 1:n_c, k in 1:n_t
            wp = copy(w); wp[c,k] += h
            wm = copy(w); wm[c,k] -= h
            fd = (state_fid(wp, dt_vec) - state_fid(wm, dt_vec)) / (2h)
            amp_max = max(amp_max, abs(fd - g_w[c,k]))
        end
        @test amp_max < 5e-5

        dt_max = 0.0
        for k in 1:n_t
            dp = copy(dt_vec); dp[k] += 1e-7
            dm = copy(dt_vec); dm[k] -= 1e-7
            fd = (state_fid(w, dp) - state_fid(w, dm)) / (2e-7)
            dt_max = max(dt_max, abs(fd - g_dt[k]))
        end
        @test dt_max < 1e-7
    end

    @testset "tgrape_optimize end-to-end" begin
        # X-gate via softmax (T fixed) — driven by H_x alone.
        U_target = ComplexF64[0 1; 1 0]
        sys2 = qubit_system(1, ComplexF64[0 0; 0 0], [σx])  # no drift
        target = unitary_target(U_target)

        n_c2, n_t2 = 1, 30
        w0     = 0.01 .* randn(MersenneTwister(7), n_c2, n_t2)
        T_fix  = π / 2  # exact π pulse time when H_c = σx and amp = 1
        dt0    = T_fix / n_t2

        w_opt, dt_opt, F_opt, stats = PULSAR.tgrape_optimize(
            sys2, target, w0, dt0;
            parameterization = :softmax,
            max_iter = 200, tol = 1e-8)

        @test F_opt > 0.99
        @test sum(dt_opt) ≈ T_fix atol = 1e-6
        @test all(dt_opt .> 0.0)
    end
end
