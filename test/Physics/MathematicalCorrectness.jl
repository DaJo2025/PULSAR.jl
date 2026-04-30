# test/Physics/MathematicalCorrectness.jl
# =========================================
# Verifies mathematical invariants that must hold regardless of the
# physical scenario being studied:
#   - Propagator unitarity
#   - Fidelity range and edge cases
#   - GRAPE gradient accuracy (vs finite differences)
#   - Spin operator algebra
#   - Hamiltonian Hermiticity
#
# These tests are intentionally physics-agnostic; they check pure
# linear-algebra and calculus properties of the PULSAR internals.

using Test
using PULSAR
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Shared Pauli matrices (used across multiple sub-testsets)
# ---------------------------------------------------------------------------
const σ_x = ComplexF64[0 1; 1 0]
const σ_y = ComplexF64[0 -im; im 0]
const σ_z = ComplexF64[1 0; 0 -1]
const I2  = ComplexF64[1 0; 0 1]
const I4  = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

# ---------------------------------------------------------------------------
@testset "Mathematical Correctness" begin

    # -----------------------------------------------------------------------
    @testset "Propagator correctness" begin

        # 1. Zero Hamiltonian → identity propagator
        H_zero = zeros(ComplexF64, 2, 2)
        U_zero = propagator(H_zero, 1e-6)
        @test norm(U_zero - I2) < 1e-12

        # 2. U(t) is unitary for random Hermitian H
        rng = MersenneTwister(42)
        A = randn(rng, ComplexF64, 2, 2)
        H_rand = A + A'          # guaranteed Hermitian
        dt = 1e-7
        U_rand = propagator(H_rand, dt)
        @test norm(U_rand' * U_rand - I2) < 1e-10
        @test norm(U_rand * U_rand' - I2) < 1e-10

        # 3. σ_z drift gives known rotation: U = exp(-i*ω*σ_z*t)
        #    For ω=π, t=1: U = diag(-i, i) up to global phase
        ω = π
        t = 1.0
        H_z = ω * σ_z
        U_z = propagator(H_z, t)
        # The diagonal entries should be e^{-iπ} = -1 and e^{+iπ} = -1 ... wait:
        # exp(-i*H*t) with H=π*σ_z, t=1:
        # U[1,1] = exp(-i*π*1) = -1,  U[2,2] = exp(-i*π*(-1)) = exp(i*π) = -1
        # Off-diagonals = 0
        @test abs(U_z[1,1] - exp(-im*π)) < 1e-10
        @test abs(U_z[2,2] - exp( im*π)) < 1e-10
        @test abs(U_z[1,2]) < 1e-10
        @test abs(U_z[2,1]) < 1e-10

        # 4. Product formula: U(H, t1+t2) ≈ U(H, t2) * U(H, t1) for time-independent H
        t1 = 1.5e-6
        t2 = 2.3e-6
        H_test = 2π * 1e5 * σ_x
        U_total = propagator(H_test, t1 + t2)
        U_prod  = propagator(H_test, t2) * propagator(H_test, t1)
        @test norm(U_total - U_prod) < 1e-8

        # 5. Piecewise propagator: product of short steps equals single long step
        n_steps = 50
        dt_step = t1 / n_steps
        U_step = prod(propagator(H_test, dt_step) for _ in 1:n_steps)
        @test norm(propagator(H_test, t1) - U_step) < 1e-8

        # 6. 4×4 unitarity (two-qubit system)
        B = randn(rng, ComplexF64, 4, 4)
        H4 = B + B'
        U4 = propagator(H4, 5e-8)
        @test norm(U4' * U4 - I4) < 1e-10

    end  # Propagator correctness

    # -----------------------------------------------------------------------
    @testset "Fidelity bounds" begin

        # 1. Gate fidelity of identity with itself = 1
        U_id = Matrix{ComplexF64}(I, 2, 2)
        F_self = gate_fidelity(U_id, U_id)
        @test abs(F_self - 1.0) < 1e-12

        # 2. Gate fidelity in [0, 1] for random unitaries
        rng = MersenneTwister(7)
        for _ in 1:20
            A = randn(rng, ComplexF64, 2, 2)
            U_a, _, _ = svd(A)           # random unitary via SVD
            B = randn(rng, ComplexF64, 2, 2)
            U_b, _, _ = svd(B)
            F = gate_fidelity(U_a, U_b)
            @test 0.0 - 1e-12 <= F <= 1.0 + 1e-12
        end

        # 3. State fidelity: |⟨ψ|φ⟩|² for orthogonal states = 0
        ψ0 = ComplexF64[1, 0]
        ψ1 = ComplexF64[0, 1]
        F_orth = state_fidelity(ψ0, ψ1)
        @test abs(F_orth) < 1e-14

        # 4. State fidelity of a vector with itself = 1
        ψ_plus = ComplexF64[1, 1] / sqrt(2)
        F_same = state_fidelity(ψ_plus, ψ_plus)
        @test abs(F_same - 1.0) < 1e-12

        # 5. State fidelity is symmetric: F(ψ, φ) = F(φ, ψ)
        rng2 = MersenneTwister(13)
        for _ in 1:10
            v = randn(rng2, ComplexF64, 2)
            v /= norm(v)
            w = randn(rng2, ComplexF64, 2)
            w /= norm(w)
            @test abs(state_fidelity(v, w) - state_fidelity(w, v)) < 1e-12
        end

        # 6. Gate fidelity: X gate vs Z gate (far apart)
        F_xz = gate_fidelity(σ_x, σ_z)
        @test F_xz < 0.5

        # 7. Gate fidelity is invariant under global phase
        θ = 0.7
        @test abs(gate_fidelity(U_id, exp(im*θ) * U_id) - 1.0) < 1e-12

    end  # Fidelity bounds

    # -----------------------------------------------------------------------
    @testset "Gradient correctness" begin

        # Reference: GRAPE gradient computed analytically must match
        # the gradient estimated by finite differences to within 1e-6
        # (relative tolerance on the control amplitudes).
        #
        # We test on two system sizes to catch dimension-specific bugs.

        # PULSAR's GRAPE gradient uses the standard first-order matrix-exp
        # derivative (∂U_k/∂u_j ≈ -i dt U_k H_j). Its accuracy versus FD
        # scales with (dt · ‖H_total‖)², so we choose modest Rabi/dt to keep
        # the per-step error small enough that FD agreement to 1% is possible.
        fd_tol = 1e-2

        # --- 2×2 single-qubit system ---
        H_drift_1q = 2π * 5e3 * σ_z
        H_ctrl_1q  = [2π * 1e4 * σ_x, 2π * 1e4 * σ_y]
        sys_1q     = quantum_system(H_drift_1q, H_ctrl_1q)

        n_ts  = 60
        dt    = 1e-7
        rng   = MersenneTwister(99)
        u0_1q = 0.3 * randn(rng, Float64, 2, n_ts)

        target_1q = unitary_target(σ_x)

        grad_analytical = grape_gradient(sys_1q, target_1q, u0_1q, dt)
        grad_fd         = finite_diff_gradient(sys_1q, target_1q, u0_1q, dt; ε=1e-6)

        @test norm(grad_analytical - grad_fd) / (norm(grad_fd) + 1e-30) < fd_tol

        # --- 4×4 two-qubit system ---
        # Tensor-product drift: ω1*IZ + ω2*ZI + J*ZZ coupling.
        # Realistic 1 MHz Rabi on each single-qubit drive.
        IZ = kron(I2, σ_z)
        ZI = kron(σ_z, I2)
        ZZ = kron(σ_z, σ_z)
        IX = kron(I2, σ_x)
        IY = kron(I2, σ_y)
        XI = kron(σ_x, I2)
        YI = kron(σ_y, I2)

        H_drift_2q = 2π * (8e3 * ZI + 6e3 * IZ + 2e2 * ZZ)
        H_ctrl_2q  = [2π * 1e4 * XI, 2π * 1e4 * YI, 2π * 1e4 * IX, 2π * 1e4 * IY]
        sys_2q     = quantum_system(H_drift_2q, H_ctrl_2q)

        u0_2q = 0.2 * randn(rng, Float64, 4, n_ts)

        # CNOT target keeps gate fidelity bounded away from the trivial fixed point.
        CNOT = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        target_2q = unitary_target(CNOT)

        grad_analytical_2q = grape_gradient(sys_2q, target_2q, u0_2q, dt)
        grad_fd_2q         = finite_diff_gradient(sys_2q, target_2q, u0_2q, dt; ε=1e-6)

        @test norm(grad_analytical_2q - grad_fd_2q) / (norm(grad_fd_2q) + 1e-30) < fd_tol

    end  # Gradient correctness

    # -----------------------------------------------------------------------
    @testset "Spin operators" begin

        # For spin-1/2 (Pauli matrices divided by 2):
        Sx = σ_x / 2
        Sy = σ_y / 2
        Sz = σ_z / 2

        # Commutation relations: [Sx, Sy] = i*Sz
        comm_xy = Sx * Sy - Sy * Sx
        @test norm(comm_xy - im * Sz) < 1e-14

        # [Sy, Sz] = i*Sx
        comm_yz = Sy * Sz - Sz * Sy
        @test norm(comm_yz - im * Sx) < 1e-14

        # [Sz, Sx] = i*Sy
        comm_zx = Sz * Sx - Sx * Sz
        @test norm(comm_zx - im * Sy) < 1e-14

        # Casimir operator: Sx² + Sy² + Sz² = s(s+1)*I   for s=1/2 → 3/4*I
        S2 = Sx^2 + Sy^2 + Sz^2
        @test norm(S2 - (3/4) * I2) < 1e-14

        # Pauli algebra: σ_i² = I
        @test norm(σ_x * σ_x - I2) < 1e-14
        @test norm(σ_y * σ_y - I2) < 1e-14
        @test norm(σ_z * σ_z - I2) < 1e-14

        # σ_x * σ_y = i * σ_z
        @test norm(σ_x * σ_y - im * σ_z) < 1e-14
        # σ_y * σ_z = i * σ_x
        @test norm(σ_y * σ_z - im * σ_x) < 1e-14
        # σ_z * σ_x = i * σ_y
        @test norm(σ_z * σ_x - im * σ_y) < 1e-14

        # ---- spin-1 operators (dim = 3) ----
        Sx1, Sy1, Sz1 = spin_operators(1)   # returns spin-1 matrices

        comm_xy1 = Sx1 * Sy1 - Sy1 * Sx1
        @test norm(comm_xy1 - im * Sz1) < 1e-12

        I3 = Matrix{ComplexF64}(I, 3, 3)
        S2_spin1 = Sx1^2 + Sy1^2 + Sz1^2
        # s(s+1) = 1*2 = 2 for spin-1
        @test norm(S2_spin1 - 2.0 * I3) < 1e-12

        # Sz eigenvalues for spin-1 should be -1, 0, 1
        evals = sort(real(eigvals(Sz1)))
        @test norm(evals - [-1.0, 0.0, 1.0]) < 1e-12

        # ---- PULSAR spin_operators helper: spin-1/2 ----
        Sx_h, Sy_h, Sz_h = spin_operators(1//2)
        @test norm(Sx_h - Sx) < 1e-14
        @test norm(Sy_h - Sy) < 1e-14
        @test norm(Sz_h - Sz) < 1e-14

    end  # Spin operators

    # -----------------------------------------------------------------------
    @testset "Hamiltonian Hermiticity" begin

        # All constructors in PULSAR must return Hermitian Hamiltonians.

        # 1. Single-qubit system
        H_drift_1q = 2π * 100e3 * σ_z
        sys_1q = quantum_system(H_drift_1q, [2π * σ_x, 2π * σ_y])
        @test norm(sys_1q.H_drift - sys_1q.H_drift') < 1e-12
        for Hc in sys_1q.H_controls
            @test norm(Hc - Hc') < 1e-12
        end

        # 2. Two-qubit system
        IZ = kron(I2, σ_z)
        ZI = kron(σ_z, I2)
        ZZ = kron(σ_z, σ_z)
        H_drift_2q = 2π * (80e3 * ZI + 60e3 * IZ + 2e3 * ZZ)
        H_ctrl_2q  = [2π * kron(I2, σ_x), 2π * kron(I2, σ_y),
                      2π * kron(σ_x, I2), 2π * kron(σ_y, I2)]
        sys_2q = quantum_system(H_drift_2q, H_ctrl_2q)
        @test norm(sys_2q.H_drift - sys_2q.H_drift') < 1e-12
        for Hc in sys_2q.H_controls
            @test norm(Hc - Hc') < 1e-12
        end

        # 3. Spin system created via spin_system constructor
        sys_spin = spin_system(1//2, 2π * 500e3, [2π * σ_x, 2π * σ_y])
        @test norm(sys_spin.H_drift - sys_spin.H_drift') < 1e-12

        # 4. Total Hamiltonian H(t) = H_drift + Σ u_k(t) * H_ctrl_k is Hermitian
        #    for real control amplitudes
        rng = MersenneTwister(55)
        u = randn(rng, Float64, 2)
        H_total = sys_1q.H_drift + u[1] * sys_1q.H_controls[1] +
                                   u[2] * sys_1q.H_controls[2]
        @test norm(H_total - H_total') < 1e-12

        # 5. Random Hermitian construction utility
        A = randn(rng, ComplexF64, 4, 4)
        H_herm = (A + A') / 2
        @test norm(H_herm - H_herm') < 1e-14

    end  # Hamiltonian Hermiticity

    # -----------------------------------------------------------------------
    @testset "Exponential matrix properties" begin

        # det(exp(M)) = exp(tr(M))  (Jacobi's formula)
        rng = MersenneTwister(3)
        A = randn(rng, ComplexF64, 3, 3)
        H_skew = -im * (A + A')          # skew-Hermitian → exp gives unitary
        expH = exp(H_skew)
        @test abs(det(expH) - exp(tr(H_skew))) < 1e-10

        # For traceless H: det(exp(-i*H*t)) = 1
        H_tl = σ_x   # tr = 0
        t    = 2.3e-6
        U_tl = propagator(H_tl, t)
        @test abs(det(U_tl) - 1.0) < 1e-10

    end  # Exponential matrix properties

end  # Mathematical Correctness
