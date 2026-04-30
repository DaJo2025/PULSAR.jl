# ============================================================================
# Gradient/QOC/TGRAPE.jl — Time-optimal GRAPE (tGRAPE)
#
# Treats per-slice durations Δt_k as additional optimization variables alongside
# the control amplitudes w[c, k].  Source: Spinach `kernel/optimcon/tgrape.m`.
#
# Two duration parameterizations are supported:
#
#   :softmax   (T_total fixed)  — Δt_k = T * exp(z_k) / Σ_j exp(z_j)
#                                  enforces Δt_k > 0 and Σ Δt_k = T exactly.
#                                  Use when total pulse time is fixed.
#
#   :softplus  (free total time) — Δt_k = log(1 + exp(z_k))
#                                  enforces Δt_k > 0; total time floats.
#                                  Use to find time-optimal solutions.
#
# Variables are flattened as [vec(w); vec(z)] of length n_c·n_t + n_t and passed
# to the generic `lbfgs_optimize` outer optimizer.  Gradients in the w-block use
# the standard GRAPE formula scaled by the local Δt_k; the z-block gradient uses
# the analytic time derivative
#
#     ∂F/∂Δt_k = 2 Im[ conj(χ) ⟨λ_{k+1}|H_k|ψ_{k+1}⟩ ]   (state target)
#     ∂F/∂Δt_k = 2 Re[ conj(Φ) Tr(Q[k]† (-i H_k) U_k P[k]) / dim ]   (gate target)
#
# chained through the duration parameterization.
# ============================================================================

using LinearAlgebra

# ---------------------------------------------------------------------------
# Duration parameterization helpers
# ---------------------------------------------------------------------------

@inline function _tgrape_softmax_dt!(dt_vec::Vector{Float64}, z::AbstractVector{<:Real},
                                      T_total::Float64)
    n = length(z)
    zmax = maximum(z)
    s = 0.0
    @inbounds for k in 1:n
        dt_vec[k] = exp(z[k] - zmax)
        s += dt_vec[k]
    end
    @inbounds for k in 1:n
        dt_vec[k] = T_total * dt_vec[k] / s
    end
    return dt_vec
end

@inline function _tgrape_softplus_dt!(dt_vec::Vector{Float64}, z::AbstractVector{<:Real})
    @inbounds for k in eachindex(z)
        # softplus, numerically stable
        zk = z[k]
        dt_vec[k] = zk > 0 ? zk + log1p(exp(-zk)) : log1p(exp(zk))
    end
    return dt_vec
end

# Chain rule: convert ∂F/∂Δt → ∂F/∂z for each parameterization.
@inline function _tgrape_chain_softmax!(g_z::AbstractVector{Float64},
                                        g_dt::AbstractVector{Float64},
                                        dt_vec::Vector{Float64},
                                        T_total::Float64)
    inner = 0.0
    @inbounds for k in eachindex(g_dt)
        inner += g_dt[k] * dt_vec[k]
    end
    inner /= T_total
    @inbounds for k in eachindex(g_dt)
        g_z[k] = dt_vec[k] * (g_dt[k] - inner)
    end
    return g_z
end

@inline function _tgrape_chain_softplus!(g_z::AbstractVector{Float64},
                                         g_dt::AbstractVector{Float64},
                                         z::AbstractVector{<:Real})
    @inbounds for k in eachindex(z)
        # σ(z) = 1 / (1 + exp(-z))
        σ = z[k] >= 0 ? 1.0 / (1.0 + exp(-z[k])) : exp(z[k]) / (1.0 + exp(z[k]))
        g_z[k] = g_dt[k] * σ
    end
    return g_z
end

# ---------------------------------------------------------------------------
# Per-step Hamiltonian + propagator (variable Δt_k)
# ---------------------------------------------------------------------------

@inline function _tgrape_step_hamiltonian!(Hk::Matrix{ComplexF64},
                                           system::AbstractQuantumSystem,
                                           w::AbstractMatrix{<:Real},
                                           k::Int)
    n_c = size(w, 1)
    copyto!(Hk, system.H_drift)
    @inbounds for c in 1:n_c
        wck = w[c, k]
        Hc  = system.H_controls[c]
        @inbounds for q in eachindex(Hk)
            Hk[q] += wck * Hc[q]
        end
    end
    return Hk
end

# ---------------------------------------------------------------------------
# tGRAPE state-target fidelity + gradient
# ---------------------------------------------------------------------------

function _tgrape_state_value_and_grad!(g_w::Matrix{Float64},
                                        g_dt::Vector{Float64},
                                        system::AbstractQuantumSystem,
                                        w::Matrix{Float64},
                                        dt_vec::Vector{Float64},
                                        psi_init::Vector{ComplexF64},
                                        psi_target::Vector{ComplexF64})
    n_c, n_t = size(w)
    dim = length(psi_init)

    # Forward states ψ_{k+1} = U_k ψ_k
    psis = Matrix{ComplexF64}(undef, dim, n_t + 1)
    @views copyto!(psis[:, 1], psi_init)

    Us = Array{ComplexF64,3}(undef, n_t, dim, dim)
    Hk = Matrix{ComplexF64}(undef, dim, dim)
    Uk = Matrix{ComplexF64}(undef, dim, dim)

    @inbounds for k in 1:n_t
        _tgrape_step_hamiltonian!(Hk, system, w, k)
        Uk = compute_propagator(Hk, dt_vec[k])
        @views copyto!(Us[k, :, :], Uk)
        @views mul!(psis[:, k + 1], Uk, psis[:, k])
    end

    chi = dot(psi_target, @view psis[:, n_t + 1])
    F   = abs2(chi)

    # Backward costates λ_k = U_k^† λ_{k+1}, with λ_{n_t+1} = ψ_target
    lambdas = Matrix{ComplexF64}(undef, dim, n_t + 1)
    @views copyto!(lambdas[:, n_t + 1], psi_target)
    Uk_dag = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in n_t:-1:1
        @views adjoint!(Uk_dag, Us[k, :, :])
        @views mul!(lambdas[:, k], Uk_dag, lambdas[:, k + 1])
    end

    # Gradients
    Hk_buf = Matrix{ComplexF64}(undef, dim, dim)
    chi_c  = conj(chi)
    @inbounds for k in 1:n_t
        psi_kp1 = @view psis[:, k + 1]
        lam_kp1 = @view lambdas[:, k + 1]

        # ∂F/∂w[c,k] = 2 Δt_k Im[ conj(χ) ⟨λ_{k+1}|H_c|ψ_{k+1}⟩ ]
        for c in 1:n_c
            Hc = system.H_controls[c]
            z  = dot(lam_kp1, Hc, psi_kp1)
            g_w[c, k] = 2.0 * dt_vec[k] * imag(chi_c * z)
        end

        # ∂F/∂Δt_k = 2 Im[ conj(χ) ⟨λ_{k+1}|H_k|ψ_{k+1}⟩ ]
        _tgrape_step_hamiltonian!(Hk_buf, system, w, k)
        z = dot(lam_kp1, Hk_buf, psi_kp1)
        g_dt[k] = 2.0 * imag(chi_c * z)
    end

    return F
end

# ---------------------------------------------------------------------------
# tGRAPE gate-target fidelity + gradient
# ---------------------------------------------------------------------------

function _tgrape_gate_value_and_grad!(g_w::Matrix{Float64},
                                       g_dt::Vector{Float64},
                                       system::AbstractQuantumSystem,
                                       w::Matrix{Float64},
                                       dt_vec::Vector{Float64},
                                       U_target::Matrix{ComplexF64})
    n_c, n_t = size(w)
    dim = size(U_target, 1)

    # Per-step propagators with variable Δt_k
    Us = Array{ComplexF64,3}(undef, n_t, dim, dim)
    Hk = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_t
        _tgrape_step_hamiltonian!(Hk, system, w, k)
        Uk = compute_propagator(Hk, dt_vec[k])
        @views copyto!(Us[k, :, :], Uk)
    end

    # Forward P[k] = U_{k-1}…U_1, P[1] = I, P[n_t+1] = U_total
    P = Array{ComplexF64,3}(undef, n_t + 1, dim, dim)
    @views copyto!(P[1, :, :], Matrix{ComplexF64}(I, dim, dim))
    Acc = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_t
        @views mul!(Acc, Us[k, :, :], P[k, :, :])
        @views copyto!(P[k + 1, :, :], Acc)
    end

    # Backward Q[k] = (U_{n_t}…U_{k+1})†.  Q[n_t+1] = Q[n_t] = I (empty product).
    # Recursion (for k < n_t):
    #   Q[k] = (U_{n_t}…U_{k+1})† = U_{k+1}† (U_{n_t}…U_{k+2})† = U_{k+1}† · Q[k+1]
    # so that the invariant Q[k]† · U_k · P[k] = U_total holds for every k.
    Q = Array{ComplexF64,3}(undef, n_t + 1, dim, dim)
    @views copyto!(Q[n_t + 1, :, :], Matrix{ComplexF64}(I, dim, dim))
    @views copyto!(Q[n_t,     :, :], Matrix{ComplexF64}(I, dim, dim))
    Uk_dag_buf = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in (n_t - 1):-1:1
        @views adjoint!(Uk_dag_buf, Us[k + 1, :, :])
        @views mul!(view(Q, k, :, :), Uk_dag_buf, Q[k + 1, :, :])
    end

    @views U_total = P[n_t + 1, :, :]
    Phi   = tr(U_target' * U_total) / dim
    Phi_c = conj(Phi)
    F     = abs2(Phi)

    # Amplitude gradient ∂F/∂w[c,k] = 2 Δt_k Re[ conj(Φ) Tr(Q[k]† (-i H_c) U_k P[k]) / dim ]
    # Time gradient    ∂F/∂Δt_k    = 2 Re[ conj(Φ) Tr(Q[k]† (-i H_k) U_k P[k]) / dim ]
    Pk      = Matrix{ComplexF64}(undef, dim, dim)
    Qk      = Matrix{ComplexF64}(undef, dim, dim)
    Uk      = Matrix{ComplexF64}(undef, dim, dim)
    Hk_buf  = Matrix{ComplexF64}(undef, dim, dim)
    Tmp1    = Matrix{ComplexF64}(undef, dim, dim)
    Tmp2    = Matrix{ComplexF64}(undef, dim, dim)
    @inbounds for k in 1:n_t
        @views copyto!(Pk, P[k, :, :])
        @views copyto!(Qk, Q[k, :, :])
        @views copyto!(Uk, Us[k, :, :])

        # Tmp1 = U_k P[k]
        mul!(Tmp1, Uk, Pk)

        # Q[k]† Tmp1 = U_total — but we still need the projected sandwich.
        # General form: Tr(Q[k]† X U_k P[k]) where X = (-i H_x).
        # Compute M = U_k P[k] U_target† Q[k]†, then Tr(M X) = Σ M_{ij} X_{ji}.
        mul!(Tmp2, Tmp1, U_target')
        Qk_dag_T = Matrix{ComplexF64}(undef, dim, dim)
        @views adjoint!(Qk_dag_T, Q[k, :, :])
        # Tmp1 = U_k P[k] U_target† Q[k]†
        mul!(Tmp1, Tmp2, Qk_dag_T)

        # Amplitude block
        for c in 1:n_c
            Hc = system.H_controls[c]
            s  = zero(ComplexF64)
            for q in 1:dim, p in 1:dim
                s += Tmp1[p, q] * Hc[q, p]
            end
            inner = -im * s / dim
            g_w[c, k] = 2.0 * dt_vec[k] * real(Phi_c * inner)
        end

        # Time block (uses local H_k, not H_c)
        _tgrape_step_hamiltonian!(Hk_buf, system, w, k)
        s = zero(ComplexF64)
        for q in 1:dim, p in 1:dim
            s += Tmp1[p, q] * Hk_buf[q, p]
        end
        inner = -im * s / dim
        g_dt[k] = 2.0 * real(Phi_c * inner)
    end

    return F
end

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

"""
    tgrape_optimize(system, target, w_init, dt_init;
                    parameterization = :softmax, T_total = nothing,
                    max_iter = 200, tol = 1e-6, memory = 10,
                    w_lower = nothing, w_upper = nothing,
                    verbose = false, callback = nothing)
        → (w_opt, dt_opt, F_opt, stats)

Time-optimal GRAPE (tGRAPE).  Optimises control amplitudes `w` (`n_controls × n_t`)
and per-slice durations `Δt_k` jointly to maximise fidelity to `target`.

# Arguments
- `system`   — `AbstractQuantumSystem` (drift + control Hamiltonians)
- `target`   — `QuantumTarget` (state or unitary)
- `w_init`   — initial amplitudes, shape `[n_controls × n_t]`
- `dt_init`  — initial uniform slice duration (seconds, > 0)

# Keyword arguments
- `parameterization = :softmax` — `:softmax` keeps `Σ Δt_k = T_total` (T fixed);
  `:softplus` allows the total time to vary (free time-optimal).
- `T_total` — total pulse time for `:softmax` mode; defaults to `dt_init * n_t`.
- `max_iter`, `tol`, `memory` — passed through to the underlying `lbfgs_optimize`.
- `w_lower`, `w_upper` — optional element-wise bounds on the amplitude block.
  The duration variables `z` are always unbounded.
- `verbose` — forwards to `lbfgs_optimize`.
- `callback(iter, F; grad, evals)` — called once per outer iteration.

# Returns
A NamedTuple-like 4-tuple `(w_opt, dt_opt, F_opt, stats)`:
- `w_opt :: Matrix{Float64}` — optimal amplitudes `[n_controls × n_t]`
- `dt_opt :: Vector{Float64}` — optimal per-slice durations `[n_t]`
- `F_opt :: Float64` — fidelity at the optimum
- `stats` — NamedTuple with fields `iters`, `evals`, `total_time`, `converged`.

# Notes
The kernel does not modify `ControlSequence`.  After optimisation the caller
typically rebuilds a uniform-`dt` `ControlSequence` from `w_opt` (when
`parameterization = :softmax` and the user wants a fixed-grid pulse) or stores
`(w_opt, dt_opt)` as a non-uniform pulse for export.
"""
function tgrape_optimize(system::AbstractQuantumSystem,
                          target::QuantumTarget,
                          w_init::AbstractMatrix{<:Real},
                          dt_init::Real;
                          parameterization::Symbol = :softmax,
                          T_total::Union{Nothing,Real} = nothing,
                          max_iter::Int = 200,
                          tol::Float64 = 1e-6,
                          memory::Int = 10,
                          w_lower::Union{Nothing,Real} = nothing,
                          w_upper::Union{Nothing,Real} = nothing,
                          verbose::Bool = false,
                          callback = nothing,
                          dt_min::Float64 = 1e-15)

    parameterization in (:softmax, :softplus) ||
        throw(ArgumentError("parameterization must be :softmax or :softplus, got $parameterization"))
    dt_init > 0 || throw(ArgumentError("dt_init must be positive, got $dt_init"))

    n_c, n_t = size(w_init)
    T_fixed  = parameterization === :softmax ?
                Float64(T_total === nothing ? dt_init * n_t : T_total) : 0.0
    parameterization === :softmax && T_fixed > 0 ||
        parameterization === :softplus ||
        throw(ArgumentError("T_total must be positive when parameterization = :softmax"))

    # State or unitary target — pre-extract to avoid per-iter type checks.
    if target.type == "state"
        target.target_state === nothing &&
            throw(ArgumentError("target.type is \"state\" but target.target_state is nothing"))
        psi_t = target.target_state
        nrm   = norm(psi_t)
        nrm < eps(Float64) && throw(ArgumentError("target_state has zero norm"))
        psi_target = psi_t ./ nrm
        if target.initial_state !== nothing
            pi_raw = target.initial_state
            length(pi_raw) == length(psi_target) || throw(DimensionMismatch(
                "target.initial_state length $(length(pi_raw)) ≠ target_state length $(length(psi_target))"))
            ni = norm(pi_raw)
            ni < eps(Float64) && throw(ArgumentError("target.initial_state has zero norm"))
            psi_init = pi_raw ./ ni
        else
            psi_init = psi_target        # legacy convention: target_state = init = target
        end
        is_state   = true
        U_target   = Matrix{ComplexF64}(undef, 0, 0)
    elseif target.type == "unitary"
        target.target_unitary === nothing &&
            throw(ArgumentError("target.type is \"unitary\" but target.target_unitary is nothing"))
        U_target   = target.target_unitary
        is_state   = false
        psi_init   = ComplexF64[]
        psi_target = ComplexF64[]
    else
        throw(ArgumentError("Unknown target type \"$(target.type)\""))
    end

    # Initial z so that softmax/softplus produce uniform Δt_k = dt_init.
    # softmax: any constant z gives uniform; pick zeros.
    # softplus: solve softplus(z) = dt_init  →  z = log(exp(dt_init) - 1).
    z_init = parameterization === :softmax ?
              zeros(n_t) :
              fill(log(expm1(dt_init)), n_t)

    # Variable layout θ = [vec(w); z] (column-major vec of w → length n_c*n_t)
    n_w = n_c * n_t
    n_θ = n_w + n_t
    θ0  = Vector{Float64}(undef, n_θ)
    @views vec(w_init') |> identity   # touch to ensure layout — w stored as [n_c × n_t]
    # We want vec(w) where w[c, k] is contiguous in c first, then k.
    @inbounds for k in 1:n_t, c in 1:n_c
        θ0[(k - 1) * n_c + c] = Float64(w_init[c, k])
    end
    @views copyto!(θ0[n_w + 1 : end], z_init)

    # Bounds (only on amplitude block)
    lb = fill(-Inf, n_θ)
    ub = fill( Inf, n_θ)
    if w_lower !== nothing
        @inbounds for i in 1:n_w
            lb[i] = Float64(w_lower)
        end
    end
    if w_upper !== nothing
        @inbounds for i in 1:n_w
            ub[i] = Float64(w_upper)
        end
    end

    # Scratch (closed over by f / grad!)
    w_buf   = Matrix{Float64}(undef, n_c, n_t)
    g_w     = Matrix{Float64}(undef, n_c, n_t)
    g_dt    = Vector{Float64}(undef, n_t)
    dt_vec  = Vector{Float64}(undef, n_t)

    # Helper to refresh dt_vec and w_buf from θ
    @inline function unpack!(θ)
        @inbounds for k in 1:n_t, c in 1:n_c
            w_buf[c, k] = θ[(k - 1) * n_c + c]
        end
        z = @view θ[n_w + 1 : end]
        if parameterization === :softmax
            _tgrape_softmax_dt!(dt_vec, z, T_fixed)
        else
            _tgrape_softplus_dt!(dt_vec, z)
            @inbounds for k in 1:n_t
                dt_vec[k] = max(dt_vec[k], dt_min)
            end
        end
        return z
    end

    # Objective (we minimise -F)
    function f(θ)
        unpack!(θ)
        F = is_state ?
            _tgrape_state_value_and_grad!(g_w, g_dt, system, w_buf, dt_vec,
                                           psi_init, psi_target) :
            _tgrape_gate_value_and_grad!(g_w, g_dt, system, w_buf, dt_vec, U_target)
        return -F
    end

    function grad!(g_out::AbstractVector, θ)
        z = unpack!(θ)
        F = is_state ?
            _tgrape_state_value_and_grad!(g_w, g_dt, system, w_buf, dt_vec,
                                           psi_init, psi_target) :
            _tgrape_gate_value_and_grad!(g_w, g_dt, system, w_buf, dt_vec, U_target)

        # Pack amplitude block (we are minimising -F, so flip sign)
        @inbounds for k in 1:n_t, c in 1:n_c
            g_out[(k - 1) * n_c + c] = -g_w[c, k]
        end

        # Chain rule for duration block
        g_z_view = @view g_out[n_w + 1 : end]
        if parameterization === :softmax
            _tgrape_chain_softmax!(g_z_view, g_dt, dt_vec, T_fixed)
        else
            _tgrape_chain_softplus!(g_z_view, g_dt, z)
        end
        @inbounds for k in 1:n_t
            g_z_view[k] = -g_z_view[k]
        end
        return g_out
    end

    t_start = time()
    θ_opt, fneg_opt, stats = lbfgs_optimize(f, grad!, θ0;
                                             memory   = memory,
                                             max_iter = max_iter,
                                             tol      = tol,
                                             lower    = lb,
                                             upper    = ub,
                                             verbose  = verbose,
                                             callback = callback)
    elapsed = time() - t_start

    # Unpack final θ
    unpack!(θ_opt)
    w_opt = copy(w_buf)
    dt_opt = copy(dt_vec)
    F_opt  = -fneg_opt

    out_stats = (
        iters       = stats.iters,
        evals       = stats.evals,
        total_time  = elapsed,
        converged   = stats.converged,
        T_total_opt = sum(dt_opt),
    )
    return w_opt, dt_opt, F_opt, out_stats
end
