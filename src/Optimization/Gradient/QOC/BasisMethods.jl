# ============================================================================
# Gradient/QOC/BasisMethods.jl
# Basis-parametrized gradient methods for quantum optimal control
#
# group_optimize  — GROUP: Gradient optimization over a user-supplied basis
# goat_optimize   — GOAT: Gradient Optimization of Analytic conTrols
#                         (Fourier basis automatically constructed from N_ts, N_freq, dt)
#
# Reference:
#   Machnes et al., "Comparing, Optimising and Benchmarking Quantum-Control
#   Algorithms in a Unifying Programming Framework", PRA 84 (2011).
#   Sørensen et al., "QEngine: A C++ library for quantum optimal control",
#   Comput. Phys. Commun. 243 (2019).
# ============================================================================

using LinearAlgebra

# ---------------------------------------------------------------------------
# Internal: L-BFGS driver (self-contained; same algorithm as QuasiNewton.jl)
# ---------------------------------------------------------------------------

function _bm_lbfgs_dir!(d, g, S, Y, ρ_list)
    m = length(S)
    m == 0 && (@. d = -g; return)
    q     = copy(g)
    α_arr = zeros(m)
    for i in m:-1:1
        α_arr[i] = ρ_list[i] * dot(S[i], q)
        @. q -= α_arr[i] * Y[i]
    end
    γ = dot(S[m], Y[m]) / max(dot(Y[m], Y[m]), 1e-30)
    @. d = γ * q
    for i in 1:m
        β = ρ_list[i] * dot(Y[i], d)
        @. d += (α_arr[i] - β) * S[i]
    end
    @. d = -d
end

# Strong-Wolfe line search lives in Gradient/_LineSearch.jl.
# BasisMethods callers use the simple bracket (α_max=10.0, max_iter=40,
# zoom_iter=25, zoom_eps=0.0 ⇒ no early zoom break).

@inline function _bm_ls!(θ_t, g_buf, f, grad!, θ, d, g0, f0)
    wolfe_line_search!(θ_t, g_buf, f, grad!, θ, d, g0, f0;
                       α_max=10.0, max_iter=40,
                       zoom_iter=25, zoom_eps=0.0,
                       two_point_bracket=false)
end

function _bm_lbfgsb!(f, grad!, θ, lb, ub, memory, max_iter, tol, verbose,
                      print_interval, label, callback=nothing)
    n      = length(θ)
    g      = zeros(n); g_new = zeros(n); d = zeros(n)
    θ_t    = similar(θ)                         # wolfe trial buffer (hoisted)
    g_ls   = zeros(n)                           # wolfe gradient scratch (hoisted)
    S      = Vector{Vector{Float64}}()
    Y      = Vector{Vector{Float64}}()
    ρ_list = Float64[]

    grad!(g, θ);  n_evals = 1
    f_cur   = f(θ);  n_evals += 1
    θ_best  = copy(θ);  f_best = f_cur
    converged = false

    for iter in 1:max_iter
        pg = norm(θ .- clamp.(θ .- g, lb, ub))
        pg < tol && (converged = true; break)

        _bm_lbfgs_dir!(d, g, S, Y, ρ_list)
        for i in 1:n
            if (θ[i] <= lb[i]+1e-12 && d[i]<0.0) || (θ[i] >= ub[i]-1e-12 && d[i]>0.0)
                d[i] = 0.0
            end
        end
        norm(d) < 1e-14 && break

        α, f_new = _bm_ls!(θ_t, g_ls, f, grad!, θ, d, g, f_cur)
        n_evals += 2
        s = α .* d                 # fresh alloc; moved into S below
        θ .+= s;  @. θ = clamp(θ, lb, ub)
        f_cur = f_new
        grad!(g_new, θ);  n_evals += 1
        y = g_new .- g;  sy = dot(s, y)     # fresh; moved into Y below
        if sy > 1e-14 * dot(s, s)
            push!(S, s); push!(Y, y); push!(ρ_list, 1.0/sy)
            length(S) > memory && (popfirst!(S); popfirst!(Y); popfirst!(ρ_list))
        end
        @. g = g_new
        if f_cur < f_best;  f_best = f_cur;  θ_best .= θ;  end

        verbose && iter % print_interval == 0 &&
            @printf("  %s iter %4d  F=%.6f  |∇P|=%.3e\n", label, iter, -f_cur, pg)
        isnothing(callback) || callback(iter, -f_cur; grad=pg, evals=n_evals)
    end
    return θ_best, f_best, n_evals, converged
end

# ---------------------------------------------------------------------------
# GROUP
# ---------------------------------------------------------------------------

"""
    group_optimize(f, grad!, basis, c0; lower_c, upper_c, memory, max_iter, tol,
                   lower_θ, upper_θ, verbose) → (θ_opt, f_opt, stats)

GROUP (Gradient-based Optimal control Using a restricted Parameter space):
parametrises the full control vector as θ = basis * c and optimises over
the coefficient vector c using L-BFGS-B.

Arguments:
- `f(θ)` — objective in the full control space (minimisation)
- `grad!(g, θ)` — gradient in the full control space
- `basis` — n_θ × n_c Matrix{Float64}; columns are basis functions
- `c0` — initial coefficients (length n_c)

The reduced gradient is: ∇_c F = basis' * ∇_θ F.
Box constraints on θ are propagated to soft bounds on c via basis norms.
"""
function group_optimize(
    f       :: Function,
    grad!   :: Function,
    basis   :: AbstractMatrix{<:Real},
    c0      :: AbstractVector{<:Real};
    lower_c :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper_c :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    lower_θ :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper_θ :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    memory  :: Int     = 10,
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    verbose :: Bool    = true,
    print_interval :: Int = 50,
    callback = nothing,
)
    n_θ, n_c = size(basis)
    B    = Float64.(basis)
    c    = float.(copy(c0))

    # Bounds for c
    if lower_c !== nothing
        lb_c = Float64.(lower_c)
    elseif lower_θ !== nothing
        # Approximate: bound c by ||basis||_∞ × max constraint
        lb_c = fill(minimum(lower_θ) / maximum(abs.(B) .+ 1e-30), n_c)
    else
        lb_c = fill(-Inf, n_c)
    end

    if upper_c !== nothing
        ub_c = Float64.(upper_c)
    elseif upper_θ !== nothing
        ub_c = fill(maximum(upper_θ) / maximum(abs.(B) .+ 1e-30), n_c)
    else
        ub_c = fill( Inf, n_c)
    end

    # Buffers
    θ_buf = zeros(n_θ)
    g_θ   = zeros(n_θ)
    g_c   = zeros(n_c)

    # Reduced objective and gradient
    f_c(cv) = begin
        mul!(θ_buf, B, cv)
        f(θ_buf)
    end
    function grad_c!(gc, cv)
        mul!(θ_buf, B, cv)
        grad!(g_θ, θ_buf)
        mul!(gc, B', g_θ)
    end

    θ_c_best, f_best, n_evals, converged =
        _bm_lbfgsb!(f_c, grad_c!, c, lb_c, ub_c, memory, max_iter, tol,
                     verbose, print_interval, "group", callback)

    # Recover full control
    mul!(θ_buf, B, θ_c_best)
    if lower_θ !== nothing || upper_θ !== nothing
        lo = lower_θ === nothing ? fill(-Inf, n_θ) : Float64.(lower_θ)
        hi = upper_θ === nothing ? fill( Inf, n_θ) : Float64.(upper_θ)
        @. θ_buf = clamp(θ_buf, lo, hi)
    end

    verbose &&
        @printf("  group done  F=%.6f  evals=%d  converged=%s\n",
                -f_best, n_evals, converged)

    stats = (evals=n_evals, iters=max_iter, converged=converged)
    return copy(θ_buf), f_best, stats
end

# ---------------------------------------------------------------------------
# GOAT
# ---------------------------------------------------------------------------

"""
    goat_optimize(f, grad!, N_ts, N_freq; n_ctrl, dt, lower_θ, upper_θ,
                  memory, max_iter, tol, verbose) → (θ_opt, f_opt, stats)

GOAT (Gradient Optimization of Analytic conTrols):
the control waveform is expanded in a truncated Fourier basis with N_freq
positive-frequency harmonics.  The basis is:

  B[:, 1]       = 1/√N_ts   (DC component)
  B[:, 2k]      = √(2/N_ts) * cos(2π k t[i] / T)
  B[:, 2k+1]    = √(2/N_ts) * sin(2π k t[i] / T)

for k = 1, …, N_freq, giving n_c = 2·N_freq + 1 coefficients per control channel.
Each control channel gets its own independent basis expansion.

Arguments:
- `f(θ)` — objective in the full θ = vec(waveform) space (n_ctrl × N_ts, column-major)
- `grad!(g, θ)` — gradient w.r.t. full waveform
- `N_ts` — number of time steps
- `N_freq` — number of Fourier harmonics (bandwidth)
- `n_ctrl` — number of control channels (default 2)
- `dt` — time step in seconds (default 2e-6)
"""
function goat_optimize(
    f       :: Function,
    grad!   :: Function,
    N_ts    :: Int,
    N_freq  :: Int;
    n_ctrl  :: Int     = 2,
    dt      :: Float64 = 2e-6,
    lower_θ :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    upper_θ :: Union{Nothing,AbstractVector{<:Real}} = nothing,
    memory  :: Int     = 10,
    max_iter:: Int     = 500,
    tol     :: Float64 = 1e-6,
    verbose :: Bool    = true,
    print_interval :: Int = 50,
    callback = nothing,
)
    T    = N_ts * dt
    t    = [(i - 0.5) * dt for i in 1:N_ts]    # midpoints
    n_c1 = 2 * N_freq + 1                       # coefficients per channel
    n_θ  = n_ctrl * N_ts                        # full waveform size
    n_c  = n_ctrl * n_c1                        # total coefficients

    # Build per-channel Fourier basis (N_ts × n_c1)
    B1   = zeros(N_ts, n_c1)
    B1[:, 1] .= 1.0 / sqrt(N_ts)
    for k in 1:N_freq
        @. B1[:, 2k]   = sqrt(2.0 / N_ts) * cos(2π * k * t / T)
        @. B1[:, 2k+1] = sqrt(2.0 / N_ts) * sin(2π * k * t / T)
    end

    # Block-diagonal basis for all channels: n_θ × n_c
    # θ layout: [ch1_step1, ch2_step1, ..., ch_n_ctrl_step1, ch1_step2, ...]
    # We use column-major layout: θ = vec(W) where W is n_ctrl × N_ts
    # So θ[i + (k-1)*n_ctrl] = W[i, k]
    # Basis: B[i + (k-1)*n_ctrl, j + (ch-1)*n_c1] = B1[k, j] for ch == i, else 0
    B = zeros(n_θ, n_c)
    for ch in 1:n_ctrl
        for k in 1:N_ts
            row = ch + (k - 1) * n_ctrl
            for j in 1:n_c1
                col = j + (ch - 1) * n_c1
                B[row, col] = B1[k, j]
            end
        end
    end

    # Initial coefficients: small random
    c0 = 0.01 .* randn(n_c)

    if verbose
        @printf("GOAT: N_ts=%d, N_freq=%d, n_ctrl=%d → %d Fourier coefficients\n",
                N_ts, N_freq, n_ctrl, n_c)
    end

    θ_opt, f_opt, stats = group_optimize(
        f, grad!, B, c0;
        lower_θ=lower_θ, upper_θ=upper_θ,
        memory=memory, max_iter=max_iter, tol=tol,
        verbose=verbose, print_interval=print_interval,
        callback=callback,
    )
    return θ_opt, f_opt, stats
end
