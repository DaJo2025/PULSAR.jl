# ============================================================================
# Gradient/QOC/CRAB.jl — Chopped Random Basis (CRAB)
#
# Caneva, Calarco, Montangero, Phys. Rev. A 84, 022326 (2011).  Parametrise the
# control waveform as a baseline guess plus a small set of randomly drawn
# Fourier modes, and optimise the (sin, cos) coefficients with a derivative-free
# outer optimiser:
#
#     w_c(t_k) = w0_c(t_k) + Σ_{n=1}^{n_modes} [ A_{n,c} sin(ω_n t_k) + B_{n,c} cos(ω_n t_k) ]
#
# with ω_n = 2π (n + r_n) / T,  r_n ~ U(-0.5, 0.5), drawn once and held fixed.
# The basis is the "Reserved" CRABRandomParam in `Optimization/Parameterization.jl`;
# this wrapper materialises it inline and pipes the resulting flat parameter
# vector through Nelder-Mead or CMA-ES.
#
# Despite the file path (`Gradient/QOC/`), the outer optimiser is derivative-free
# — placement here mirrors GOAT/GROUP for catalog symmetry.
# ============================================================================

using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

@inline function _crab_random_frequencies(n_modes::Int, T::Real, rng::AbstractRNG)
    ω = Vector{Float64}(undef, n_modes)
    @inbounds for n in 1:n_modes
        r    = rand(rng) - 0.5                 # U(-0.5, 0.5)
        ω[n] = 2π * (n + r) / Float64(T)
    end
    return ω
end

@inline function _crab_basis_matrix(ω::Vector{Float64}, t::Vector{Float64})
    n_t  = length(t)
    n_m  = length(ω)
    Φ    = Matrix{Float64}(undef, n_t, 2 * n_m)
    @inbounds for n in 1:n_m, k in 1:n_t
        s = sin(ω[n] * t[k])
        c = cos(ω[n] * t[k])
        Φ[k, n]         = s
        Φ[k, n_m + n]   = c
    end
    return Φ                                    # n_t × (2 n_modes)
end

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

"""
    crab_optimize(f_kernel, w0, dt;
                  n_modes = 10, outer = :nelder_mead, rng_seed = 0,
                  max_iter = 500, max_evals = 5_000,
                  amplitude0 = 0.0, kwargs...)
        → (w_opt, coeffs_opt, F_opt, stats)

Canonical CRAB optimiser.  Reconstructs the control waveform as

    w_c(t_k) = w0[c, k] + Σ_n [ A_{n,c} sin(ω_n t_k) + B_{n,c} cos(ω_n t_k) ]

with random Fourier frequencies `ω_n = 2π (n + r_n) / T` (`r_n` drawn once from
`U(-0.5, 0.5)`, frozen for the run).  The outer optimiser searches the
`2 · n_modes · n_controls`-dimensional coefficient vector; the gradient through
the basis is never required.

# Arguments
- `f_kernel(w) -> Real` — fidelity *to maximise* given a `[n_controls × n_t]`
  waveform.
- `w0` — baseline waveform, shape `[n_controls × n_t]` (held fixed; the CRAB
  modulation is added on top).
- `dt` — slice duration in seconds.

# Keyword arguments
- `n_modes` — number of random Fourier modes (default 10).
- `outer` — `:nelder_mead` or `:cmaes` (or `:cma_es`).
- `rng_seed` — seed for the basis frequency draw (reproducibility).
- `max_iter`, `max_evals` — forwarded to the outer optimiser.
- `amplitude0` — initial spread of the coefficient guess (zeros if 0).
- Additional `kwargs...` are forwarded to the outer optimiser.

# Returns
A 4-tuple `(w_opt, coeffs_opt, F_opt, stats)`:
- `w_opt :: Matrix{Float64}` — optimal waveform `[n_controls × n_t]`
- `coeffs_opt :: Vector{Float64}` — optimal CRAB coefficients
- `F_opt :: Float64` — fidelity at the optimum
- `stats` — outer-optimiser stats NamedTuple, with extra fields `frequencies`
  (the random ω drawn) and `n_modes`.
"""
function crab_optimize(f_kernel::Function,
                        w0::AbstractMatrix{<:Real},
                        dt::Real;
                        n_modes::Int = 10,
                        outer::Symbol = :nelder_mead,
                        rng_seed::Integer = 0,
                        max_iter::Int = 500,
                        max_evals::Int = 5_000,
                        amplitude0::Float64 = 0.0,
                        kwargs...)
    n_modes > 0 || throw(ArgumentError("n_modes must be > 0, got $n_modes"))
    dt > 0      || throw(ArgumentError("dt must be > 0, got $dt"))
    outer in (:nelder_mead, :cmaes, :cma_es) ||
        throw(ArgumentError("outer must be :nelder_mead or :cmaes, got $outer"))

    n_c, n_t = size(w0)
    T   = dt * n_t
    rng = MersenneTwister(Int(rng_seed))
    ω   = _crab_random_frequencies(n_modes, T, rng)
    t   = collect(((1:n_t) .- 0.5) .* dt)         # cell-centre times
    Φ   = _crab_basis_matrix(ω, t)                # n_t × 2n_modes

    n_coeffs = 2 * n_modes * n_c
    w0_mat   = Matrix{Float64}(w0)
    w_buf    = Matrix{Float64}(undef, n_c, n_t)

    # Reconstruct waveform from a flat coefficient vector laid out as
    # [n_c × 2n_modes], C-storage so coeffs[:, c] are contiguous per control.
    function _reconstruct!(w::AbstractMatrix, θ::AbstractVector)
        # θ packed control-major: for c in 1:n_c, the 2n_modes coeffs are
        # θ[(c-1)*2n_modes + 1 : c*2n_modes].
        @inbounds for c in 1:n_c
            offset = (c - 1) * 2 * n_modes
            for k in 1:n_t
                acc = w0_mat[c, k]
                for j in 1:(2 * n_modes)
                    acc += θ[offset + j] * Φ[k, j]
                end
                w[c, k] = acc
            end
        end
        return w
    end

    # Outer-loop objective: the outer optimisers minimise, so negate.
    function f_outer(θ)
        _reconstruct!(w_buf, θ)
        return -Float64(f_kernel(w_buf))
    end

    θ0 = amplitude0 == 0.0 ?
          zeros(n_coeffs) :
          amplitude0 .* (2 .* rand(MersenneTwister(Int(rng_seed) + 1), n_coeffs) .- 1)

    θ_opt, fneg_opt, stats =
        if outer === :nelder_mead
            nelder_mead_optimize(f_outer, θ0;
                                  max_iters = max_iter, max_evals = max_evals,
                                  kwargs...)
        else
            cmaes_optimize(f_outer, θ0;
                            max_iters = max_iter, max_evals = max_evals,
                            kwargs...)
        end

    _reconstruct!(w_buf, θ_opt)
    w_opt   = copy(w_buf)
    F_opt   = -fneg_opt

    out_stats = (
        evals       = haskey(stats, :evals) ? stats.evals : -1,
        iters       = haskey(stats, :iters) ? stats.iters : -1,
        converged   = haskey(stats, :converged) ? stats.converged : false,
        frequencies = ω,
        n_modes     = n_modes,
    )
    return w_opt, θ_opt, F_opt, out_stats
end
