# Application/QuantumComputing/NoiseModels/NonMarkovian.jl
# Non-Markovian noise models via the filter function formalism.
#
# Non-Markovian noise has a finite correlation time and cannot be described by
# simple Lindblad operators.  The filter function (FF) formalism computes the
# dephasing induced by a classical noise spectrum S(ω) acting on a control
# Hamiltonian:
#
#   χ = (1/π) ∫₀^∞ dω S(ω) |F(ω)|² / ω²
#
# where F(ω) is the filter function of the pulse, determined by the time-domain
# propagators.  A good dynamical-decoupling (DD) sequence shapes F(ω) to have
# a zero or small overlap with S(ω).
#
# Provides:
#   NoiseSpectrum              — struct: noise PSD on a frequency grid
#   compute_filter_function    — compute F(ω) from a sequence of propagators
#   filter_function_infidelity — compute χ from F(ω) and S(ω)
#   optimcon_ff                — GRAPE with filter-function penalty
#   pink_noise_spectrum        — 1/f noise PSD constructor
#   white_noise_spectrum       — white noise PSD constructor
#   ohmic_noise_spectrum       — Ohmic bath PSD constructor

using LinearAlgebra

# ============================================================================
# NoiseSpectrum
# ============================================================================

"""
    NoiseSpectrum

Container for a one-sided noise power spectral density S(ω) sampled on a
regular frequency grid.

# Fields
- `omega`       — Vector{Float64} angular frequency samples (rad/s), positive, uniform spacing
- `S`           — Vector{Float64} one-sided PSD values S(ω_k) (rad²/s or 1/s)
- `dω`          — Float64 frequency resolution (rad/s)
- `description` — String: noise model label
"""
struct NoiseSpectrum
    omega       :: Vector{Float64}
    S           :: Vector{Float64}
    dω          :: Float64
    description :: String
end

"""
    pink_noise_spectrum(omega_min, omega_max, n_freq; A=1.0) -> NoiseSpectrum

Construct a 1/f (pink) noise power spectral density:

    S(ω) = A / ω

# Arguments
- `omega_min`, `omega_max` — frequency range (rad/s)
- `n_freq`                 — number of frequency points
- `A`                      — spectral weight prefactor (default 1.0)
"""
function pink_noise_spectrum(omega_min :: Float64,
                               omega_max :: Float64,
                               n_freq    :: Int;
                               A         :: Float64 = 1.0)::NoiseSpectrum
    ω  = collect(range(omega_min, omega_max; length=n_freq))
    S  = A ./ ω
    dω = (omega_max - omega_min) / (n_freq - 1)
    return NoiseSpectrum(ω, S, dω, "1/f noise (A=$A)")
end

"""
    white_noise_spectrum(omega_min, omega_max, n_freq; S0=1.0) -> NoiseSpectrum

Construct a flat (white) noise PSD: S(ω) = S₀.
"""
function white_noise_spectrum(omega_min :: Float64,
                                omega_max :: Float64,
                                n_freq    :: Int;
                                S0        :: Float64 = 1.0)::NoiseSpectrum
    ω  = collect(range(omega_min, omega_max; length=n_freq))
    S  = fill(S0, n_freq)
    dω = (omega_max - omega_min) / (n_freq - 1)
    return NoiseSpectrum(ω, S, dω, "White noise (S0=$S0)")
end

"""
    ohmic_noise_spectrum(omega_min, omega_max, n_freq; A=1.0, omega_c=1.0) -> NoiseSpectrum

Ohmic bath spectral function: S(ω) = A × ω × exp(−ω/ωc).
"""
function ohmic_noise_spectrum(omega_min :: Float64,
                                omega_max :: Float64,
                                n_freq    :: Int;
                                A         :: Float64 = 1.0,
                                omega_c   :: Float64 = 1.0)::NoiseSpectrum
    ω  = collect(range(omega_min, omega_max; length=n_freq))
    S  = A .* ω .* exp.(.- ω ./ omega_c)
    dω = (omega_max - omega_min) / (n_freq - 1)
    return NoiseSpectrum(ω, S, dω, "Ohmic noise (A=$A, ωc=$omega_c)")
end

# ============================================================================
# Filter function computation
# ============================================================================

"""
    compute_filter_function(Us, H_noise, dt, omega) -> Vector{ComplexF64}

Compute the filter function F(ω) for a pulse sequence given as a set of
step propagators `Us` and a noise coupling operator `H_noise`.

The filter function is:

    F(ω) = Σ_{k=1}^{N} Tr[ H_noise · Ũ_k(ω) ]

where Ũ_k(ω) is the modulation function in the toggling frame:

    Ũ_k(ω) = exp(iω t_k) P_k† H_noise P_k × dt

and P_k = U_{k-1} ⋯ U_1 is the cumulative propagator.

This is the single-axis dephasing filter function (Biercuk et al. 2011).

# Arguments
- `Us`      — Array{ComplexF64,3} size (n_t, dim, dim): step propagators
- `H_noise` — Matrix{ComplexF64} noise coupling operator (dephasing axis)
- `dt`      — Float64 time step (s)
- `omega`   — AbstractVector{Float64} frequencies (rad/s) to evaluate F

# Returns
`Vector{ComplexF64}` of length n_freq.

# Reference
Biercuk et al., PRA 83, 020305(R) (2011).
"""
function compute_filter_function(Us      :: Array{ComplexF64,3},
                                  H_noise :: Matrix{ComplexF64},
                                  dt      :: Float64,
                                  omega   :: AbstractVector{Float64}
                                  )::Vector{ComplexF64}
    n_t   = size(Us, 1)
    n_freq = length(omega)
    dim   = size(Us, 2)

    P = compute_forward_propagators(Us)

    F = zeros(ComplexF64, n_freq)
    for (m, ω) in enumerate(omega)
        for k in 1:n_t
            t_k     = (k - 0.5) * dt
            Pk      = @view P[k, :, :]
            # Toggling-frame operator: P_k† H_noise P_k
            H_tog   = Pk' * H_noise * Pk
            F[m]   += exp(im * ω * t_k) * tr(H_tog) * dt
        end
    end
    return F
end

"""
    filter_function_infidelity(Us, H_noise, dt, noise_spectrum) -> Float64

Compute the dephasing infidelity χ from a pulse sequence and a noise spectrum:

    χ = (1/π) ∫ dω S(ω) |F(ω)|² / ω²

# Arguments
- `Us`            — Array{ComplexF64,3}: step propagators
- `H_noise`       — Matrix{ComplexF64}: noise coupling operator
- `dt`            — Float64 time step (s)
- `noise_spectrum` — [`NoiseSpectrum`](@ref)

# Returns
Float64 dephasing infidelity χ ≥ 0.
"""
function filter_function_infidelity(Us             :: Array{ComplexF64,3},
                                     H_noise         :: Matrix{ComplexF64},
                                     dt              :: Float64,
                                     noise_spectrum  :: NoiseSpectrum)::Float64
    F = compute_filter_function(Us, H_noise, dt, noise_spectrum.omega)
    return filter_function_penalty(F, noise_spectrum.S, noise_spectrum.dω)
end

# ============================================================================
# Filter-function-penalised optimal control
# ============================================================================

"""
    optimcon_ff(sys::QuantumSystem, target::QuantumTarget, ctrl::ControlSequence,
                H_noise, noise_spectrum;
                config, ff_weight) -> OptimizationResult

Optimal control with a filter-function penalty that minimises dephasing from a
specified noise spectrum.

The total objective is:

    F_total = F_gate − ff_weight × χ

where χ is the filter-function dephasing infidelity.

# Arguments
- `sys`            — `QuantumSystem`
- `target`         — `QuantumTarget`
- `ctrl`           — initial `ControlSequence`
- `H_noise`        — `Matrix{ComplexF64}` noise coupling operator
- `noise_spectrum`  — [`NoiseSpectrum`](@ref)
- `config`         — `GRAPEConfig`
- `ff_weight`      — Float64 penalty weight (default 1.0)

# Returns
`OptimizationResult`
"""
function optimcon_ff(sys            :: QuantumSystem,
                      target         :: QuantumTarget,
                      ctrl           :: ControlSequence,
                      H_noise        :: Matrix{ComplexF64},
                      noise_spectrum  :: NoiseSpectrum;
                      config         :: GRAPEConfig = GRAPEConfig(),
                      ff_weight      :: Float64     = 1.0)::OptimizationResult

    penalty_fn = function(system, c, tgt)
        Us = compute_propagators(system, c)
        return filter_function_infidelity(Us, H_noise, c.dt, noise_spectrum) * ff_weight
    end

    # Numerical gradient via finite differences (analytical FF gradient requires
    # the Jacobian dF/dw which is system-specific; use finite differences here
    # and let the user provide analytical gradient via penalty_grad_fns if desired)
    penalty_grad_fn = function(system, c, tgt)
        eps_fd = 1e-6
        w0  = copy(c.controls)
        G   = zeros(size(w0))
        P0  = penalty_fn(system, c, tgt)
        for j in axes(w0, 1), k in axes(w0, 2)
            w0[j, k] += eps_fd
            c_p = ControlSequence(w0, c.dt, c.total_time, c.n_timesteps)
            G[j, k] = (penalty_fn(system, c_p, tgt) - P0) / eps_fd
            w0[j, k] -= eps_fd
        end
        return G
    end

    return grape_optimize(sys, target, ctrl;
                          penalty_fns      = [penalty_fn],
                          penalty_grad_fns = [penalty_grad_fn],
                          config           = config)
end
