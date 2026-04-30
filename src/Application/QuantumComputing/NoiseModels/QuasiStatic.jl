# Application/QuantumComputing/NoiseModels/QuasiStatic.jl
# Quasi-static noise models for quantum optimal control robustness.
#
# Quasi-static (low-frequency) noise adds a time-independent random offset to
# the Hamiltonian parameters.  The pulse experiences a fixed-but-unknown
# perturbation during its execution.
#
# Strategy: build an ensemble of drift Hamiltonians over a noise distribution
# and average the GRAPE gradient.  This is the standard approach for designing
# robust pulses against 1/f charge noise (superconducting), magnetic field
# fluctuations (spin qubits), laser intensity noise (trapped ions), etc.
#
# Provides:
#   QuasiStaticNoise          — struct: noise parameter specification
#   quasi_static_ensemble     — build drift ensemble from noise model
#   robust_optimcon_qs        — GRAPE over quasi-static ensemble

using LinearAlgebra
using Random
using Statistics

# ============================================================================
# QuasiStaticNoise
# ============================================================================

"""
    QuasiStaticNoise

Specification of a quasi-static noise source affecting one or more Hamiltonian
parameters.

# Fields
- `parameter`   — Symbol: which parameter is noisy (e.g. `:freq`, `:coupling`,
  `:amplitude`)
- `distribution` — Symbol: `:gaussian`, `:uniform`, or `:custom`
- `sigma`        — Float64 standard deviation (for `:gaussian`; half-width for
  `:uniform`), in natural units of the parameter (Hz for frequencies, etc.)
- `n_samples`    — Int: number of ensemble members (default 9)
- `samples`      — optional Vector{Float64} of explicit sample values
  (overrides `distribution` and `sigma` when provided)
"""
struct QuasiStaticNoise
    parameter    :: Symbol
    distribution :: Symbol
    sigma        :: Float64
    n_samples    :: Int
    samples      :: Vector{Float64}
end

"""
    QuasiStaticNoise(parameter, sigma; distribution=:gaussian, n_samples=9,
                     samples=Float64[]) -> QuasiStaticNoise

Construct a [`QuasiStaticNoise`](@ref) specification.

# Example
```julia
# Gaussian frequency noise, σ = 100 kHz, 7 samples
noise = QuasiStaticNoise(:freq, 100e3; n_samples=7)
```
"""
function QuasiStaticNoise(parameter    :: Symbol,
                            sigma        :: Float64;
                            distribution :: Symbol        = :gaussian,
                            n_samples    :: Int            = 9,
                            samples      :: Vector{Float64}= Float64[])::QuasiStaticNoise
    @assert distribution ∈ (:gaussian, :uniform, :custom) ||
        !isempty(samples) "distribution must be :gaussian, :uniform, or :custom"
    @assert n_samples > 0 "n_samples must be positive"
    return QuasiStaticNoise(parameter, distribution, sigma, n_samples, samples)
end

# ============================================================================
# Ensemble builders
# ============================================================================

"""
    _qs_samples(noise::QuasiStaticNoise; rng=GLOBAL_RNG) -> Vector{Float64}

Draw or return noise sample values.
"""
function _qs_samples(noise :: QuasiStaticNoise;
                      rng   :: AbstractRNG = Random.GLOBAL_RNG)::Vector{Float64}
    !isempty(noise.samples) && return noise.samples
    n = noise.n_samples
    if noise.distribution == :gaussian
        # Gauss-Hermite quadrature points (approximate) for symmetric coverage
        # For n samples: evenly spaced z-scores from -2σ to +2σ
        if n == 1
            return [0.0]
        end
        z = range(-2.0, 2.0; length=n)
        return noise.sigma .* collect(z)
    elseif noise.distribution == :uniform
        # Uniform on [−σ, +σ]
        return collect(range(-noise.sigma, noise.sigma; length=n))
    else
        # Custom / random Gaussian
        return noise.sigma .* randn(rng, n)
    end
end

"""
    quasi_static_ensemble(H_drift_fn, noise_sources; rng=GLOBAL_RNG)
    -> Vector{Matrix{ComplexF64}}

Build a vector of drift Hamiltonians by sampling quasi-static noise.

`H_drift_fn(params::Vector{Float64})` is a user-supplied function that
constructs a Hamiltonian from a vector of perturbed parameter values.
`noise_sources` is an ordered `Vector{QuasiStaticNoise}`.

The ensemble covers the full Cartesian product of samples from each noise
source.

# Arguments
- `H_drift_fn`    — function: `params -> Matrix{ComplexF64}`; params[k] is the
  value drawn for noise_sources[k]
- `noise_sources` — Vector{QuasiStaticNoise}
- `rng`           — AbstractRNG (default: Random.GLOBAL_RNG)

# Returns
`Vector{Matrix{ComplexF64}}` ensemble of drift Hamiltonians.

# Example
```julia
H_fn(p) = 2π * p[1] / 2 * σz    # frequency noise
noise = [QuasiStaticNoise(:freq, 50e3; n_samples=5)]
drifts = quasi_static_ensemble(H_fn, noise)
```
"""
function quasi_static_ensemble(H_drift_fn   :: Function,
                                 noise_sources :: Vector{QuasiStaticNoise};
                                 rng           :: AbstractRNG = Random.GLOBAL_RNG
                                 )::Vector{Matrix{ComplexF64}}
    # Collect samples for each noise source
    all_samples = [_qs_samples(ns; rng=rng) for ns in noise_sources]

    # Cartesian product
    drifts = Matrix{ComplexF64}[]
    n_sources = length(noise_sources)

    function _recurse(k, params)
        if k > n_sources
            push!(drifts, H_drift_fn(params))
        else
            for s in all_samples[k]
                _recurse(k + 1, [params; s])
            end
        end
    end
    _recurse(1, Float64[])

    return drifts
end

# ============================================================================
# Robust optimal control over quasi-static ensemble
# ============================================================================

"""
    robust_optimcon_qs(H_drift_fn, H_controls, noise_sources,
                       target, ctrl;
                       config, pwr_levels, weights) -> OptimizationResult

Design a pulse robust to quasi-static noise by GRAPE over an ensemble of
drift Hamiltonians sampled from the noise distribution.

# Arguments
- `H_drift_fn`    — `params -> Matrix{ComplexF64}` drift constructor
- `H_controls`    — `Vector{Matrix{ComplexF64}}` control Hamiltonians
- `noise_sources` — `Vector{QuasiStaticNoise}` noise specifications
- `target`        — `QuantumTarget` (state or unitary)
- `ctrl`          — initial `ControlSequence`
- `config`        — `GRAPEConfig`
- `pwr_levels`    — `Vector{Float64}` drive power levels (rad/s per unit amplitude);
  multiplied into H_controls (default: ones = no scaling)
- `weights`       — `Vector{Float64}` weights for each ensemble member (uniform
  by default)
- `rng`           — AbstractRNG for noise sampling (default: Random.GLOBAL_RNG)

# Returns
`OptimizationResult` from ensemble GRAPE.

# Example
```julia
σz = ComplexF64[1 0; 0 -1]
H_fn(p) = 2π * p[1] / 2 * σz
noise = [QuasiStaticNoise(:freq, 100e3; n_samples=7)]
result = robust_optimcon_qs(H_fn, [σz ./ 2], noise,
                             unitary_target(X_gate()), ctrl)
```
"""
function robust_optimcon_qs(H_drift_fn    :: Function,
                              H_controls    :: Vector{Matrix{ComplexF64}},
                              noise_sources :: Vector{QuasiStaticNoise},
                              target        :: QuantumTarget,
                              ctrl          :: ControlSequence;
                              config        :: GRAPEConfig     = GRAPEConfig(),
                              pwr_levels    :: Vector{Float64}  = ones(length(H_controls)),
                              weights       :: Vector{Float64}   = Float64[],
                              rng           :: AbstractRNG       = Random.GLOBAL_RNG
                              )::OptimizationResult
    drifts = quasi_static_ensemble(H_drift_fn, noise_sources; rng=rng)
    n_ens  = length(drifts)

    H_ctrl_scaled = _scale_controls(H_controls, pwr_levels)

    # Build QuantumSystem ensemble
    dim = size(drifts[1], 1)
    systems_ens = [QuantumSystem(drifts[i], H_ctrl_scaled, dim,
                                  length(H_controls), Dict{String,Any}())
                   for i in 1:n_ens]

    return grape_optimize_ensemble(systems_ens, target, ctrl; config=config)
end

# ============================================================================
# Pulse robustness evaluation
# ============================================================================

"""
    evaluate_qs_robustness(H_drift_fn, H_controls, noise_sources,
                           target, ctrl; pwr_levels, rng) -> (mean_F, std_F, min_F)

Evaluate the robustness of a fixed pulse `ctrl` by computing the fidelity at
each quasi-static noise sample point.

# Returns
Tuple `(mean_F, std_F, min_F)`.
"""
function evaluate_qs_robustness(H_drift_fn    :: Function,
                                  H_controls    :: Vector{Matrix{ComplexF64}},
                                  noise_sources :: Vector{QuasiStaticNoise},
                                  target        :: QuantumTarget,
                                  ctrl          :: ControlSequence;
                                  pwr_levels    :: Vector{Float64} = ones(length(H_controls)),
                                  rng           :: AbstractRNG     = Random.GLOBAL_RNG
                                  )
    drifts = quasi_static_ensemble(H_drift_fn, noise_sources; rng=rng)
    H_ctrl_scaled = _scale_controls(H_controls, pwr_levels)
    dim = size(drifts[1], 1)

    Fs = Float64[]
    for H0 in drifts
        qs = QuantumSystem(H0, H_ctrl_scaled, dim, length(H_controls), Dict{String,Any}())
        push!(Fs, compute_fidelity(qs, ctrl, target))
    end
    return (mean(Fs), std(Fs), minimum(Fs))
end
