# ============================================================
# Pulsar.jl — Unified noise / ensemble abstraction (Theme 5)
# ============================================================
#
# Collapses the three side-by-side noise architectures
#
#   Application/QuantumComputing/NoiseModels/QuasiStatic.jl
#   Application/QuantumComputing/NoiseModels/Markovian.jl
#   Application/QuantumComputing/NoiseModels/NonMarkovian.jl
#
# and the bespoke MR-side ensemble structures (`pwr_levels` outer products,
# `drifts` ensembles, MAS powder grids, DNP-specific paths) into a single
# `AbstractNoiseModel` hierarchy that any optimiser can consume.
#
# The abstraction lives in the Physics layer so both MR and QC inherit it
# automatically — no upward layer dep.
#
# Phase-2 status
# ──────────────
# This file lands the **type hierarchy** + **sample_ensemble interface**.
# Adapters from the existing `QuasiStaticNoise` / `MarkovianNoise` /
# `NoiseSpectrum` structs to the new types are provided so callers can
# migrate one site at a time.  The full rewrite of MR's EnsembleBuilder
# and the Robust optimiser onto `AbstractNoiseModel` is left for Phase 2b.
# ============================================================

"""
    AbstractNoiseModel

Supertype for noise / ensemble specifications consumed by both the MR and
QC application layers.  Concrete subtypes describe one *physical* noise
mechanism each:

  • [`ParametricDrift`](@ref)       — static drift-Hamiltonian perturbation
  • [`PowderOrientation`](@ref)     — SO(3) crystallite average (MAS / EPR)
  • [`DriveCalibration`](@ref)      — per-channel amplitude rescaling (B₁⁺)
  • [`MarkovianDissipation`](@ref)  — Lindblad jump operators
  • [`ColoredNoiseSpectrum`](@ref)  — non-Markovian filter-function model
  • [`CompositeNoise`](@ref)        — Cartesian product of components

Optimisers query the model via [`sample_ensemble`](@ref) (returns a vector
of perturbed system specifications) and aggregate per-sample fidelities or
gradients with one of the standard reductions (`:mean`, `:worst_case`,
`:cvar`).
"""
abstract type AbstractNoiseModel end

# ─────────────────────────────────────────────────────────────────────────────
# Concrete noise types
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParametricDrift(delta_fn, distribution, sigma; n_samples = 9)

Static perturbation of the drift Hamiltonian:

    H_drift_k = H_drift + delta_fn(sample_k)

Subsumes MR's `drifts` ensembles and QC's `QuasiStaticNoise(:freq, σ)`,
`QuasiStaticNoise(:coupling, σ)`, etc.

# Fields
- `delta_fn`     :: Function — `(sample::Float64) -> Matrix{ComplexF64}`,
  the perturbation operator at a given sample value
- `distribution` :: Symbol   — `:gaussian`, `:uniform`, or `:sobol`
- `sigma`        :: Union{Float64,Vector{Float64}} — std dev (Gaussian)
  or half-width (Uniform); a vector indicates a multivariate sweep
- `n_samples`    :: Int      — ensemble size
- `samples`      :: Vector{Float64} — optional explicit grid (overrides
  `distribution`/`sigma` when non-empty)
"""
struct ParametricDrift <: AbstractNoiseModel
    delta_fn     :: Function
    distribution :: Symbol
    sigma        :: Union{Float64,Vector{Float64}}
    n_samples    :: Int
    samples      :: Vector{Float64}
    function ParametricDrift(delta_fn::Function,
                              distribution::Symbol,
                              sigma::Union{Real,AbstractVector{<:Real}};
                              n_samples::Integer = 9,
                              samples::AbstractVector{<:Real} = Float64[])
        distribution in (:gaussian, :uniform, :sobol, :custom) ||
            throw(ArgumentError(
                "ParametricDrift distribution must be :gaussian, :uniform, " *
                ":sobol, or :custom (got $distribution)"))
        n_samples > 0 ||
            throw(ArgumentError("ParametricDrift n_samples must be > 0"))
        σ = sigma isa AbstractVector ? Vector{Float64}(sigma) : Float64(sigma)
        return new(delta_fn, distribution, σ, Int(n_samples),
                   Vector{Float64}(samples))
    end
end

"""
    PowderOrientation(euler_grid, weights)

Crystallite-orientation ensemble for solid-state NMR / EPR powder samples.
Equivalent to Spinach's `ensemble.m` powder grids and to Pulsar's MAS
`orientations` field.

# Fields
- `euler_grid` :: Vector{NTuple{3,Float64}} — Zaremba/Repulsion (α,β,γ) angles
- `weights`    :: Vector{Float64}            — quadrature weights (Σ = 1)
"""
struct PowderOrientation <: AbstractNoiseModel
    euler_grid :: Vector{NTuple{3,Float64}}
    weights    :: Vector{Float64}
    function PowderOrientation(euler_grid::AbstractVector,
                                weights::AbstractVector{<:Real})
        n = length(euler_grid)
        n == length(weights) ||
            throw(ArgumentError(
                "PowderOrientation: euler_grid and weights must agree in length"))
        n > 0 ||
            throw(ArgumentError("PowderOrientation must have ≥ 1 orientation"))
        sum_w = sum(weights)
        sum_w > 0 ||
            throw(ArgumentError("PowderOrientation weights must sum to > 0"))
        # normalise weights so consumers can ignore the absolute scale
        return new([NTuple{3,Float64}(eg) for eg in euler_grid],
                   Vector{Float64}(weights) ./ sum_w)
    end
end

"""
    DriveCalibration(factors, distribution = :gaussian)

Per-channel amplitude calibration uncertainty: each ensemble member sees
control waveforms `w[j, k]` rescaled by a sample of `factors[j]`.
Subsumes MR's `pwr_levels` outer-product ensembles and the QC-platform-
specific drive-strength robustness paths.

# Fields
- `factors`      :: Vector{Float64} — per-channel scale factors (one entry
  per control)
- `distribution` :: Symbol           — `:gaussian`, `:uniform`, or `:exact`
  (treat `factors` as the literal sample set)
"""
struct DriveCalibration <: AbstractNoiseModel
    factors      :: Vector{Float64}
    distribution :: Symbol
    function DriveCalibration(factors::AbstractVector{<:Real};
                               distribution::Symbol = :exact)
        length(factors) > 0 ||
            throw(ArgumentError("DriveCalibration factors must be non-empty"))
        distribution in (:gaussian, :uniform, :exact) ||
            throw(ArgumentError(
                "DriveCalibration distribution must be :gaussian, :uniform, " *
                "or :exact (got $distribution)"))
        return new(Vector{Float64}(factors), distribution)
    end
end

"""
    MarkovianDissipation(jump_ops, decay_rates)

Lindblad-form open-system noise.  Mirrors `MarkovianNoise` but lives in the
Physics layer so MR / QC / atomic platforms can all consume it.

# Fields
- `jump_ops`    :: Vector{Matrix{ComplexF64}} — collapse operators `L_k`
- `decay_rates` :: Vector{Float64}             — non-negative rates `γ_k`
"""
struct MarkovianDissipation <: AbstractNoiseModel
    jump_ops    :: Vector{Matrix{ComplexF64}}
    decay_rates :: Vector{Float64}
    function MarkovianDissipation(jump_ops::AbstractVector,
                                   decay_rates::AbstractVector{<:Real})
        n = length(jump_ops)
        n == length(decay_rates) ||
            throw(ArgumentError(
                "MarkovianDissipation: jump_ops and decay_rates must agree in length"))
        n > 0 ||
            throw(ArgumentError("MarkovianDissipation must have ≥ 1 jump op"))
        all(>=(0.0), decay_rates) ||
            throw(ArgumentError("MarkovianDissipation decay_rates must be non-negative"))
        return new([Matrix{ComplexF64}(L) for L in jump_ops],
                   Vector{Float64}(decay_rates))
    end
end

"""
    ColoredNoiseSpectrum(psd_fn, coupling_op, omega_grid)

Non-Markovian (filter-function) noise model.  The qopt-style filter-
function infidelity `Σ_k S(ω_k) F(ω_k)` is computed by the consumer; this
struct just packages the inputs.

# Fields
- `psd_fn`      :: Function                — `ω -> S(ω)` (rad/s ↦ Hz/Hz)
- `coupling_op` :: Matrix{ComplexF64}      — operator the noise couples to
- `omega_grid`  :: Vector{Float64}         — grid of angular frequencies (rad/s)
"""
struct ColoredNoiseSpectrum <: AbstractNoiseModel
    psd_fn      :: Function
    coupling_op :: Matrix{ComplexF64}
    omega_grid  :: Vector{Float64}
    function ColoredNoiseSpectrum(psd_fn::Function,
                                   coupling_op::AbstractMatrix,
                                   omega_grid::AbstractVector{<:Real})
        size(coupling_op, 1) == size(coupling_op, 2) ||
            throw(ArgumentError("ColoredNoiseSpectrum coupling_op must be square"))
        length(omega_grid) > 0 ||
            throw(ArgumentError("ColoredNoiseSpectrum omega_grid must be non-empty"))
        return new(psd_fn, Matrix{ComplexF64}(coupling_op), Vector{Float64}(omega_grid))
    end
end

"""
    CompositeNoise(components)

Cartesian product of independent noise mechanisms.  Sample size grows
multiplicatively across components.  This is the natural rewrite target
for Spinach's `ensemble.m` (drift × calibration × powder × dissipation).
"""
struct CompositeNoise <: AbstractNoiseModel
    components :: Vector{AbstractNoiseModel}
    function CompositeNoise(components::AbstractVector{<:AbstractNoiseModel})
        length(components) > 0 ||
            throw(ArgumentError("CompositeNoise must have ≥ 1 component"))
        return new(Vector{AbstractNoiseModel}(components))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Sampling interface
# ─────────────────────────────────────────────────────────────────────────────

"""
    NoiseSample

A single realised draw from an [`AbstractNoiseModel`](@ref).  Different
fields are populated depending on which mechanism produced the sample —
consumers pattern-match on the components they care about.

# Fields
- `delta_drift`   :: Union{Nothing, Matrix{ComplexF64}} — additive drift Δ
- `euler`         :: Union{Nothing, NTuple{3,Float64}}  — orientation (α,β,γ)
- `drive_factors` :: Union{Nothing, Vector{Float64}}     — per-channel scales
- `jump_ops`      :: Union{Nothing, Vector{Matrix{ComplexF64}}} — Lindblad Lₖ
- `decay_rates`   :: Union{Nothing, Vector{Float64}}     — Lindblad γₖ
- `weight`        :: Float64 — quadrature weight (Σ_samples weight = 1)
"""
struct NoiseSample
    delta_drift   :: Union{Nothing, Matrix{ComplexF64}}
    euler         :: Union{Nothing, NTuple{3,Float64}}
    drive_factors :: Union{Nothing, Vector{Float64}}
    jump_ops      :: Union{Nothing, Vector{Matrix{ComplexF64}}}
    decay_rates   :: Union{Nothing, Vector{Float64}}
    weight        :: Float64
end

"""
    NoiseSample(; delta_drift = nothing, euler = nothing,
                  drive_factors = nothing, jump_ops = nothing,
                  decay_rates = nothing, weight = 1.0)

Keyword constructor — populate only the fields produced by the noise
mechanism in question.
"""
NoiseSample(; delta_drift = nothing,
              euler = nothing,
              drive_factors = nothing,
              jump_ops = nothing,
              decay_rates = nothing,
              weight = 1.0)::NoiseSample =
    NoiseSample(delta_drift, euler, drive_factors,
                jump_ops, decay_rates, Float64(weight))

"""
    sample_ensemble(noise::AbstractNoiseModel; rng = Random.default_rng())
        -> Vector{NoiseSample}

Draw the ensemble of [`NoiseSample`](@ref)s described by `noise`.

For deterministic noise types (`PowderOrientation`, `MarkovianDissipation`,
`DriveCalibration{:exact}`, `ColoredNoiseSpectrum`) the returned vector is
the canonical fixed grid.  For stochastic types (`ParametricDrift`,
`DriveCalibration{:gaussian|:uniform}`) `rng` controls the draw.

`CompositeNoise` returns the Cartesian product of its components' samples
with multiplicative weights.
"""
function sample_ensemble end

# ── ParametricDrift ────────────────────────────────────────────────────────
function sample_ensemble(p::ParametricDrift;
                          rng::AbstractRNG = Random.default_rng())::Vector{NoiseSample}
    if !isempty(p.samples)
        vals = p.samples
    elseif p.distribution === :gaussian
        σ = p.sigma isa AbstractVector ? p.sigma[1] : p.sigma
        vals = σ .* randn(rng, p.n_samples)
    elseif p.distribution === :uniform
        σ = p.sigma isa AbstractVector ? p.sigma[1] : p.sigma
        vals = σ .* (2 .* rand(rng, p.n_samples) .- 1)
    else  # :sobol or :custom without explicit samples
        error("ParametricDrift: distribution=$(p.distribution) requires " *
              "explicit `samples`.")
    end
    w = 1 / length(vals)
    return [NoiseSample(; delta_drift = Matrix{ComplexF64}(p.delta_fn(v)),
                          weight = w) for v in vals]
end

# ── PowderOrientation ──────────────────────────────────────────────────────
function sample_ensemble(p::PowderOrientation;
                          rng::AbstractRNG = Random.default_rng())::Vector{NoiseSample}
    return [NoiseSample(; euler = p.euler_grid[i], weight = p.weights[i])
            for i in eachindex(p.euler_grid)]
end

# ── DriveCalibration ───────────────────────────────────────────────────────
function sample_ensemble(p::DriveCalibration;
                          rng::AbstractRNG = Random.default_rng())::Vector{NoiseSample}
    if p.distribution === :exact
        # `factors` is the literal sample set — one element per ensemble member,
        # uniformly broadcast across all channels.
        w = 1 / length(p.factors)
        return [NoiseSample(; drive_factors = fill(f, 1), weight = w)
                for f in p.factors]
    else
        error("DriveCalibration: distribution=$(p.distribution) requires " *
              "explicit factor draws.  Pass `distribution=:exact` with a " *
              "pre-computed sample set, or use ParametricDrift for now.")
    end
end

# ── MarkovianDissipation: single deterministic sample ──────────────────────
function sample_ensemble(p::MarkovianDissipation;
                          rng::AbstractRNG = Random.default_rng())::Vector{NoiseSample}
    return [NoiseSample(; jump_ops = p.jump_ops,
                          decay_rates = p.decay_rates,
                          weight = 1.0)]
end

# ── ColoredNoiseSpectrum: opaque single-sample placeholder ─────────────────
function sample_ensemble(::ColoredNoiseSpectrum;
                          rng::AbstractRNG = Random.default_rng())::Vector{NoiseSample}
    # Filter-function noise is consumed in frequency space; the ensemble
    # interpretation is a single "sample" with weight 1, and the consumer
    # is expected to read `noise.psd_fn` / `noise.omega_grid` directly.
    return [NoiseSample(; weight = 1.0)]
end

# ── CompositeNoise: Cartesian product ──────────────────────────────────────
function sample_ensemble(p::CompositeNoise;
                          rng::AbstractRNG = Random.default_rng())::Vector{NoiseSample}
    component_samples = [sample_ensemble(c; rng = rng) for c in p.components]
    out = NoiseSample[]
    for combo in Iterators.product(component_samples...)
        push!(out, _merge_samples(combo))
    end
    return out
end

# Internal: merge a tuple of NoiseSamples drawn from independent
# components into a single NoiseSample.  Multiplicative weight; first-set
# wins for any specific field (caller's responsibility to keep components
# orthogonal in their populated fields).
function _merge_samples(samples::Tuple)::NoiseSample
    delta_drift   = nothing
    euler         = nothing
    drive_factors = nothing
    jump_ops      = nothing
    decay_rates   = nothing
    weight        = 1.0
    for s in samples
        s.delta_drift   === nothing || (delta_drift   = s.delta_drift)
        s.euler         === nothing || (euler         = s.euler)
        s.drive_factors === nothing || (drive_factors = s.drive_factors)
        s.jump_ops      === nothing || (jump_ops      = s.jump_ops)
        s.decay_rates   === nothing || (decay_rates   = s.decay_rates)
        weight *= s.weight
    end
    return NoiseSample(delta_drift, euler, drive_factors,
                       jump_ops, decay_rates, weight)
end

"""
    n_samples(noise::AbstractNoiseModel) -> Int

Total ensemble size produced by `sample_ensemble(noise)`.  Cheap to
evaluate — no actual sampling.  Useful for UI / progress bars before the
ensemble is realised.
"""
n_samples(p::ParametricDrift)::Int      = isempty(p.samples) ? p.n_samples :
                                                                length(p.samples)
n_samples(p::PowderOrientation)::Int    = length(p.euler_grid)
n_samples(p::DriveCalibration)::Int     = length(p.factors)
n_samples(::MarkovianDissipation)::Int  = 1
n_samples(::ColoredNoiseSpectrum)::Int  = 1
n_samples(p::CompositeNoise)::Int       = prod(n_samples(c) for c in p.components)
