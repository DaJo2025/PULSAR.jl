# ============================================================================
# Optimization/Ensemble/EnsembleObjective.jl
# ============================================================================
# Generic ensemble-objective wrapper.
#
# Unifies PULSAR's four pre-existing ensemble mechanisms (`grape_optimize_ensemble`,
# `robust_optimize`, MR kernels, QC quasi-static noise) behind a single type that
# produces closures compatible with every generic optimizer — gradient or
# derivative-free — in the library.
#
# Two evaluation modes:
#   (A) per-sample — one closure per ensemble member; aggregator applied in
#       Julia. Works with any aggregator (`:mean`, `:worst_case`, `:cvar`).
#   (B) batched_eval — a single user-supplied (F, ∇F) function that internally
#       handles the full ensemble. Lets MR layer keep its GPU-batched kernel as
#       a `:mean` fast path without a per-sample rewrite.
#
# Public API
#   EnsembleObjective                               (struct)
#   ensemble_value(obj, θ)            -> Float64
#   ensemble_value_and_grad(obj, θ)   -> (F, ∇F)
#   ensemble_wrap(obj)                -> (f, grad!)       # (f, grad!, θ₀) optimizers
#   ensemble_wrap_fonly(obj)          -> f                # (f, θ₀) optimizers
#
# Aggregator surgery matches `RobustOpt.jl` lines 277-298 exactly (extracted once).
# ============================================================================

using LinearAlgebra
using Statistics
using Random

# ---------------------------------------------------------------------------
# Core type
# ---------------------------------------------------------------------------

"""
    EnsembleObjective{F, G, R, B}

Generic container for an ensemble of per-sample objectives + an aggregator.

Parametrised over closure types:
- `F` — element type of `f_samples` (concrete closure type avoids
  `jl_apply_generic` dispatch inside the per-sample loop).
- `G` — element type of `grad_samples`.
- `R` — type of `resample!` (`Nothing` or concrete function type).
- `B` — type of `batched_eval` (`Nothing` or concrete function type).

Builders in `SystemBuilder.jl`, `PerturbationBuilder.jl`, and
`EnsembleBuilder.jl` produce concrete-element vectors via array comprehensions,
so the inner `Threads.@threads` per-sample dispatch is fully type-stable.

# Fields
- `f_samples::Vector{F}` — each `f_i(θ) -> Float64` returns a per-sample
  fidelity. Length must equal `n_samples`.
- `grad_samples::Union{Nothing,Vector{G}}` — each `g_i(gv, θ)` fills
  `gv::Vector{Float64}` with the per-sample gradient; pass `nothing` if only
  derivative-free optimizers will be used.
- `aggregator::Symbol` — `:mean`, `:worst_case`, or `:cvar`.
- `cvar_alpha::Float64` — tail fraction for `:cvar` (worst `alpha·N` samples).
- `resample!::R` — optional hook `resample!(obj) -> nothing` invoked at the
  start of each `ensemble_value_and_grad` call. When non-nothing the builder
  must close over mutable sample storage and refresh `f_samples` /
  `grad_samples` in place.
- `batched_eval::B` — optional fast-path `batched_eval(θ) -> (F, ∇F)`. When
  non-nothing the per-sample path is skipped.
- `n_samples::Int` — matches `length(f_samples)` in per-sample mode or the
  caller-supplied logical count in batched mode.
- `_fv::Vector{Float64}` — persistent scratch for per-sample fidelities.
- `_gv::Vector{Vector{Float64}}` — persistent scratch for per-sample gradients
  (one pre-allocated column per sample; resized on first call).

# Notes
- Per-sample evaluation is parallelised with `Threads.@threads`.
- Invariants: `0 < cvar_alpha ≤ 1`; `aggregator ∈ (:mean, :worst_case, :cvar)`;
  either `batched_eval === nothing` or `aggregator === :mean`.
"""
struct EnsembleObjective{F, G, R, B}
    f_samples    :: Vector{F}
    grad_samples :: Union{Nothing, Vector{G}}
    aggregator   :: Symbol
    cvar_alpha   :: Float64
    resample!    :: R
    batched_eval :: B
    n_samples    :: Int
    _fv          :: Vector{Float64}
    _gv          :: Vector{Vector{Float64}}
end

@inline function _validate_ensemble_args(n_f::Int, grad_samples, aggregator::Symbol,
                                         cvar_alpha::Float64, batched_eval, n_samples::Int)
    aggregator in (:mean, :worst_case, :cvar) ||
        throw(ArgumentError("aggregator must be :mean, :worst_case, or :cvar; got $(aggregator)"))
    (0 < cvar_alpha ≤ 1) ||
        throw(ArgumentError("cvar_alpha must lie in (0, 1]; got $(cvar_alpha)"))
    if batched_eval !== nothing
        aggregator === :mean ||
            throw(ArgumentError("batched_eval fast path requires aggregator=:mean; got $(aggregator)"))
        n_samples ≥ 1 ||
            throw(ArgumentError("n_samples must be positive in batched mode"))
    else
        n_samples == n_f ||
            throw(ArgumentError(
                "n_samples ($(n_samples)) does not match length(f_samples) ($(n_f))"))
        if grad_samples !== nothing
            length(grad_samples) == n_samples ||
                throw(ArgumentError(
                    "length(grad_samples) must equal length(f_samples)"))
        end
    end
    return nothing
end

# Convenience keyword-style constructor
"""
    EnsembleObjective(f_samples;
                       grad_samples = nothing,
                       aggregator   = :mean,
                       cvar_alpha   = 0.2,
                       resample!    = nothing,
                       batched_eval = nothing,
                       n_samples    = length(f_samples))

Keyword form; see field docstring on the struct for semantics.
"""
function EnsembleObjective(f_samples::Vector;
                            grad_samples = nothing,
                            aggregator::Symbol = :mean,
                            cvar_alpha::Real   = 0.2,
                            resample! = nothing,
                            batched_eval = nothing,
                            n_samples::Int = length(f_samples))
    α = Float64(cvar_alpha)
    _validate_ensemble_args(length(f_samples), grad_samples, aggregator, α,
                            batched_eval, n_samples)
    F = eltype(f_samples)
    G = grad_samples === nothing ? Function : eltype(grad_samples)
    R = typeof(resample!)
    B = typeof(batched_eval)
    return EnsembleObjective{F, G, R, B}(
        f_samples, grad_samples, aggregator, α,
        resample!, batched_eval, n_samples,
        Float64[], Vector{Float64}[])
end

# Batched-only constructor — no per-sample closures required.
"""
    EnsembleObjective(; batched_eval, n_samples, aggregator=:mean)

Construct a batched-mode `EnsembleObjective` with no per-sample closures.
`batched_eval(θ) -> (F, ∇F)` must aggregate internally.
"""
function EnsembleObjective(; batched_eval::Function,
                            n_samples::Int,
                            aggregator::Symbol = :mean,
                            cvar_alpha::Real   = 0.2)
    α = Float64(cvar_alpha)
    _validate_ensemble_args(0, nothing, aggregator, α, batched_eval, n_samples)
    B = typeof(batched_eval)
    return EnsembleObjective{Function, Function, Nothing, B}(
        Function[], nothing, aggregator, α,
        nothing, batched_eval, n_samples,
        Float64[], Vector{Float64}[])
end

# ---------------------------------------------------------------------------
# Aggregator surgery — extracted once from RobustOpt.jl lines 277-298
# ---------------------------------------------------------------------------

@inline function _agg_value(fv::Vector{Float64}, aggregator::Symbol, alpha::Float64)
    aggregator === :mean       && return mean(fv)
    aggregator === :worst_case && return minimum(fv)
    aggregator === :cvar       && return _cvar_mean(fv, alpha)
    throw(ArgumentError("Unknown aggregator $(aggregator)"))
end

# NOTE — do not call RobustOpt's `_cvar` directly: the new ensemble module is
# included *before* RobustOpt.jl in the load order, so that symbol is not yet
# defined.  Keep a local tail-mean helper here; RobustOpt can delegate later.
@inline function _cvar_mean(fv::Vector{Float64}, alpha::Float64)
    n_tail   = max(1, round(Int, alpha * length(fv)))
    sorted_f = sort(fv)                   # ascending: worst first
    return mean(@view sorted_f[1:n_tail])
end

# Aggregate a Vector of per-sample gradients (each is a Vector{Float64}).
# Reuses `fv` to pick worst/tail samples under `:worst_case` / `:cvar`.
function _agg_grad!(out::Vector{Float64},
                     fv::Vector{Float64},
                     gv::Vector{Vector{Float64}},
                     aggregator::Symbol,
                     alpha::Float64)
    n = length(gv)
    @assert length(fv) == n
    @assert length(out) == length(gv[1])

    if aggregator === :mean
        fill!(out, 0.0)
        @inbounds for gi in gv
            out .+= gi
        end
        out ./= n
    elseif aggregator === :worst_case
        w = argmin(fv)
        copyto!(out, gv[w])
    elseif aggregator === :cvar
        n_tail = max(1, round(Int, alpha * n))
        order  = sortperm(fv)              # ascending
        fill!(out, 0.0)
        @inbounds for idx in 1:n_tail
            out .+= gv[order[idx]]
        end
        out ./= n_tail
    else
        throw(ArgumentError("Unknown aggregator $(aggregator)"))
    end
    return out
end

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

"""
    ensemble_value(obj, θ) -> Float64

Evaluate the aggregated ensemble fidelity. Batched-mode shortcuts through
`obj.batched_eval` (discards its gradient). Per-sample mode runs
`Threads.@threads` across `obj.f_samples`.
"""
function ensemble_value(obj::EnsembleObjective{F, G, R, B},
                         θ::AbstractVector{<:Real})::Float64 where {F, G, R, B}
    obj.resample! === nothing || obj.resample!(obj)

    if obj.batched_eval !== nothing
        Fval, _ = obj.batched_eval(θ)
        return Float64(Fval)
    end

    f_samples = obj.f_samples
    n  = length(f_samples)
    fv = obj._fv
    resize!(fv, n)
    Threads.@threads for i in 1:n
        @inbounds fv[i] = Float64(f_samples[i](θ))
    end
    return _agg_value(fv, obj.aggregator, obj.cvar_alpha)
end

# Resize the persistent per-sample gradient buffers to (n × d); reuse existing
# inner vectors whenever possible so steady-state calls are allocation-free.
@inline function _prepare_gv!(gv::Vector{Vector{Float64}}, n::Int, d::Int)
    if length(gv) != n
        old = length(gv)
        resize!(gv, n)
        @inbounds for i in (old + 1):n
            gv[i] = Vector{Float64}(undef, d)
        end
    end
    @inbounds for i in 1:n
        gi = gv[i]
        length(gi) == d || resize!(gi, d)
    end
    return gv
end

"""
    ensemble_value_and_grad(obj, θ) -> (F, ∇F)

Evaluate aggregated fidelity and its gradient. Requires `obj.grad_samples`
unless `obj.batched_eval` is supplied.
"""
function ensemble_value_and_grad(obj::EnsembleObjective{F, G, R, B},
                                  θ::AbstractVector{<:Real}) where {F, G, R, B}
    obj.resample! === nothing || obj.resample!(obj)

    if obj.batched_eval !== nothing
        Fval, ∇F = obj.batched_eval(θ)
        return Float64(Fval), Vector{Float64}(vec(∇F))
    end

    gs_raw = obj.grad_samples
    if gs_raw === nothing
        throw(ArgumentError("ensemble_value_and_grad called without grad_samples; " *
                            "use ensemble_value for derivative-free optimizers."))
    end
    # Force the compiler to specialize the per-sample dispatch on the concrete
    # element type G rather than the Union{Nothing, Vector{G}} field type.
    grad_samples = gs_raw::Vector{G}

    f_samples = obj.f_samples
    n  = length(f_samples)
    d  = length(θ)
    fv = obj._fv
    resize!(fv, n)
    gv = _prepare_gv!(obj._gv, n, d)

    Threads.@threads for i in 1:n
        @inbounds begin
            fv[i] = Float64(f_samples[i](θ))
            grad_samples[i](gv[i], θ)
        end
    end
    Fval = _agg_value(fv, obj.aggregator, obj.cvar_alpha)
    ∇F   = Vector{Float64}(undef, d)
    _agg_grad!(∇F, fv, gv, obj.aggregator, obj.cvar_alpha)
    return Fval, ∇F
end

# ---------------------------------------------------------------------------
# Optimizer-facing closures
# ---------------------------------------------------------------------------

"""
    ensemble_wrap(obj) -> (f, grad!)

Return a `(f, grad!)` pair suitable for PULSAR's `(f, grad!, θ₀)` optimizers
(L-BFGS, BFGS, CG, Adam, Newton, trust-region, L-BFGS-B, etc.). Every call to
`grad!(gv, θ)` also fills `gv`, so one pass computes F and ∇F together.

Convention (PULSAR-wide): these optimizers *minimize* their objective. The
wrappers therefore return `-F` and `-∇F` so callers can feed the result into a
minimizer to maximize fidelity — mirroring the historical inversion used by
`grape_optimize` and `robust_optimize`.
"""
function ensemble_wrap(obj::EnsembleObjective)
    # Cache last (θ, F, ∇F) to avoid double work when optimizers call f and grad!
    # on the same θ in sequence. `last_θ` is a persistent buffer resized in-place.
    last_θ     = Vector{Float64}()
    last_valid = Ref(false)
    last_F     = Ref(0.0)
    last_G     = Ref(Vector{Float64}())

    function _eval(θ::AbstractVector{<:Real})
        if last_valid[] && length(last_θ) == length(θ) && last_θ == θ
            return last_F[], last_G[]
        end
        F, ∇F = ensemble_value_and_grad(obj, θ)
        resize!(last_θ, length(θ))
        copyto!(last_θ, θ)
        last_valid[] = true
        last_F[] = F
        last_G[] = ∇F
        return F, ∇F
    end

    f(θ) = begin
        F, _ = _eval(θ)
        return -F       # minimizer → negate
    end

    grad!(gv, θ) = begin
        F, ∇F = _eval(θ)
        @. gv = -∇F     # minimizer → negate gradient
        return gv
    end

    return f, grad!
end

"""
    ensemble_wrap_fonly(obj) -> f

Return a scalar `f(θ) -> -F` for derivative-free optimizers (`cmaes_optimize`,
`nelder_mead_optimize`, `pso_optimize`, etc.). Sign is negated for the same
reason as [`ensemble_wrap`](@ref).
"""
function ensemble_wrap_fonly(obj::EnsembleObjective)
    f(θ) = -ensemble_value(obj, θ)
    return f
end

# ---------------------------------------------------------------------------
# Ascent variants (some callers prefer maximization)
# ---------------------------------------------------------------------------

"""
    ensemble_wrap_ascent(obj) -> (f, grad!)

Same as [`ensemble_wrap`](@ref) but returns `+F` and `+∇F` (maximization
convention). Useful when plugging into a custom ascent loop or into the Krotov
dispatch which handles sign internally.
"""
function ensemble_wrap_ascent(obj::EnsembleObjective)
    last_θ     = Vector{Float64}()
    last_valid = Ref(false)
    last_F     = Ref(0.0)
    last_G     = Ref(Vector{Float64}())

    function _eval(θ::AbstractVector{<:Real})
        if last_valid[] && length(last_θ) == length(θ) && last_θ == θ
            return last_F[], last_G[]
        end
        F, ∇F = ensemble_value_and_grad(obj, θ)
        resize!(last_θ, length(θ))
        copyto!(last_θ, θ)
        last_valid[] = true
        last_F[] = F
        last_G[] = ∇F
        return F, ∇F
    end

    f(θ) = (_eval(θ)[1])
    grad!(gv, θ) = (@. gv = _eval(θ)[2]; gv)
    return f, grad!
end
