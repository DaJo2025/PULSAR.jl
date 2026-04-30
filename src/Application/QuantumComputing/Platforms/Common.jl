# ============================================================================
# Application/QuantumComputing/Platforms/Common.jl
# Shared helpers for the QC platform optimcon dispatches.
# ============================================================================

"""
    _build_platform_qs(sys; use_lindblad=false) -> QuantumSystem

Build a `QuantumSystem` from any platform system (`TransmonSystem`,
`TrappedIonSystem`, `NeutralAtomSystem`, `SpinQubitSystem`, `NVCenterSystem`).

If `use_lindblad=true` and `sys.collapse_ops` is non-empty, the returned system
is a Liouville-space `QuantumSystem` via `lindblad_system_from_jump_ops` with
unit rates (the collapse operators are assumed to be pre-scaled by √γ).
Otherwise a Hilbert-space `QuantumSystem` is returned.
"""
function _build_platform_qs(sys; use_lindblad::Bool = false)::QuantumSystem
    if use_lindblad && !isempty(sys.collapse_ops)
        rates = ones(length(sys.collapse_ops))
        return lindblad_system_from_jump_ops(
            sys.H_drift, sys.collapse_ops, rates, sys.H_controls)
    else
        return QuantumSystem(sys.H_drift, sys.H_controls,
                             sys.dim, sys.n_controls, sys.metadata)
    end
end

"""
    _platform_grape(sys, target, ctrl; config, use_lindblad=false,
                    penalty_fns=Function[], penalty_grad_fns=Function[])
    -> OptimizationResult

Standard platform optimcon skeleton: build the `QuantumSystem` (optionally
in Liouville space) and hand off to `grape_optimize`.  Platform-specific
preprocessing (e.g. DRAG pre-conditioning, B₁ ensembles) is performed by
the caller *before* invoking this helper.
"""
function _platform_grape(sys, target, ctrl;
                         config         :: GRAPEConfig,
                         use_lindblad   :: Bool = false,
                         penalty_fns         = nothing,
                         penalty_grad_fns    = nothing)::OptimizationResult
    qs = _build_platform_qs(sys; use_lindblad = use_lindblad)
    if penalty_fns === nothing
        return grape_optimize(qs, target, ctrl; config = config)
    else
        return grape_optimize(qs, target, ctrl;
                              config           = config,
                              penalty_fns      = penalty_fns,
                              penalty_grad_fns = penalty_grad_fns)
    end
end

# ============================================================================
# Small shared helpers used across QC platforms + noise models
# ============================================================================

"""
    _scale_controls(H_controls, scale) -> Vector{Matrix{ComplexF64}}

Return a fresh list where entry `k` is `scale[k] .* H_controls[k]`. Used when a
noise/robustness path perturbs per-channel drive strength.
"""
function _scale_controls(H_controls :: Vector{Matrix{ComplexF64}},
                          scale      :: AbstractVector{<:Real})::Vector{Matrix{ComplexF64}}
    length(scale) == length(H_controls) ||
        throw(ArgumentError("scale must have the same length as H_controls"))
    return [scale[k] .* H_controls[k] for k in eachindex(H_controls)]
end

"""
    _diagonal_projector(dim, indices) -> Matrix{ComplexF64}

Hermitian projector Π with `Π[i,i] = 1` for `i ∈ indices`, zero elsewhere.
"""
function _diagonal_projector(dim     :: Int,
                              indices :: AbstractVector{<:Integer})::Matrix{ComplexF64}
    Π = zeros(ComplexF64, dim, dim)
    @inbounds for i in indices
        Π[i, i] = 1.0 + 0im
    end
    return Π
end

"""
    _numeric_derivative(x, dt) -> Vector{eltype(x)}

Centered finite-difference first derivative with one-sided endpoints. Used for
DRAG-style quadrature corrections on piecewise-constant waveforms.
"""
function _numeric_derivative(x :: AbstractVector, dt :: Real)
    N  = length(x)
    dx = similar(x)
    N  < 2 && return (fill!(dx, zero(eltype(x))); dx)
    dx[1] = (x[2] - x[1]) / dt
    dx[N] = (x[N] - x[N - 1]) / dt
    @inbounds for k in 2:(N - 1)
        dx[k] = (x[k + 1] - x[k - 1]) / (2 * dt)
    end
    return dx
end
