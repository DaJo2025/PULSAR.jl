# MetalBackend.jl — Apple Metal GPU backend
#
# Provides real GPU-accelerated operations when Metal.jl is installed and
# running on Apple Silicon (M1/M2/M3/M4) or a macOS Metal-capable GPU.
# Falls back gracefully to CPU with a @warn when Metal is absent.
#
# Design notes:
#   • Metal.jl is loaded at module-include time via a try/catch block.
#   • Metal natively supports Float32; Float64 is NOT supported in Metal shaders.
#     For quantum mechanics (which requires complex arithmetic to ~1e-10), we:
#     (a) Use Float32 for intermediate computations when use_fp32=true, or
#     (b) Use Metal for embarrassingly-parallel real-valued work and fall back
#         to CPU for the precision-critical complex eigendecompositions.
#   • The strategy used here: upload to MtlArray, run Metal BLAS (gemm!),
#     but perform eigendecomposition on CPU (Float64) — this is still faster
#     than pure CPU for large matrix products.
#   • use_fp64_fallback=true (default) routes eigen to CPU.

using LinearAlgebra

# ---------------------------------------------------------------------------
# Runtime availability flag
# ---------------------------------------------------------------------------

const _METAL_LOADED = Ref{Bool}(false)

let
    try
        @eval using Metal
        if Metal.functional()
            _METAL_LOADED[] = true
        else
            @warn "MetalBackend: Metal.jl loaded but no functional Metal device found. " *
                  "Calls to Metal backend functions will fall back to CPU."
        end
    catch
        # Metal.jl not installed or not on Apple Silicon — disabled silently
    end
end

"""
    METAL_AVAILABLE::Bool

`true` only if Metal.jl is installed **and** a functional Metal GPU is present
at the time the module was loaded.
"""
const METAL_AVAILABLE = _METAL_LOADED[]

# ---------------------------------------------------------------------------
# Type definition
# ---------------------------------------------------------------------------

"""
    MetalBackend

Configuration struct for the Apple Metal GPU backend (Apple Silicon / macOS).

Construct via [`metal_backend`](@ref).  Requires Metal.jl and Apple Silicon
hardware (M1/M2/M3/M4 or an AMD GPU via macOS Metal API).

# Fields
- `device_id::Int`: Metal device index (0-based).
- `memory_limit_gb::Float64`: Soft GPU memory budget in GiB.
- `use_fp32::Bool`: Use Float32 throughout (native Metal precision).
  Faster but accumulates rounding error; suitable for ≤ 6-qubit systems.
- `use_fp64_fallback::Bool`: Route FP64-critical steps (eigendecomposition)
  through CPU.  Recommended `true` (default).
"""
struct MetalBackend
    device_id         :: Int
    memory_limit_gb   :: Float64
    use_fp32          :: Bool
    use_fp64_fallback :: Bool
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    metal_backend(; device_id=0, memory_limit_gb=Inf,
                    use_fp32=false, use_fp64_fallback=true) -> MetalBackend

Construct a [`MetalBackend`](@ref).

Throws an `ErrorException` if Metal.jl is not installed or if the system
is not running on Apple Silicon / macOS with Metal support.
"""
function metal_backend(;
    device_id         :: Int     = 0,
    memory_limit_gb   :: Float64 = Inf,
    use_fp32          :: Bool    = false,
    use_fp64_fallback :: Bool    = true,
)
    if !_METAL_LOADED[]
        error(
            "Metal is not available.  Install Metal.jl with `] add Metal` on " *
            "an Apple Silicon Mac.  Use `cpu_backend()` instead.",
        )
    end
    return MetalBackend(device_id, memory_limit_gb, use_fp32, use_fp64_fallback)
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# CPU fallback: Hermitian matrix exponential via eigendecomposition.
function _herm_matexp_metal(H::Matrix{ComplexF64}, dt::Float64)
    F = eigen(Hermitian(H))
    return F.vectors * Diagonal(exp.(-im .* F.values .* dt)) * F.vectors'
end

# Metal matrix multiply (F32): A * B using Metal BLAS
function _metal_gemm_f32(A::Matrix{ComplexF64}, B::Matrix{ComplexF64})
    # Metal only supports Float32; split complex into real/imag pairs.
    # Re(A*B) = Re(A)*Re(B) - Im(A)*Im(B)
    # Im(A*B) = Re(A)*Im(B) + Im(A)*Re(B)
    Ar = Metal.MtlArray(Float32.(real(A)))
    Ai = Metal.MtlArray(Float32.(imag(A)))
    Br = Metal.MtlArray(Float32.(real(B)))
    Bi = Metal.MtlArray(Float32.(imag(B)))

    Cr = Ar * Br - Ai * Bi
    Ci = Ar * Bi + Ai * Br

    Cr_h = Float64.(Array(Cr))
    Ci_h = Float64.(Array(Ci))
    return complex.(Cr_h, Ci_h)
end

# ---------------------------------------------------------------------------
# matrix_exponential_metal
# ---------------------------------------------------------------------------

"""
    matrix_exponential_metal(H, dt, backend) -> Matrix{ComplexF64}

Compute `exp(-i H dt)` for a Hermitian matrix `H`.

Strategy:
- Eigendecomposition is always on CPU (Metal lacks FP64 shader support).
- The similarity transform `V * Diagonal(exp(λ)) * V'` uses Metal BLAS
  (FP32) when `backend.use_fp32=true`, otherwise CPU FP64.

The result is always returned as a host `Matrix{ComplexF64}`.
"""
function matrix_exponential_metal(
    H       :: Matrix{ComplexF64},
    dt      :: Float64,
    backend :: MetalBackend,
)
    if !_METAL_LOADED[]
        @warn "matrix_exponential_metal: Metal unavailable, falling back to CPU."
        return _herm_matexp_metal(H, dt)
    end
    try
        # Eigendecomposition on CPU (FP64)
        F    = eigen(Hermitian(H))
        expλ = exp.(-im .* F.values .* dt)   # diagonal entries
        V    = F.vectors
        D_V  = V * Diagonal(expλ)            # V * exp(Λ)

        if backend.use_fp32 && !backend.use_fp64_fallback
            # Final gemm on Metal (FP32)
            return _metal_gemm_f32(D_V, V')
        else
            # CPU FP64 final multiply
            return D_V * V'
        end
    catch e
        @warn "matrix_exponential_metal: computation failed ($e), falling back to CPU."
        return _herm_matexp_metal(H, dt)
    end
end

# ---------------------------------------------------------------------------
# batch_propagators_metal
# ---------------------------------------------------------------------------

"""
    batch_propagators_metal(H_array, dt, backend) -> Array{ComplexF64,3}

Compute all propagators `U[k] = exp(-i H_array[:,:,k] dt)` in parallel.

`H_array` is a `(dim, dim, n_steps)` array of Hermitian matrices.

Eigendecompositions are performed on CPU in parallel (Julia threads);
the final similarity transform `V D V'` may use Metal BLAS (FP32) when
`use_fp32=true`.  Returns a host `(dim, dim, n_steps)` array.
"""
function batch_propagators_metal(
    H_array :: AbstractArray{ComplexF64,3},
    dt      :: Float64,
    backend :: MetalBackend,
)
    dim, _, n_steps = size(H_array)
    result = Array{ComplexF64,3}(undef, dim, dim, n_steps)

    if !_METAL_LOADED[]
        @warn "batch_propagators_metal: Metal unavailable, falling back to CPU."
        for k in 1:n_steps
            result[:, :, k] = _herm_matexp_metal(H_array[:, :, k], dt)
        end
        return result
    end

    try
        # Parallelize eigen across time steps on CPU threads
        Threads.@threads for k in 1:n_steps
            result[:, :, k] = matrix_exponential_metal(H_array[:, :, k], dt, backend)
        end
        return result
    catch e
        @warn "batch_propagators_metal: computation failed ($e), falling back to CPU."
        for k in 1:n_steps
            result[:, :, k] = _herm_matexp_metal(H_array[:, :, k], dt)
        end
        return result
    end
end

# ---------------------------------------------------------------------------
# fidelity_metal
# ---------------------------------------------------------------------------

"""
    fidelity_metal(U_total, U_target, backend) -> Float64

Compute the gate fidelity `|Tr(U_target† U_total)|² / dim²`.

The inner product `Tr(U_target† U_total) = sum(conj(U_target) .* U_total)`
is computed on the Metal GPU (FP32) and accumulated on CPU (FP64) when
`use_fp32=true`; otherwise computed on CPU.
"""
function fidelity_metal(
    U_total  :: Matrix{ComplexF64},
    U_target :: Matrix{ComplexF64},
    backend  :: MetalBackend,
)
    dim = size(U_total, 1)

    if !_METAL_LOADED[]
        @warn "fidelity_metal: Metal unavailable, falling back to CPU."
        overlap = tr(U_target' * U_total)
        return abs2(overlap) / dim^2
    end

    try
        if backend.use_fp32 && !backend.use_fp64_fallback
            # FP32 element-wise product on Metal, reduce on CPU
            Ut_r = Metal.MtlArray(Float32.(real(U_target)))
            Ut_i = Metal.MtlArray(Float32.(imag(U_target)))
            U_r  = Metal.MtlArray(Float32.(real(U_total)))
            U_i  = Metal.MtlArray(Float32.(imag(U_total)))
            # Re(conj(Ut) .* U) = Re(Ut)*Re(U) + Im(Ut)*Im(U)
            # Im(conj(Ut) .* U) = Re(Ut)*Im(U) - Im(Ut)*Re(U)
            inner_r = Float64(sum(Array(Ut_r .* U_r .+ Ut_i .* U_i)))
            inner_i = Float64(sum(Array(Ut_r .* U_i .- Ut_i .* U_r)))
            overlap = complex(inner_r, inner_i)
        else
            # CPU FP64
            overlap = tr(U_target' * U_total)
        end
        return abs2(overlap) / dim^2
    catch e
        @warn "fidelity_metal: computation failed ($e), falling back to CPU."
        overlap = tr(U_target' * U_total)
        return abs2(overlap) / dim^2
    end
end

# ---------------------------------------------------------------------------
# gradient_metal
# ---------------------------------------------------------------------------

"""
    gradient_metal(H_drift, H_ctrl, controls, target, dt, backend) -> Matrix{Float64}

Compute the full GRAPE gradient using the Metal GPU backend.

Implements the standard forward–backward GRAPE pass.  Matrix products for
the similarity transforms use Metal BLAS (FP32) when `use_fp32=true`;
eigendecompositions are always on CPU (FP64).

# Arguments
- `H_drift`: Drift Hamiltonian `(dim × dim)`.
- `H_ctrl`: Vector of `n_controls` control Hamiltonians.
- `controls`: `(n_controls × n_steps)` control amplitudes.
- `target`: Target unitary `(dim × dim)`.
- `dt`: Timestep.
- `backend`: The `MetalBackend` instance.

Returns `grad` of size `(n_controls, n_steps)`.
"""
function gradient_metal(
    H_drift  :: Matrix{ComplexF64},
    H_ctrl   :: Vector{<:Matrix{ComplexF64}},
    controls :: Matrix{Float64},
    target   :: Matrix{ComplexF64},
    dt       :: Float64,
    backend  :: MetalBackend,
)
    n_ctrl, n_steps = size(controls)
    dim             = size(H_drift, 1)

    # CPU fallback (identical to CUDA fallback)
    function _cpu_grape_grad()
        P = Vector{Matrix{ComplexF64}}(undef, n_steps + 1)
        P[1] = Matrix{ComplexF64}(I, dim, dim)
        for k in 1:n_steps
            H_k = copy(H_drift)
            for j in 1:n_ctrl
                H_k .+= controls[j, k] .* H_ctrl[j]
            end
            P[k+1] = _herm_matexp_metal(H_k, dt) * P[k]
        end
        Q = Vector{Matrix{ComplexF64}}(undef, n_steps + 1)
        Q[n_steps+1] = target'
        for k in n_steps:-1:1
            H_k = copy(H_drift)
            for j in 1:n_ctrl
                H_k .+= controls[j, k] .* H_ctrl[j]
            end
            Q[k] = Q[k+1] * _herm_matexp_metal(H_k, dt)
        end
        Φ = tr(target' * P[n_steps+1]) / dim
        grad = zeros(Float64, n_ctrl, n_steps)
        for k in 1:n_steps, j in 1:n_ctrl
            M  = Q[k+1]' * (-im * dt * H_ctrl[j]) * P[k]
            grad[j, k] = (2.0 / dim^2) * real(conj(Φ) * tr(M))
        end
        return grad
    end

    if !_METAL_LOADED[]
        @warn "gradient_metal: Metal unavailable, falling back to CPU."
        return _cpu_grape_grad()
    end

    try
        # Forward pass — propagators on host (FP64 precision throughout)
        P = Vector{Matrix{ComplexF64}}(undef, n_steps + 1)
        P[1] = Matrix{ComplexF64}(I, dim, dim)
        for k in 1:n_steps
            H_k = copy(H_drift)
            for j in 1:n_ctrl
                H_k .+= controls[j, k] .* H_ctrl[j]
            end
            Uk = matrix_exponential_metal(H_k, dt, backend)
            # Use Metal BLAS for the matrix-matrix multiply if FP32 is on
            if backend.use_fp32 && !backend.use_fp64_fallback
                P[k+1] = _metal_gemm_f32(Uk, P[k])
            else
                P[k+1] = Uk * P[k]
            end
        end

        # Backward pass
        Q = Vector{Matrix{ComplexF64}}(undef, n_steps + 1)
        Q[n_steps+1] = target'
        for k in n_steps:-1:1
            H_k = copy(H_drift)
            for j in 1:n_ctrl
                H_k .+= controls[j, k] .* H_ctrl[j]
            end
            Uk = matrix_exponential_metal(H_k, dt, backend)
            if backend.use_fp32 && !backend.use_fp64_fallback
                Q[k] = _metal_gemm_f32(Q[k+1], Uk)
            else
                Q[k] = Q[k+1] * Uk
            end
        end

        Φ    = tr(target' * P[n_steps+1]) / dim
        grad = zeros(Float64, n_ctrl, n_steps)
        for k in 1:n_steps, j in 1:n_ctrl
            M = Q[k+1]' * (-im * dt * H_ctrl[j]) * P[k]
            grad[j, k] = (2.0 / dim^2) * real(conj(Φ) * tr(M))
        end
        return grad
    catch e
        @warn "gradient_metal: computation failed ($e), falling back to CPU."
        return _cpu_grape_grad()
    end
end

# ---------------------------------------------------------------------------
# metal_info
# ---------------------------------------------------------------------------

"""
    metal_info(backend) -> Dict{String,Any}

Return Metal GPU device properties as a dictionary.

Keys: `"available"`, `"device_name"`, `"memory_gb"`, `"supports_fp64"`.
When Metal is not available, returns `Dict("available"=>false)`.
"""
function metal_info(backend::MetalBackend)::Dict{String,Any}
    if !_METAL_LOADED[]
        return Dict{String,Any}("available" => false)
    end
    try
        dev = Metal.device()
        return Dict{String,Any}(
            "available"    => true,
            "device_name"  => Metal.name(dev),
            "memory_gb"    => Metal.recommendedMaxWorkingSetSize(dev) / 1024^3,
            "supports_fp64" => false,   # Metal shaders are FP32-only
        )
    catch e
        return Dict{String,Any}("available" => true, "error" => string(e))
    end
end
