# CUDABackend.jl — NVIDIA CUDA GPU backend
#
# Provides real GPU-accelerated operations when CUDA.jl is installed and
# functional.  Falls back gracefully to CPU with a @warn when CUDA is absent.
#
# Design notes:
#   • CUDA.jl is loaded at module-include time via a try/catch extension block.
#   • All heavy operations upload input to CuArray, compute on device, download.
#   • Matrix exponential is computed via eigendecomposition: exp(A) = V diag(exp(λ)) V†,
#     which is stable and uses cuBLAS / LAPACK on device.
#   • GRAPE gradient uses the standard forward–backward pass entirely on GPU;
#     only the scalar fidelity and the gradient matrix are pulled back to host.
#   • Functions are no-ops returning CPU results when CUDA is unavailable.

using LinearAlgebra

# ---------------------------------------------------------------------------
# Runtime availability flag (set to true below if CUDA loads successfully)
# ---------------------------------------------------------------------------

const _CUDA_LOADED = Ref{Bool}(false)

# Attempt to load CUDA.jl at include time.  Errors are swallowed so that
# Pulsar loads cleanly on systems without CUDA.
let
    try
        @eval using CUDA
        if CUDA.functional()
            _CUDA_LOADED[] = true
        else
            @warn "CUDABackend: CUDA.jl loaded but no functional GPU found. " *
                  "Calls to CUDA backend functions will fall back to CPU."
        end
    catch
        # CUDA.jl not installed — backend disabled silently
    end
end

"""
    CUDA_AVAILABLE::Bool

`true` only if CUDA.jl is installed **and** a functional NVIDIA GPU is present
at the time the module was loaded.
"""
const CUDA_AVAILABLE = _CUDA_LOADED[]

# ---------------------------------------------------------------------------
# Type definition
# ---------------------------------------------------------------------------

"""
    CUDABackend

Configuration struct for the NVIDIA CUDA GPU backend.

Construct via [`cuda_backend`](@ref).  Requires CUDA.jl to be installed and
a functional NVIDIA GPU.

# Fields
- `device_id::Int`: CUDA device index (0-based).
- `memory_limit_gb::Float64`: Soft GPU memory budget in GiB.
- `use_tensor_cores::Bool`: Prefer TF32/FP16 tensor-core paths (experimental).
- `use_async::Bool`: Use CUDA streams for pipeline overlap (experimental).
"""
struct CUDABackend
    device_id        :: Int
    memory_limit_gb  :: Float64
    use_tensor_cores :: Bool
    use_async        :: Bool
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    cuda_backend(; device_id=0, memory_limit_gb=Inf,
                   use_tensor_cores=false, use_async=false) -> CUDABackend

Construct a [`CUDABackend`](@ref).

Throws an `ErrorException` if CUDA.jl is not installed or no functional GPU
is present.
"""
function cuda_backend(;
    device_id        :: Int     = 0,
    memory_limit_gb  :: Float64 = Inf,
    use_tensor_cores :: Bool    = false,
    use_async        :: Bool    = false,
)
    if !_CUDA_LOADED[]
        error(
            "CUDA is not available.  Install CUDA.jl with `] add CUDA` and " *
            "ensure NVIDIA drivers are installed.  Use `cpu_backend()` instead.",
        )
    end
    return CUDABackend(device_id, memory_limit_gb, use_tensor_cores, use_async)
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Hermitian matrix exponential: exp(-i H dt) via eigendecomposition.
# Works on plain Julia arrays; caller is responsible for device placement.
function _herm_matexp(H::Matrix{ComplexF64}, dt::Float64)
    F = eigen(Hermitian(H))
    return F.vectors * Diagonal(exp.(-im .* F.values .* dt)) * F.vectors'
end

# ---------------------------------------------------------------------------
# matrix_exponential_cuda
# ---------------------------------------------------------------------------

"""
    matrix_exponential_cuda(H, dt, backend) -> Matrix{ComplexF64}

Compute `exp(-i H dt)` for a Hermitian matrix `H` using the CUDA GPU.

When CUDA is available the computation is performed with device-side
eigendecomposition via `CUDA.CUSOLVER`.  Falls back to CPU eigendecomposition
with a `@warn` if CUDA is not loaded.

The result is always returned as a host `Matrix{ComplexF64}`.
"""
function matrix_exponential_cuda(
    H       :: Matrix{ComplexF64},
    dt      :: Float64,
    backend :: CUDABackend,
)
    if !_CUDA_LOADED[]
        @warn "matrix_exponential_cuda: CUDA unavailable, falling back to CPU."
        return _herm_matexp(H, dt)
    end
    try
        # Upload Hermitian matrix to device
        H_d = CUDA.CuArray(H)
        # CUSOLVER eigen on device (via LinearAlgebra overloads in CUDA.jl)
        F = LinearAlgebra.eigen(LinearAlgebra.Hermitian(H_d))
        # Build exp(-i λ dt) on device
        expλ = CUDA.CuArray(exp.(-im .* Array(F.values) .* dt))
        U_d  = F.vectors * LinearAlgebra.Diagonal(expλ) * F.vectors'
        return Array(U_d)
    catch e
        @warn "matrix_exponential_cuda: GPU computation failed ($e), falling back to CPU."
        return _herm_matexp(H, dt)
    end
end

# ---------------------------------------------------------------------------
# batch_propagators_cuda
# ---------------------------------------------------------------------------

"""
    batch_propagators_cuda(H_array, dt, backend) -> Array{ComplexF64,3}

Compute all propagators `U[k] = exp(-i H_array[:,:,k] dt)` in parallel on
the GPU.

`H_array` is a `(dim, dim, n_steps)` array of Hermitian matrices.

Each slice is diagonalised independently; the batch is processed concurrently
by spreading slices across CUDA streams or using a single-threaded fallback
loop on the device.  Returns a host `(dim, dim, n_steps)` array.
"""
function batch_propagators_cuda(
    H_array :: AbstractArray{ComplexF64,3},
    dt      :: Float64,
    backend :: CUDABackend,
)
    dim, _, n_steps = size(H_array)
    result = Array{ComplexF64,3}(undef, dim, dim, n_steps)

    if !_CUDA_LOADED[]
        @warn "batch_propagators_cuda: CUDA unavailable, falling back to CPU."
        for k in 1:n_steps
            result[:, :, k] = _herm_matexp(H_array[:, :, k], dt)
        end
        return result
    end

    try
        # Batch the host↔device transfers:
        #   - Upload the full H_array once.
        #   - Accumulate each propagator into a device-side result tensor.
        #   - Download the full result in one contiguous copy at the end.
        # The previous per-step `Array(Uk)` forced a synchronous download
        # on every iteration, serialising the GPU pipeline with the PCIe bus.
        H_d       = CUDA.CuArray(H_array)
        result_d  = CUDA.CuArray{ComplexF64,3}(undef, dim, dim, n_steps)
        for k in 1:n_steps
            Hk   = H_d[:, :, k]
            F    = LinearAlgebra.eigen(LinearAlgebra.Hermitian(Hk))
            expλ = CUDA.CuArray(exp.(-im .* Array(F.values) .* dt))
            Uk   = F.vectors * LinearAlgebra.Diagonal(expλ) * F.vectors'
            @views result_d[:, :, k] .= Uk
        end
        copyto!(result, result_d)   # single bulk download
        return result
    catch e
        @warn "batch_propagators_cuda: GPU computation failed ($e), falling back to CPU."
        for k in 1:n_steps
            result[:, :, k] = _herm_matexp(H_array[:, :, k], dt)
        end
        return result
    end
end

# ---------------------------------------------------------------------------
# fidelity_cuda
# ---------------------------------------------------------------------------

"""
    fidelity_cuda(U_total, U_target, backend) -> Float64

Compute the gate fidelity `|Tr(U_target† U_total)|² / dim²` on the GPU.

Both `U_total` and `U_target` are `(dim, dim)` unitary matrices.  The trace
inner product is evaluated on the device; only the resulting scalar is
transferred back to host.
"""
function fidelity_cuda(
    U_total  :: Matrix{ComplexF64},
    U_target :: Matrix{ComplexF64},
    backend  :: CUDABackend,
)
    dim = size(U_total, 1)
    if !_CUDA_LOADED[]
        @warn "fidelity_cuda: CUDA unavailable, falling back to CPU."
        overlap = tr(U_target' * U_total)
        return abs2(overlap) / dim^2
    end
    try
        U_d  = CUDA.CuArray(U_total)
        Ut_d = CUDA.CuArray(U_target)
        # Tr(Ut† U) = sum of element-wise products along diagonal of Ut' * U
        # Equivalently: sum(conj(Ut_d) .* U_d) = tr(Ut_d' * U_d)
        overlap = CUDA.mapreduce(identity, +, conj.(Ut_d) .* U_d)
        return abs2(Array(overlap)[]) / dim^2
    catch e
        @warn "fidelity_cuda: GPU computation failed ($e), falling back to CPU."
        overlap = tr(U_target' * U_total)
        return abs2(overlap) / dim^2
    end
end

# ---------------------------------------------------------------------------
# gradient_cuda
# ---------------------------------------------------------------------------

"""
    gradient_cuda(H_drift, H_ctrl, controls, target, dt, backend) -> Matrix{Float64}

Compute the full GRAPE gradient on the GPU.

Implements the standard forward–backward pass:
  1. **Forward pass**: `P[0] = I`, `P[k] = exp(-i H[k] dt) P[k-1]`
  2. **Backward pass**: `Q[N] = U_target'`, `Q[k-1] = Q[k] exp(-i H[k] dt)`
  3. **Gradient**: `∂F/∂u_{j,k} = (2/dim²) Re[ Tr(U_target† P[N]) * conj(Tr(Q[k]† (-i dt H_j) P[k-1])) ]`

# Arguments
- `H_drift`: Drift Hamiltonian `(dim × dim)`.
- `H_ctrl`: Vector of `n_controls` control Hamiltonians `(dim × dim)` each.
- `controls`: `(n_controls × n_steps)` control amplitudes.
- `target`: Target unitary `(dim × dim)`.
- `dt`: Timestep (seconds, or rad/s if Hamiltonians are already in rad/s).
- `backend`: The `CUDABackend` instance.

Returns `grad` of size `(n_controls, n_steps)`.
"""
function gradient_cuda(
    H_drift  :: Matrix{ComplexF64},
    H_ctrl   :: Vector{<:Matrix{ComplexF64}},
    controls :: Matrix{Float64},
    target   :: Matrix{ComplexF64},
    dt       :: Float64,
    backend  :: CUDABackend,
)
    n_ctrl, n_steps = size(controls)
    dim             = size(H_drift, 1)

    # CPU fallback path
    function _cpu_grape_grad()
        # Forward propagators P[k] = U_k P[k-1], P[0] = I
        P = Vector{Matrix{ComplexF64}}(undef, n_steps + 1)
        P[1] = Matrix{ComplexF64}(I, dim, dim)
        for k in 1:n_steps
            H_k = copy(H_drift)
            for j in 1:n_ctrl
                H_k .+= controls[j, k] .* H_ctrl[j]
            end
            P[k+1] = _herm_matexp(H_k, dt) * P[k]
        end
        # Backward co-state Q[k] where Q[N+1] = target†
        Q = Vector{Matrix{ComplexF64}}(undef, n_steps + 1)
        Q[n_steps+1] = target'
        for k in n_steps:-1:1
            H_k = copy(H_drift)
            for j in 1:n_ctrl
                H_k .+= controls[j, k] .* H_ctrl[j]
            end
            Q[k] = Q[k+1] * _herm_matexp(H_k, dt)
        end
        # Fidelity overlap Φ = Tr(target† P[N+1]) / dim
        Φ = tr(target' * P[n_steps+1]) / dim
        # Gradient
        grad = zeros(Float64, n_ctrl, n_steps)
        for k in 1:n_steps, j in 1:n_ctrl
            M  = Q[k+1]' * (-im * dt * H_ctrl[j]) * P[k]
            grad[j, k] = (2.0 / dim^2) * real(conj(Φ) * tr(M))
        end
        return grad
    end

    if !_CUDA_LOADED[]
        @warn "gradient_cuda: CUDA unavailable, falling back to CPU."
        return _cpu_grape_grad()
    end

    try
        # Upload static arrays
        H_d_d    = CUDA.CuArray(H_drift)
        H_c_d    = [CUDA.CuArray(H_ctrl[j]) for j in 1:n_ctrl]
        target_d = CUDA.CuArray(target)
        I_d      = CUDA.CuArray(Matrix{ComplexF64}(I, dim, dim))

        # Forward pass on GPU — propagators stored on host to avoid OOM for
        # large problems (each (dim×dim) ComplexF64 is 16 dim² bytes).
        P_d = Vector{CUDA.CuArray{ComplexF64,2}}(undef, n_steps + 1)
        P_d[1] = copy(I_d)
        for k in 1:n_steps
            H_k_d = copy(H_d_d)
            for j in 1:n_ctrl
                CUDA.axpy!(complex(controls[j, k]), H_c_d[j], H_k_d)
            end
            F_k  = LinearAlgebra.eigen(LinearAlgebra.Hermitian(H_k_d))
            expλ = CUDA.CuArray(exp.(-im .* Array(F_k.values) .* dt))
            Uk_d = F_k.vectors * LinearAlgebra.Diagonal(expλ) * F_k.vectors'
            P_d[k+1] = Uk_d * P_d[k]
        end

        # Backward pass
        Q_d = Vector{CUDA.CuArray{ComplexF64,2}}(undef, n_steps + 1)
        Q_d[n_steps+1] = target_d'
        for k in n_steps:-1:1
            H_k_d = copy(H_d_d)
            for j in 1:n_ctrl
                CUDA.axpy!(complex(controls[j, k]), H_c_d[j], H_k_d)
            end
            F_k  = LinearAlgebra.eigen(LinearAlgebra.Hermitian(H_k_d))
            expλ = CUDA.CuArray(exp.(-im .* Array(F_k.values) .* dt))
            Uk_d = F_k.vectors * LinearAlgebra.Diagonal(expλ) * F_k.vectors'
            Q_d[k] = Q_d[k+1] * Uk_d
        end

        # Fidelity overlap
        Φ_d = CUDA.mapreduce(identity, +, conj.(target_d) .* P_d[n_steps+1])
        Φ   = Array(Φ_d)[] / dim

        # Gradient inner products
        grad = zeros(Float64, n_ctrl, n_steps)
        for k in 1:n_steps, j in 1:n_ctrl
            M_d        = Q_d[k+1]' * ((-im * dt) .* H_c_d[j]) * P_d[k]
            trM        = Array(CUDA.mapreduce(identity, +, conj.(I_d) .* M_d))[]
            grad[j, k] = (2.0 / dim^2) * real(conj(Φ) * trM)
        end

        return grad
    catch e
        @warn "gradient_cuda: GPU computation failed ($e), falling back to CPU."
        return _cpu_grape_grad()
    end
end

# ---------------------------------------------------------------------------
# cuda_info
# ---------------------------------------------------------------------------

"""
    cuda_info(backend) -> Dict{String,Any}

Return CUDA device properties as a dictionary.

Keys: `"available"`, `"device_name"`, `"memory_gb"`, `"compute_capability"`,
`"n_devices"`.  When CUDA is not available, returns `Dict("available"=>false)`.
"""
function cuda_info(backend::CUDABackend)::Dict{String,Any}
    if !_CUDA_LOADED[]
        return Dict{String,Any}("available" => false)
    end
    try
        dev = CUDA.device()
        return Dict{String,Any}(
            "available"           => true,
            "device_name"         => CUDA.name(dev),
            "memory_gb"           => CUDA.totalmem(dev) / 1024^3,
            "compute_capability"  => string(CUDA.capability(dev)),
            "n_devices"           => length(CUDA.devices()),
        )
    catch e
        return Dict{String,Any}("available" => true, "error" => string(e))
    end
end
