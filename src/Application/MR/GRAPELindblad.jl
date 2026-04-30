"""
    MR/GRAPELindblad.jl

GRAPE kernel for state-transfer optimisation in Liouville space (open systems
governed by the Lindblad master equation).

Mirrors the structure of `MR/GRAPEState.jl` but operates on vectorised density
matrices σ = vec(ρ) ∈ ℂ^(N²) instead of pure-state vectors ψ ∈ ℂ^N.

## Algorithm (per ensemble member j, per state pair s)

  Forward propagation:
    σ[0]   = σ_init[s]  (= vec(ρ_init[s]))
    σ[n+1] = Φ[n] · σ[n]
    Φ[n]   = exp( (𝓛_drift[j] + pwr · Σ_k w[k,n] · 𝓛_ctrl[k]) · dt[n] )

  Overlap and fidelity:
    z = ⟨σ_targ[s] | σ[N]⟩          (Hilbert–Schmidt inner product)
    F = |z|²   (:square)  or  Re(z)  (:real)

  Backward propagation:
    λ[N+1] = σ_targ[s]
    λ[n]   = Φ[n]† · λ[n+1]

  Gradient:
    ∂F/∂w[k,n] = lindblad_grad_prefactor(z, ⟨λ[n+1]|𝓛_ctrl[k]|σ[n]⟩, dt[n]·pwr; type)

  where `lindblad_grad_prefactor` uses Re() instead of the Hilbert-space Im(),
  because 𝓛_ctrl = −i(commutator) already absorbs the −i propagator factor.

## Key differences from GRAPEState.jl

| Aspect             | Hilbert space          | Liouville space              |
|--------------------|------------------------|------------------------------|
| State dimension    | N                      | N²                           |
| Propagator type    | exp(−iH dt)  unitary   | exp(𝓛 dt)  generally non-unitary |
| Propagator builder | Hermitian eigendecomp  | LinearAlgebra.exp (Padé/Schur)|
| Grad prefactor     | Im(z̄·inner)            | Re(z̄·inner)                  |
| Memory per step    | N × N (scratch matrix) | N² × N² (propagator stored)   |

For spin-1/2 (N=2): Liouville space is 4×4 — negligible cost.
For 2 spins (N=4): 16×16 — still fast.

## Parallelisation

The outer loop over (drift, pwr) pairs is parallelised with `Threads.@threads`,
identical to `_grape_cpu` in GRAPEState.jl.

Reference:
  Optimal control of open quantum systems via the GRAPE algorithm:
  Schulte-Herbrüggen et al., "Optimal control for generating quantum gates
  in open dissipative systems", J. Phys. B 44 (2011) 154013.
"""

using LinearAlgebra

# ─── Public entry point ───────────────────────────────────────────────────────

"""
    grape_lindblad_kernel(waveform, ctrl) → (fidelity, grad)

Compute ensemble-averaged state-transfer fidelity and its gradient w.r.t. the
normalised control waveform for open quantum systems (Lindblad dynamics).

# Arguments
- `waveform::Matrix{Float64}` — `[n_ctrl × n_t]` normalised amplitudes.
- `ctrl::LindbladMRControl`   — complete open-system problem specification,
  including precomputed Liouvillians `_L_drifts`, `_L_controls`, vectorised
  initial/target states `_sigma_init`, `_sigma_targ`.

# Returns
- `fidelity::Float64` — ensemble-averaged fidelity ∈ [0, 1].
- `grad::Matrix{Float64}` — gradient ∂F/∂waveform, same shape as `waveform`.

# Physics
The fidelity for one (drift j, pwr p, state pair s) member is:

    F_jps = |⟨σ_targ[s] | σ_jps(T)⟩|²   (`:square`)

where σ = vec(ρ) evolves under exp(𝓛[n,j,p] dt[n]) at each step.
The returned fidelity and gradient are the mean over all N_ens = n_drifts × n_pwr × n_pairs.
"""
function grape_lindblad_kernel(waveform::Matrix{Float64}, ctrl)
    if ctrl.backend == :cpu
        return _grape_lindblad_cpu(waveform, ctrl)
    elseif ctrl.backend ∈ (:metal, :cuda)
        return _grape_lindblad_gpu(waveform, ctrl, ctrl.backend)
    else
        throw(ArgumentError(
            "Unknown backend ':$(ctrl.backend)'. Use :cpu (default), :metal, or :cuda."))
    end
end

# ─── CPU implementation (multi-threaded) ─────────────────────────────────────

function _grape_lindblad_cpu(waveform::Matrix{Float64}, ctrl)
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_drift = length(ctrl._L_drifts)
    n_pwr   = length(ctrl.pwr_levels)
    n_pairs = length(ctrl._sigma_init)
    N_ens   = n_drift * n_pwr * n_pairs
    N2      = ctrl._liouville_dim          # N² (state vector length)

    # Flatten (drift, pwr) pairs into a single ensemble index
    ens_pairs = [(L_d, p) for L_d in ctrl._L_drifts for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)

    # Per-task result buffers (one entry per (drift, pwr) pair, no lock needed)
    fid_buf  = zeros(Float64, n_outer)
    grad_buf = [zeros(Float64, n_ctrl, n_t) for _ in 1:n_outer]

    # Per-thread scratch — avoids allocations inside the hot loop.
    # Use maxthreadid() so buffers cover any tid that may be reported
    # under Julia 1.12+ scheduling.
    n_th = Threads.maxthreadid()

    # Propagator matrices: [N²×N²] per (thread, step)
    Phi_bufs  = [[Matrix{ComplexF64}(undef, N2, N2) for _ in 1:n_t]
                  for _ in 1:n_th]
    # Liouvillian scratch (assembled as: L_drift + Σ w[k,n]·pwr·L_ctrl[k])
    L_bufs    = [Matrix{ComplexF64}(undef, N2, N2) for _ in 1:n_th]
    # Forward state trajectory: N2 × (n_t+1) per thread
    σ_bufs    = [Matrix{ComplexF64}(undef, N2, n_t + 1) for _ in 1:n_th]
    # Backward co-state trajectory
    λ_bufs    = [Matrix{ComplexF64}(undef, N2, n_t + 1) for _ in 1:n_th]
    # Gradient inner-product scratch
    tmp_bufs  = [Vector{ComplexF64}(undef, N2) for _ in 1:n_th]

    # Prevent BLAS from spawning extra threads inside @threads
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads :static for idx in 1:n_outer
        tid = Threads.threadid()

        L_drift, pwr = ens_pairs[idx]
        Phi          = Phi_bufs[tid]
        L_buf        = L_bufs[tid]
        σ_fwd        = σ_bufs[tid]
        λ_mat        = λ_bufs[tid]
        tmp          = tmp_bufs[tid]
        grad_local   = grad_buf[idx]

        # ── Build propagators: Φ[n] = exp(𝓛_total[n] dt[n]) ─────────────────
        # 𝓛_total[n] = L_drift + pwr · Σ_k w[k,n] · L_ctrl[k]
        # For each step n, assemble into L_buf, then take matrix exponential.
        for n in 1:n_t
            L_buf .= L_drift
            for k in 1:n_ctrl
                @. L_buf += pwr * waveform[k, n] * ctrl._L_controls[k]
            end
            # exp of a general complex matrix (Padé approximation via LAPACK)
            Phi[n] .= exp(L_buf .* ctrl.pulse_dt[n])
        end

        fid_local = 0.0

        for s in 1:n_pairs
            σ₀   = ctrl._sigma_init[s]
            σ_tg = ctrl._sigma_targ[s]

            # ── Forward propagation ───────────────────────────────────────────
            σ_fwd[:, 1] .= σ₀
            for n in 1:n_t
                mul!(view(σ_fwd, :, n + 1), Phi[n], view(σ_fwd, :, n))
            end

            # ── Overlap and fidelity ──────────────────────────────────────────
            σ_T  = view(σ_fwd, :, n_t + 1)
            z    = ComplexF64(dot(σ_tg, σ_T))     # ⟨σ_targ | σ(T)⟩

            fid_local += if ctrl.fidelity == :square
                real(conj(z) * z)                  # |z|²
            else  # :real
                real(z)
            end

            # ── Backward propagation ──────────────────────────────────────────
            λ_mat[:, n_t + 1] .= σ_tg
            for n in n_t:-1:1
                mul!(view(λ_mat, :, n), Phi[n]', view(λ_mat, :, n + 1))
            end

            # ── Gradient accumulation ─────────────────────────────────────────
            for n in 1:n_t
                dt_pwr = ctrl.pulse_dt[n] * pwr
                λ_v    = view(λ_mat, :, n + 1)
                σ_v    = view(σ_fwd, :, n)
                for k in 1:n_ctrl
                    # tmp = 𝓛_ctrl[k] · σ[n]
                    mul!(tmp, ctrl._L_controls[k], σ_v)
                    inner = ComplexF64(dot(λ_v, tmp))  # ⟨λ[n+1] | 𝓛_ctrl[k] | σ[n]⟩
                    grad_local[k, n] += lindblad_grad_prefactor(
                        z, inner, dt_pwr; type = ctrl.fidelity)
                end
            end
        end  # state pairs

        fid_buf[idx] = fid_local
    end  # @threads

    BLAS.set_num_threads(old_blas)

    # ── Reduction ──────────────────────────────────────────────────────────────
    fidelity = sum(fid_buf) / N_ens
    grad     = zeros(Float64, n_ctrl, n_t)
    for g in grad_buf
        grad .+= g
    end
    grad ./= N_ens

    # ── Penalty terms (reused from GRAPEState._apply_penalties!) ──────────────
    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)

    return fidelity, grad
end

# ─── GPU entry point ──────────────────────────────────────────────────────────
#
# Mirrors _grape_gpu / _grape_gpu_kernel in GRAPEState.jl.
#
# Two execution strategies selected automatically based on available GPU memory:
#
#   Chunked mode  — propagators for `chunk_size` ensemble members are built on
#                   CPU in parallel, bulk-transferred to GPU, then forward and
#                   backward propagation run as batched GPU matvecs.  Fidelity
#                   and gradient accumulate across chunks.  Used whenever at
#                   least one chunk fits in GPU memory.
#
#   Streaming mode — all ensemble members processed simultaneously, but ONE
#                   time step at a time.  GPU holds only the current propagator
#                   batch [N2×N2×n_outer] and current state batch [N2×n_outer].
#                   Forward-pass snapshots are cached on CPU.  Propagators are
#                   also cached on CPU (built once, reused for backward pass).
#                   GPU memory is O(N2² × n_outer) — a few MB even for 4 spins.
#                   Used when even a single chunk member's full n_t propagators
#                   exceed available GPU memory.
#
# Precision:
#   ctrl.precision = :f64  → ComplexF64  (default; CUDA only)
#   ctrl.precision = :f32  → Complex{Float32}  (halves memory; both backends)
#   Metal always uses Complex{Float32} regardless of ctrl.precision.

"""
    _grape_lindblad_gpu(waveform, ctrl, gpu_sym) → (fidelity, grad)

Package-check and element-type dispatcher for GPU Lindblad GRAPE.
Falls back to CPU with a warning if the required package is not loaded.
"""
function _grape_lindblad_gpu(waveform::Matrix{Float64}, ctrl, gpu_sym::Symbol)
    if gpu_sym == :metal && !_METAL_LOADED[]
        @warn "Pulsar: Metal.jl not loaded — falling back to CPU backend for Lindblad" maxlog=1
        return _grape_lindblad_cpu(waveform, ctrl)
    end
    if gpu_sym == :cuda && !_CUDA_LOADED[]
        @warn "Pulsar: CUDA.jl not loaded — falling back to CPU backend for Lindblad" maxlog=1
        return _grape_lindblad_cpu(waveform, ctrl)
    end

    if gpu_sym == :metal
        Metal    = Base.loaded_modules[Base.identify_package("Metal")]
        T        = Complex{Float32}          # Metal always Float32
        to_gpu   = x -> Metal.mtl(T.(x))
        free_mem = floor(Int, Sys.free_memory() * 0.4)   # unified memory estimate
    else  # :cuda
        CUDA     = Base.loaded_modules[Base.identify_package("CUDA")]
        T        = ctrl.precision == :f32 ? Complex{Float32} : ComplexF64
        to_gpu   = x -> CUDA.cu(T.(x))
        free_mem = Int(CUDA.available_memory())
    end

    return _grape_lindblad_gpu_kernel(waveform, ctrl, to_gpu, free_mem, T)
end

# ─── GPU kernel router: choose chunked vs streaming ───────────────────────────

function _grape_lindblad_gpu_kernel(
        waveform::Matrix{Float64}, ctrl, to_gpu, free_mem::Int, ::Type{T}) where T
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    N2      = ctrl._liouville_dim
    n_pairs = length(ctrl._sigma_init)
    ens_pairs = [(L_d, p) for L_d in ctrl._L_drifts for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)
    N_ens     = n_outer * n_pairs

    # Memory required on GPU per ensemble member for the full n_t propagators:
    #   Phi + PhiAdj: 2 × N2² × n_t × sizeof(T)
    bytes_per_member = 2 * N2^2 * n_t * sizeof(T)
    budget           = floor(Int, free_mem * 0.75)   # keep 25% headroom

    streaming  = bytes_per_member > budget
    chunk_size = streaming ? n_outer :
                 min(n_outer, max(1, floor(Int, budget / bytes_per_member)))

    if ctrl.verbose
        if streaming
            @printf("[Lindblad GPU] Memory: need %d MB/member, free ≈ %d MB — using temporal streaming mode.\n",
                    bytes_per_member >> 20, budget >> 20)
        elseif chunk_size < n_outer
            n_chunks = ceil(Int, n_outer / chunk_size)
            prec_str = T == ComplexF64 ? "f64" : "f32"
            @printf("[Lindblad GPU] Ensemble chunking: %d member(s)/chunk x %d chunk(s) (%d MB/chunk, T=%s)\n",
                    chunk_size, n_chunks, bytes_per_member * chunk_size >> 20, prec_str)
        else
            prec_str = T == ComplexF64 ? "f64" : "f32"
            @printf("[Lindblad GPU] Single bulk transfer (%d MB, T=%s)\n",
                    bytes_per_member * n_outer >> 20, prec_str)
        end
    end

    if streaming
        return _grape_lindblad_streaming(waveform, ctrl, ens_pairs,
                                          to_gpu, T,
                                          n_ctrl, n_t, N2, n_outer, n_pairs, N_ens)
    else
        return _grape_lindblad_chunked(waveform, ctrl, ens_pairs,
                                        to_gpu, T, chunk_size,
                                        n_ctrl, n_t, N2, n_outer, n_pairs, N_ens)
    end
end

# ─── Chunked GPU kernel ───────────────────────────────────────────────────────
#
# Propagators for `chunk_size` ensemble members are built on CPU in parallel,
# bulk-transferred to GPU, then forward + backward propagation run as batched
# GPU matvecs (same broadcasting pattern as the Hilbert-space GPU kernel).
# Fidelity and gradient accumulate across chunks.

function _grape_lindblad_chunked(
        waveform, ctrl, ens_pairs, to_gpu, ::Type{T}, chunk_size,
        n_ctrl, n_t, N2, n_outer, n_pairs, N_ens) where T

    fidelity_sum = 0.0
    grad_sum     = zeros(Float64, n_ctrl, n_t)

    n_th   = Threads.maxthreadid()
    L_bufs = [Matrix{ComplexF64}(undef, N2, N2) for _ in 1:n_th]

    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    for chunk_start in 1:chunk_size:n_outer
        chunk_end = min(chunk_start + chunk_size - 1, n_outer)
        n_chunk   = chunk_end - chunk_start + 1

        # ── Build propagators on CPU in parallel ─────────────────────────────
        # Layout: [N2 × N2 × n_chunk × n_t] → reshaped to [N2 × N2 × n_chunk*n_t]
        # Column-major: index (a,b,local_i,n) → slice (n-1)*n_chunk + local_i
        Phi_cpu    = Array{T}(undef, N2, N2, n_chunk, n_t)
        PhiAdj_cpu = Array{T}(undef, N2, N2, n_chunk, n_t)

        Threads.@threads :static for local_i in 1:n_chunk
            tid               = Threads.threadid()
            L_buf             = L_bufs[tid]
            global_i          = chunk_start + local_i - 1
            L_drift, pwr      = ens_pairs[global_i]
            for n in 1:n_t
                L_buf .= L_drift
                for k in 1:n_ctrl
                    @. L_buf += pwr * waveform[k, n] * ctrl._L_controls[k]
                end
                Phi_n = exp(L_buf .* ctrl.pulse_dt[n])
                Phi_cpu[:, :, local_i, n]    = T.(Phi_n)
                PhiAdj_cpu[:, :, local_i, n] = T.(Phi_n')
            end
        end

        # ── Single bulk host→GPU transfer ────────────────────────────────────
        Phi_gpu    = to_gpu(reshape(Phi_cpu,    N2, N2, n_chunk * n_t))
        PhiAdj_gpu = to_gpu(reshape(PhiAdj_cpu, N2, N2, n_chunk * n_t))

        for s in 1:n_pairs
            σ0_T  = Vector{T}(ctrl._sigma_init[s])
            σtg_T = Vector{T}(ctrl._sigma_targ[s])

            # ── Batched forward propagation on GPU ────────────────────────────
            # σ_batch: [N2 × n_chunk]; snapshots stored in σ_fwd_all
            σ_batch   = to_gpu(repeat(reshape(σ0_T, N2, 1), 1, n_chunk))
            σ_fwd_all = similar(σ_batch, N2, n_chunk, n_t + 1)
            σ_fwd_all[:, :, 1] = σ_batch
            for n in 1:n_t
                Phi_n   = Phi_gpu[:, :, (n-1)*n_chunk+1 : n*n_chunk]
                σ_batch = reshape(
                    sum(Phi_n .* reshape(σ_batch, 1, N2, n_chunk); dims=2),
                    N2, n_chunk)
                σ_fwd_all[:, :, n+1] = σ_batch
            end

            # ── Overlaps (final states to CPU) ────────────────────────────────
            σ_final_cpu = ComplexF64.(Array(σ_fwd_all[:, :, n_t+1]))
            σ_targ_cpu  = ctrl._sigma_targ[s]
            z_vec = [ComplexF64(dot(σ_targ_cpu, σ_final_cpu[:, i]))
                     for i in 1:n_chunk]
            for i in 1:n_chunk
                z = z_vec[i]
                fidelity_sum += ctrl.fidelity == :square ?
                                real(conj(z) * z) : real(z)
            end

            # ── Batched backward propagation on GPU ───────────────────────────
            λ_batch = to_gpu(repeat(reshape(σtg_T, N2, 1), 1, n_chunk))
            λ_all   = similar(λ_batch, N2, n_chunk, n_t + 1)
            λ_all[:, :, n_t+1] = λ_batch
            for n in n_t:-1:1
                PhiAdj_n = PhiAdj_gpu[:, :, (n-1)*n_chunk+1 : n*n_chunk]
                λ_batch  = reshape(
                    sum(PhiAdj_n .* reshape(λ_batch, 1, N2, n_chunk); dims=2),
                    N2, n_chunk)
                λ_all[:, :, n] = λ_batch
            end

            # ── Single bulk GPU→CPU transfer ──────────────────────────────────
            σ_all_cpu = ComplexF64.(Array(σ_fwd_all[:, :, 1:n_t]))     # [N2,n_chunk,n_t]
            λ_all_cpu = ComplexF64.(Array(λ_all[:, :, 2:n_t+1]))       # [N2,n_chunk,n_t]

            # ── Gradient accumulation (CPU, parallel over time steps) ─────────
            # Thread t owns exclusive time steps → no race on grad_sum[:,n]
            tmp_bufs_g = [Vector{ComplexF64}(undef, N2) for _ in 1:n_th]
            Threads.@threads :static for n in 1:n_t
                tmp = tmp_bufs_g[Threads.threadid()]
                for local_i in 1:n_chunk
                    global_i = chunk_start + local_i - 1
                    _, pwr_i = ens_pairs[global_i]
                    z_i      = z_vec[local_i]
                    dt_pwr   = ctrl.pulse_dt[n] * pwr_i
                    λ_v      = view(λ_all_cpu, :, local_i, n)
                    σ_v      = view(σ_all_cpu, :, local_i, n)
                    for k in 1:n_ctrl
                        mul!(tmp, ctrl._L_controls[k], σ_v)
                        inner = ComplexF64(dot(λ_v, tmp))
                        grad_sum[k, n] += lindblad_grad_prefactor(
                            z_i, inner, dt_pwr; type=ctrl.fidelity)
                    end
                end
            end
        end  # state pairs
    end  # chunks

    BLAS.set_num_threads(old_blas)

    fidelity = fidelity_sum / N_ens
    grad     = grad_sum ./ N_ens
    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)
    return fidelity, grad
end

# ─── Per-sample kernel (for EnsembleObjective per-sample closures) ───────────

"""
    grape_lindblad_kernel_single(waveform, L_drift, pwr, σ_init, σ_targ, ctrl)
        → (F::Float64, grad::Matrix{Float64})

Single-sample Liouville-space GRAPE kernel for one
`(L_drift, pwr, σ_init, σ_targ)` combination. Returns the fidelity and gradient
without averaging and without applying penalties.

Used by [`build_ensemble_from_mrcontrol`](@ref) to back `:worst_case` / `:cvar`
aggregators. For `:mean` aggregation prefer [`grape_lindblad_kernel`](@ref) —
the batched CPU/GPU fast path.
"""
function grape_lindblad_kernel_single(waveform::Matrix{Float64},
                                       L_drift::AbstractMatrix,
                                       pwr::Real,
                                       sigma_init::AbstractVector,
                                       sigma_targ::AbstractVector,
                                       ctrl)
    n_ctrl = size(waveform, 1)
    n_t    = size(waveform, 2)
    N2     = length(sigma_init)

    Phi    = [Matrix{ComplexF64}(undef, N2, N2) for _ in 1:n_t]
    L_buf  = Matrix{ComplexF64}(undef, N2, N2)
    σ_fwd  = Matrix{ComplexF64}(undef, N2, n_t + 1)
    λ_mat  = Matrix{ComplexF64}(undef, N2, n_t + 1)
    tmp    = Vector{ComplexF64}(undef, N2)

    for n in 1:n_t
        L_buf .= L_drift
        for k in 1:n_ctrl
            @. L_buf += pwr * waveform[k, n] * ctrl._L_controls[k]
        end
        Phi[n] .= exp(L_buf .* ctrl.pulse_dt[n])
    end

    σ_fwd[:, 1] .= sigma_init
    for n in 1:n_t
        mul!(view(σ_fwd, :, n + 1), Phi[n], view(σ_fwd, :, n))
    end

    σ_T = view(σ_fwd, :, n_t + 1)
    z   = ComplexF64(dot(sigma_targ, σ_T))
    F   = ctrl.fidelity == :square ? real(conj(z) * z) : real(z)

    λ_mat[:, n_t + 1] .= sigma_targ
    for n in n_t:-1:1
        mul!(view(λ_mat, :, n), Phi[n]', view(λ_mat, :, n + 1))
    end

    grad = zeros(Float64, n_ctrl, n_t)
    for n in 1:n_t
        dt_pwr = ctrl.pulse_dt[n] * pwr
        λ_v    = view(λ_mat, :, n + 1)
        σ_v    = view(σ_fwd, :, n)
        for k in 1:n_ctrl
            mul!(tmp, ctrl._L_controls[k], σ_v)
            inner = ComplexF64(dot(λ_v, tmp))
            grad[k, n] = lindblad_grad_prefactor(
                z, inner, dt_pwr; type = ctrl.fidelity)
        end
    end

    return F, grad
end

# ─── Streaming GPU kernel ─────────────────────────────────────────────────────
#
# One time step processed per GPU launch: GPU holds only [N2×N2×n_outer] (one
# propagator batch) and [N2×n_outer] (current state batch) — O(N2²×n_outer) GPU
# memory regardless of n_t.
#
# All propagators are built once on CPU (parallel) and cached in CPU RAM.
# Forward-pass snapshots are cached on CPU.
# On Apple Silicon (unified memory) there is no physical transfer overhead.
# On CUDA the n_t PCIe round-trips are significant for large systems; a runtime
# warning is emitted so the user can switch to :cpu if preferred.

function _grape_lindblad_streaming(
        waveform, ctrl, ens_pairs, to_gpu, ::Type{T},
        n_ctrl, n_t, N2, n_outer, n_pairs, N_ens) where T

    fidelity_sum = 0.0
    grad_sum     = zeros(Float64, n_ctrl, n_t)

    n_th   = Threads.maxthreadid()
    L_bufs = [Matrix{ComplexF64}(undef, N2, N2) for _ in 1:n_th]

    # ── Build ALL propagators on CPU (once, parallel) ─────────────────────────
    # Cached in CPU RAM: [N2 × N2 × n_outer × n_t] × 2 (Phi + PhiAdj)
    # For 4 spins, n_outer=75, n_t=500, Float64: ~2 × 256² × 75 × 500 × 8 ≈ 4 GB.
    # If this exceeds available CPU RAM the OS will swap — warn at construction
    # time is not feasible here, so we proceed and let Julia raise OutOfMemory.
    Phi_cpu    = Array{T}(undef, N2, N2, n_outer, n_t)
    PhiAdj_cpu = Array{T}(undef, N2, N2, n_outer, n_t)

    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads :static for idx in 1:n_outer
        tid          = Threads.threadid()
        L_buf        = L_bufs[tid]
        L_drift, pwr = ens_pairs[idx]
        for n in 1:n_t
            L_buf .= L_drift
            for k in 1:n_ctrl
                @. L_buf += pwr * waveform[k, n] * ctrl._L_controls[k]
            end
            Phi_n = exp(L_buf .* ctrl.pulse_dt[n])
            Phi_cpu[:, :, idx, n]    = T.(Phi_n)
            PhiAdj_cpu[:, :, idx, n] = T.(Phi_n')
        end
    end

    BLAS.set_num_threads(old_blas)

    # Per-ensemble-member gradient scratch (one column per member → no races)
    grad_step = zeros(Float64, n_ctrl, n_outer)
    tmp_bufs_s = [Vector{ComplexF64}(undef, N2) for _ in 1:n_th]

    for s in 1:n_pairs
        σ0_T  = Vector{T}(ctrl._sigma_init[s])
        σtg_T = Vector{T}(ctrl._sigma_targ[s])

        # ── Streaming forward pass ────────────────────────────────────────────
        # GPU: σ_batch [N2 × n_outer] + Phi_n [N2 × N2 × n_outer] (one step)
        # CPU: all snapshots σ_fwd_cpu [N2 × n_outer × (n_t+1)]
        σ_fwd_cpu = Array{T}(undef, N2, n_outer, n_t + 1)
        σ_batch   = to_gpu(repeat(reshape(σ0_T, N2, 1), 1, n_outer))
        σ_fwd_cpu[:, :, 1] = T.(Array(σ_batch))

        for n in 1:n_t
            Phi_n_gpu = to_gpu(Phi_cpu[:, :, :, n])          # [N2 × N2 × n_outer]
            σ_batch   = reshape(
                sum(Phi_n_gpu .* reshape(σ_batch, 1, N2, n_outer); dims=2),
                N2, n_outer)
            σ_fwd_cpu[:, :, n+1] = T.(Array(σ_batch))
        end

        # ── Overlaps ──────────────────────────────────────────────────────────
        σ_final_cpu = ComplexF64.(σ_fwd_cpu[:, :, n_t+1])
        σ_targ_cpu  = ctrl._sigma_targ[s]
        z_vec = [ComplexF64(dot(σ_targ_cpu, σ_final_cpu[:, i])) for i in 1:n_outer]
        for i in 1:n_outer
            z = z_vec[i]
            fidelity_sum += ctrl.fidelity == :square ? real(conj(z) * z) : real(z)
        end

        # ── Streaming backward pass + gradient ───────────────────────────────
        # Stream PhiAdj one step at a time; keep λ_batch on GPU.
        # Transfer λ[n+1] to CPU each step for the gradient dot products.
        λ_batch = to_gpu(repeat(reshape(σtg_T, N2, 1), 1, n_outer))

        for n in n_t:-1:1
            # λ[n+1] on CPU for gradient
            λ_n1_cpu = ComplexF64.(Array(λ_batch))            # [N2 × n_outer]
            σ_n_cpu  = ComplexF64.(σ_fwd_cpu[:, :, n])        # [N2 × n_outer]

            # Gradient: parallel over ensemble members (each writes to own column)
            fill!(grad_step, 0.0)
            Threads.@threads :static for i in 1:n_outer
                tmp    = tmp_bufs_s[Threads.threadid()]
                _, pwr_i = ens_pairs[i]
                z_i    = z_vec[i]
                dt_pwr = ctrl.pulse_dt[n] * pwr_i
                λ_v    = view(λ_n1_cpu, :, i)
                σ_v    = view(σ_n_cpu,  :, i)
                for k in 1:n_ctrl
                    mul!(tmp, ctrl._L_controls[k], σ_v)
                    inner = ComplexF64(dot(λ_v, tmp))
                    grad_step[k, i] = lindblad_grad_prefactor(
                        z_i, inner, dt_pwr; type=ctrl.fidelity)
                end
            end
            # Reduce over ensemble members into grad_sum[:,n]
            for i in 1:n_outer
                @inbounds for k in 1:n_ctrl
                    grad_sum[k, n] += grad_step[k, i]
                end
            end

            # Backward propagate λ on GPU (stream one PhiAdj step)
            PhiAdj_n_gpu = to_gpu(PhiAdj_cpu[:, :, :, n])    # [N2 × N2 × n_outer]
            λ_batch      = reshape(
                sum(PhiAdj_n_gpu .* reshape(λ_batch, 1, N2, n_outer); dims=2),
                N2, n_outer)
        end
    end  # state pairs

    fidelity = fidelity_sum / N_ens
    grad     = grad_sum ./ N_ens
    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)
    return fidelity, grad
end
