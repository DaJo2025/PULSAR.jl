"""
    MR/GRAPEState.jl

GRAPE kernel for state-transfer optimisation in Hilbert space, following the
Khaneja et al. (2005) formulation adapted for ensemble averaging.

The waveform is normalised (dimensionless): actual RF contribution at step n
for ensemble member j is:

    H_ctrl(n,j) = pwr_levels[j] × Σ_k waveform[k,n] × operators[k]

Fidelity and gradient are averaged over ALL ensemble members
(n_drifts × n_pwr_levels × n_state_pairs).

## CPU parallelization (default, `ctrl.backend = :cpu`)

The outer loop over (drift, pwr) pairs is parallelised with `Threads.@threads`.
Each thread owns its own result buffers — no locks, no shared mutable state.
Start Julia with multiple threads to exploit this:

    julia -t auto --project=. examples/NMR/nmr_01_broadband_90_pulse.jl

## GPU parallelization (`ctrl.backend = :metal` or `:cuda`)

Matrix exponentials are computed on CPU (LAPACK eigendecomposition, fast for
small Hermitian matrices). The propagator matrices are then transferred to the
GPU in a batch, and all forward/backward propagation matrix-vector products run
on the GPU. The full sequences of intermediate states are transferred back to
CPU in one bulk call for gradient accumulation.

  - `:metal` — Apple Silicon, via Metal.jl. Uses Complex{Float32} arithmetic.
               Effective for matrix dim ≥ 32 (roughly a 3-qubit system).
  - `:cuda`  — NVIDIA GPU, via CUDA.jl. Uses ComplexF64 arithmetic.
               Effective for matrix dim ≥ 16.

Both GPU paths fall back to CPU with a runtime warning if the package is not
loaded.

Reference:
  Khaneja et al., "Optimal control of coupled spin dynamics: design of NMR pulse
  sequences by gradient ascent algorithms", J. Magn. Reson. 172 (2005) 296–305.
"""

using LinearAlgebra

# ─── Public entry point ───────────────────────────────────────────────────────

"""
    grape_state_kernel(waveform, ctrl) → (fidelity, grad)

Compute ensemble-averaged state-transfer fidelity and its gradient w.r.t. the
normalised control waveform.

Dispatches to a CPU-threaded or GPU implementation based on `ctrl.backend`
(`:cpu`, `:metal`, or `:cuda`).

# Arguments
- `waveform::Matrix{Float64}` — shape `[n_ctrl × n_t]`; normalised amplitudes
  in `[ctrl.l_bound, ctrl.u_bound]`.
- `ctrl::MRControl` — complete problem specification.

# Returns
- `fidelity::Float64` — ensemble-averaged fidelity ∈ [0, 1].
- `grad::Matrix{Float64}` — gradient ∂F/∂waveform, same shape as `waveform`.

# Physics (per ensemble member)
1. Forward:  ψ[n+1] = exp(−i H[n] dt[n]) ψ[n]
2. Overlap:  z = ⟨ψ_targ | ψ(T)⟩
3. Fidelity: |z|²  (`:square`) or Re(z) (`:real`)
4. Backward: λ[n] = P[n]† λ[n+1],   λ[N+1] = ψ_targ
5. Gradient (`:square`): ∂F/∂w[k,n] = 2 dt pwr Im(z̄ ⟨λ[n+1]|Op[k]|ψ[n]⟩)
"""
function grape_state_kernel(waveform::Matrix{Float64}, ctrl)
    backend = _resolve_backend_auto(ctrl, waveform)
    if backend == :cpu
        return _grape_cpu(waveform, ctrl)
    elseif backend == :metal
        return _grape_gpu(waveform, ctrl, :metal)
    elseif backend == :cuda
        return _grape_gpu(waveform, ctrl, :cuda)
    else
        throw(ArgumentError(
            "Unknown backend ':$(backend)'. Use :cpu (default), :metal, :cuda, or :auto."))
    end
end

# Lesson 4: when `ctrl.backend == :auto`, consult `plan_hybrid_execution`
# to pick `:cpu` or `:gpu` based on Hilbert-space dim, n_timesteps and the
# planner's CPU/GPU thresholds. Maps `:gpu` to whichever GPU extension is
# loaded (CUDA preferred over Metal). `:hybrid` falls back to `:cpu` here
# because the GRAPE state kernel does not currently support split execution.
function _resolve_backend_auto(ctrl, waveform::Matrix{Float64})
    backend = getfield(ctrl, :backend)
    backend === :auto || return backend

    cuda_ok = _CUDA_LOADED[]
    metal_ok = _METAL_LOADED[]
    gpu_ok = cuda_ok || metal_ok

    dim = length(ctrl.rho_init[1])
    n_t = size(waveform, 2)
    n_c = size(waveform, 1)
    planner = HybridExecutionPlanner()
    decision = plan_hybrid_execution(dim, n_t, gpu_ok, planner;
                                      op = "gradient", n_controls = n_c)
    if decision === :gpu
        return cuda_ok ? :cuda : :metal
    else
        return :cpu
    end
end

# ─── CPU implementation (multi-threaded) ─────────────────────────────────────

"""
    _grape_cpu(waveform, ctrl) → (fidelity, grad)

Thread-parallel GRAPE kernel.  The outer loop over `(drift, pwr)` ensemble
pairs is distributed across Julia threads via `Threads.@threads`.

Each thread writes into its own pre-allocated fidelity scalar and gradient
matrix (indices into `fid_buf` / `grad_buf`), so there is no shared mutable
state and no synchronisation needed beyond the final reduction.

## Optimisations (vs naïve implementation)
- **Hermitian eigendecomposition** for all matrix exponentials: `_expm_neg_i`
  uses `eigen(Hermitian(H))` rather than Padé approximation, giving 3–5×
  faster propagator building for the small matrices common in NMR/EPR.
- **In-place `mul!`** for forward and backward propagation: forward states
  stored as columns of a pre-allocated `[dim × (n_t+1)]` matrix; backward
  co-states likewise.  Eliminates `O(n_t × n_pairs × n_outer)` vector
  allocations per call.
- **Pre-allocated `tmp_vec`** for the `Op[k] × ψ_n` inner products in the
  gradient loop: one scratch vector per thread, reused across all `(n, k)`.
"""
function _grape_cpu(waveform::Matrix{Float64}, ctrl)
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_drift = length(ctrl.drifts)
    n_pwr   = length(ctrl.pwr_levels)
    n_pairs = length(ctrl.rho_init)
    N_ens   = n_drift * n_pwr * n_pairs
    dim     = length(ctrl.rho_init[1])

    # Flatten (drift, pwr) pairs → one entry per independent propagator set.
    ens_pairs = [(H, p) for H in ctrl.drifts for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)

    # Per-task result buffers — indexed by task, not thread.
    fid_buf  = zeros(Float64, n_outer)
    grad_buf = [zeros(Float64, n_ctrl, n_t) for _ in 1:n_outer]

    # ── Pre-allocate per-thread scratch to eliminate hot-loop allocations ─────
    # Indexed by Threads.threadid() (1-based, static scheduler).
    n_th = Threads.nthreads()
    # Propagator matrices: one pre-allocated [dim×dim] matrix per (thread, step)
    Ps_bufs   = [[Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_t]
                  for _ in 1:n_th]
    # Scratch for in-place H assembly and expm intermediate: 2 per thread
    H_bufs    = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]
    VD_bufs   = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th]
    # State / co-state trajectory matrices
    ψ_bufs    = [Matrix{ComplexF64}(undef, dim, n_t + 1) for _ in 1:n_th]
    λ_bufs    = [Matrix{ComplexF64}(undef, dim, n_t + 1) for _ in 1:n_th]
    # Gradient inner-product scratch vector
    tmp_bufs  = [Vector{ComplexF64}(undef, dim) for _ in 1:n_th]

    # ── Prevent BLAS/LAPACK from spawning extra threads inside @threads ────────
    # Without this, each eigen() call multiplies the OS thread count by
    # BLAS_threads, causing severe oversubscription and near-zero speedup.
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads :static for idx in 1:n_outer
        tid = Threads.threadid()

        H_drift, pwr   = ens_pairs[idx]
        Ps             = Ps_bufs[tid]
        H_buf          = H_bufs[tid]
        VD_buf         = VD_bufs[tid]
        ψ_fwd          = ψ_bufs[tid]
        λ_mat          = λ_bufs[tid]
        tmp            = tmp_bufs[tid]
        grad_local     = grad_buf[idx]

        # ── Build propagators in-place: H_buf = H_drift + pwr·Σ w[k,n]·Op[k] ─
        for n in 1:n_t
            H_buf .= H_drift
            for k in 1:n_ctrl
                @. H_buf += pwr * waveform[k, n] * ctrl.operators[k]
            end
            _expm_neg_i_into!(Ps[n], H_buf, ctrl.pulse_dt[n], VD_buf)
        end

        fid_local = 0.0

        for s in 1:n_pairs
            ψ₀   = ctrl.rho_init[s]
            ψ_tg = ctrl.rho_targ[s]

            # ── Forward propagation ───────────────────────────────────────────
            ψ_fwd[:, 1] .= ψ₀
            for n in 1:n_t
                mul!(view(ψ_fwd, :, n + 1), Ps[n], view(ψ_fwd, :, n))
            end

            # ── Overlap and fidelity ──────────────────────────────────────────
            ψ_T      = view(ψ_fwd, :, n_t + 1)
            z        = state_overlap(ψ_tg, ψ_T)
            F_member = state_fidelity(ψ_tg, ψ_T; type = ctrl.fidelity)
            fid_local += F_member

            # ── Backward propagation ──────────────────────────────────────────
            λ_mat[:, n_t + 1] .= ψ_tg
            for n in n_t:-1:1
                mul!(view(λ_mat, :, n), Ps[n]', view(λ_mat, :, n + 1))
            end

            # ── Gradient accumulation ─────────────────────────────────────────
            for n in 1:n_t
                dt_n = ctrl.pulse_dt[n]
                λ_v  = view(λ_mat, :, n + 1)
                ψ_v  = view(ψ_fwd, :, n)
                for k in 1:n_ctrl
                    mul!(tmp, ctrl.operators[k], ψ_v)
                    inner = dot(λ_v, tmp)
                    grad_local[k, n] += fidelity_grad_prefactor(
                        z, inner, dt_n * pwr; type = ctrl.fidelity)
                end
            end
        end

        fid_buf[idx] = fid_local
    end  # @threads

    BLAS.set_num_threads(old_blas)

    # ── Reduction ─────────────────────────────────────────────────────────────
    fidelity = sum(fid_buf) / N_ens
    grad     = zeros(Float64, n_ctrl, n_t)
    for g in grad_buf
        grad .+= g
    end
    grad ./= N_ens

    # ── Penalty terms ─────────────────────────────────────────────────────────
    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)

    return fidelity, grad
end

# ─── GPU implementation (Metal / CUDA) ───────────────────────────────────────

"""
    _grape_gpu(waveform, ctrl, gpu_sym) → (fidelity, grad)

GPU-accelerated GRAPE kernel. Dispatches to `_grape_gpu_kernel` after
resolving the backend package and element type.

Falls back to CPU (`_grape_cpu`) if the requested package is not loaded.
"""
function _grape_gpu(waveform::Matrix{Float64}, ctrl, gpu_sym::Symbol)
    if gpu_sym == :metal && !_METAL_LOADED[]
        @warn "PULSAR: Metal.jl not loaded — falling back to CPU backend" maxlog=1
        return _grape_cpu(waveform, ctrl)
    end
    if gpu_sym == :cuda && !_CUDA_LOADED[]
        @warn "PULSAR: CUDA.jl not loaded — falling back to CPU backend" maxlog=1
        return _grape_cpu(waveform, ctrl)
    end

    if gpu_sym == :metal
        Metal  = Base.loaded_modules[Base.identify_package("Metal")]
        T      = Complex{Float32}
        to_gpu = x -> Metal.mtl(T.(x))
    else  # :cuda
        CUDA   = Base.loaded_modules[Base.identify_package("CUDA")]
        T      = ComplexF64
        to_gpu = x -> CUDA.cu(x)
    end

    return _grape_gpu_kernel(waveform, ctrl, to_gpu, T)
end

function _grape_gpu_kernel(waveform::Matrix{Float64}, ctrl, to_gpu, ::Type{T}) where {T}
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_drift = length(ctrl.drifts)
    n_pwr   = length(ctrl.pwr_levels)
    n_pairs = length(ctrl.rho_init)
    N_ens   = n_drift * n_pwr * n_pairs
    dim     = size(ctrl.drifts[1], 1)

    # Flatten (drift, pwr) into a single ensemble index.
    ens_pairs = [(H, p) for H in ctrl.drifts for p in ctrl.pwr_levels]
    n_outer   = length(ens_pairs)   # n_drift × n_pwr

    # ── Build ALL propagators on CPU in parallel, then single bulk GPU transfer
    Ps_cpu    = Array{T}(undef, dim, dim, n_outer, n_t)
    PsAdj_cpu = Array{T}(undef, dim, dim, n_outer, n_t)

    n_th_gpu  = Threads.nthreads()
    H_bufs_g  = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_gpu]
    VD_bufs_g = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_gpu]
    P_bufs_g  = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_gpu]

    old_blas_g = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads :static for idx in 1:n_outer
        tid     = Threads.threadid()
        H_drift, pwr = ens_pairs[idx]
        H_buf   = H_bufs_g[tid]
        VD_buf  = VD_bufs_g[tid]
        P_buf   = P_bufs_g[tid]
        for n in 1:n_t
            H_buf .= H_drift
            for k in 1:n_ctrl
                @. H_buf += pwr * waveform[k, n] * ctrl.operators[k]
            end
            _expm_neg_i_into!(P_buf, H_buf, ctrl.pulse_dt[n], VD_buf)
            Ps_cpu[:, :, idx, n]    = T.(P_buf)
            PsAdj_cpu[:, :, idx, n] = T.(P_buf')
        end
    end
    BLAS.set_num_threads(old_blas_g)

    # Single host→device transfer: [dim × dim × n_outer × n_t] in one call.
    # Reshape to 3D so each [:,:,j] slice is a concrete GPU matrix view.
    Ps_gpu    = to_gpu(reshape(Ps_cpu,    dim, dim, n_outer * n_t))
    PsAdj_gpu = to_gpu(reshape(PsAdj_cpu, dim, dim, n_outer * n_t))

    fidelity_sum = 0.0
    grad_sum     = zeros(Float64, n_ctrl, n_t)

    for s in 1:n_pairs
        ψ0_T  = Vector{T}(ctrl.rho_init[s])
        ψtg_T = Vector{T}(ctrl.rho_targ[s])

        # ── Batched forward propagation on GPU ─────────────────────────────
        # ψ_batch[:, i] = state for ensemble member i, shape [dim × n_outer].
        # All n_t+1 snapshots written into a single pre-allocated
        # [dim × n_outer × (n_t+1)] GPU tensor.
        ψ_batch    = to_gpu(repeat(reshape(ψ0_T, dim, 1), 1, n_outer))  # [dim × n_outer]
        ψ_fwd_all  = similar(ψ_batch, dim, n_outer, n_t + 1)
        ψ_fwd_all[:, :, 1] = ψ_batch
        for n in 1:n_t
            Ps_n    = Ps_gpu[:, :, (n - 1) * n_outer + 1 : n * n_outer]
            ψ_batch = reshape(sum(Ps_n .* reshape(ψ_batch, 1, dim, n_outer); dims = 2), dim, n_outer)
            ψ_fwd_all[:, :, n + 1] = ψ_batch
        end

        # ── Overlap: single transfer of final states ────────────────────────
        ψ_final_cpu = ComplexF64.(Array(ψ_fwd_all[:, :, n_t + 1]))  # [dim × n_outer]
        ψ_targ_cpu  = ctrl.rho_targ[s]
        z_vec = [state_overlap(ψ_targ_cpu, ψ_final_cpu[:, i]) for i in 1:n_outer]
        for i in 1:n_outer
            fidelity_sum += state_fidelity(ψ_targ_cpu, ψ_final_cpu[:, i]; type = ctrl.fidelity)
        end

        # ── Batched backward propagation on GPU ────────────────────────────
        λ_batch  = to_gpu(repeat(reshape(ψtg_T, dim, 1), 1, n_outer))  # [dim × n_outer]
        λ_all    = similar(λ_batch, dim, n_outer, n_t + 1)
        λ_all[:, :, n_t + 1] = λ_batch
        for n in n_t:-1:1
            PsAdj_n = PsAdj_gpu[:, :, (n - 1) * n_outer + 1 : n * n_outer]
            λ_batch  = reshape(sum(PsAdj_n .* reshape(λ_batch, 1, dim, n_outer); dims = 2), dim, n_outer)
            λ_all[:, :, n] = λ_batch
        end

        # ── Single bulk host←device transfer ────────────────────────────────
        # ψ_all_cpu[:, i, n]  = ψ[n]   (state before applying P[n])
        # λ_all_cpu[:, i, n]  = λ[n+1] (co-state used in gradient at step n)
        ψ_all_cpu = ComplexF64.(Array(ψ_fwd_all[:, :, 1:n_t]))        # [dim, n_outer, n_t]
        λ_all_cpu = ComplexF64.(Array(λ_all[:, :, 2:n_t + 1]))        # [dim, n_outer, n_t]

        # ── Gradient accumulation on CPU (parallel over time steps) ────────
        Threads.@threads for n in 1:n_t
            dt_n = ctrl.pulse_dt[n]
            tmp  = Vector{ComplexF64}(undef, dim)   # thread-local scratch
            for i in 1:n_outer
                _, pwr_i = ens_pairs[i]
                z_i   = z_vec[i]
                λ_v   = view(λ_all_cpu, :, i, n)
                ψ_v   = view(ψ_all_cpu, :, i, n)
                for k in 1:n_ctrl
                    mul!(tmp, ctrl.operators[k], ψ_v)
                    inner = dot(λ_v, tmp)
                    grad_sum[k, n] += fidelity_grad_prefactor(
                        z_i, inner, dt_n * pwr_i; type = ctrl.fidelity)
                end
            end
        end

    end  # state pairs

    fidelity = fidelity_sum / N_ens
    grad     = grad_sum ./ N_ens

    fidelity = _apply_penalties!(fidelity, grad, waveform, ctrl)

    return fidelity, grad
end

# ─── Forward-only fidelity (no gradient) ─────────────────────────────────────

"""
    fidelity_forward(waveform, drifts, pwr_levels, operators,
                     rho_init, rho_targ, pulse_dt;
                     fidelity_type = :real,
                     backend       = get_device()) → Float64

GPU-accelerated ensemble-averaged state-transfer fidelity (forward pass only).

Use this as the objective function for **derivative-free** optimizers
(GA, PSO, DE, CMA-ES, Nelder-Mead, …) when a GPU backend is active.
It uses the same batched-GPU propagation strategy as `grape_state_kernel`
(CPU-parallel matrix exponentials → single bulk GPU transfer → GPU matvec
forward pass) without computing the gradient.

# Arguments
- `waveform`   : `[n_ctrl × n_t]` normalised control amplitudes.
- `drifts`     : `Vector{Matrix}` of drift Hamiltonians (one per ensemble member).
- `pwr_levels` : `Vector{Float64}` of RF power levels in rad/s.
- `operators`  : `Vector{Matrix}` of dimensionless control operators.
- `rho_init`   : `Vector{Vector}` of initial states.
- `rho_targ`   : `Vector{Vector}` of target states.
- `pulse_dt`   : `AbstractVector{Float64}` of time-step durations (s).
- `fidelity_type` : `:real` or `:square`.
- `backend`    : `:cpu`, `:metal`, or `:cuda`.

# Returns
`Float64` — ensemble-averaged fidelity (average over all drift × pwr × state pairs).
"""
function fidelity_forward(
    waveform   :: Matrix{Float64},
    drifts     :: Vector{<:Matrix},
    pwr_levels :: Vector{Float64},
    operators  :: Vector{<:Matrix},
    rho_init   :: Vector{<:AbstractVector},
    rho_targ   :: Vector{<:AbstractVector},
    pulse_dt   :: AbstractVector{Float64};
    fidelity_type :: Symbol = :real,
    backend       :: Symbol = get_device(),
)
    if backend == :metal && _METAL_LOADED[]
        Metal  = Base.loaded_modules[Base.identify_package("Metal")]
        T      = Complex{Float32}
        to_gpu = x -> Metal.mtl(T.(x))
        return _fidelity_gpu(waveform, drifts, pwr_levels, operators,
                             rho_init, rho_targ, pulse_dt, fidelity_type, to_gpu, T)
    elseif backend == :cuda && _CUDA_LOADED[]
        CUDA   = Base.loaded_modules[Base.identify_package("CUDA")]
        T      = ComplexF64
        to_gpu = x -> CUDA.cu(x)
        return _fidelity_gpu(waveform, drifts, pwr_levels, operators,
                             rho_init, rho_targ, pulse_dt, fidelity_type, to_gpu, T)
    else
        if backend ∉ (:cpu, :metal, :cuda)
            @warn "fidelity_forward: unknown backend :$backend — using CPU" maxlog=1
        elseif backend != :cpu
            @warn "fidelity_forward: $backend package not loaded — using CPU" maxlog=1
        end
        return _fidelity_cpu(waveform, drifts, pwr_levels, operators,
                             rho_init, rho_targ, pulse_dt, fidelity_type)
    end
end

# CPU implementation (multi-threaded over ensemble members, in-place propagation)
function _fidelity_cpu(waveform, drifts, pwr_levels, operators,
                       rho_init, rho_targ, pulse_dt, fidelity_type)
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    n_pairs = length(rho_init)
    ens_pairs = [(H, p) for H in drifts for p in pwr_levels]
    n_outer   = length(ens_pairs)
    N_ens     = n_outer * n_pairs

    n_th_f   = Threads.nthreads()
    dim_f    = size(drifts[1], 1)
    H_bufs_f = [Matrix{ComplexF64}(undef, dim_f, dim_f) for _ in 1:n_th_f]
    VD_bufs_f= [Matrix{ComplexF64}(undef, dim_f, dim_f) for _ in 1:n_th_f]
    Ps_bufs_f= [[Matrix{ComplexF64}(undef, dim_f, dim_f) for _ in 1:n_t] for _ in 1:n_th_f]

    fid_buf  = zeros(Float64, n_outer)
    old_blas_f = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads :static for idx in 1:n_outer
        tid     = Threads.threadid()
        H_drift, pwr = ens_pairs[idx]
        H_buf   = H_bufs_f[tid]
        VD_buf  = VD_bufs_f[tid]
        Ps      = Ps_bufs_f[tid]
        for n in 1:n_t
            H_buf .= H_drift
            for k in 1:n_ctrl
                @. H_buf += pwr * waveform[k, n] * operators[k]
            end
            _expm_neg_i_into!(Ps[n], H_buf, pulse_dt[n], VD_buf)
        end
        local_fid = 0.0
        for s in 1:n_pairs
            ψ_cur = copy(rho_init[s])
            ψ_nxt = similar(ψ_cur)
            for n in 1:n_t
                mul!(ψ_nxt, Ps[n], ψ_cur)
                ψ_cur, ψ_nxt = ψ_nxt, ψ_cur
            end
            local_fid += state_fidelity(rho_targ[s], ψ_cur; type = fidelity_type)
        end
        fid_buf[idx] = local_fid
    end
    BLAS.set_num_threads(old_blas_f)
    return sum(fid_buf) / N_ens
end

# GPU implementation (batched forward pass only, no snapshots needed)
function _fidelity_gpu(waveform, drifts, pwr_levels, operators,
                       rho_init, rho_targ, pulse_dt, fidelity_type, to_gpu, ::Type{T}) where {T}
    n_ctrl  = size(waveform, 1)
    n_t     = size(waveform, 2)
    dim     = size(drifts[1], 1)
    n_pairs = length(rho_init)
    ens_pairs = [(H, p) for H in drifts for p in pwr_levels]
    n_outer   = length(ens_pairs)
    N_ens     = n_outer * n_pairs

    # Build propagators on CPU in parallel, single bulk GPU transfer
    Ps_cpu   = Array{T}(undef, dim, dim, n_outer, n_t)
    n_th_fg  = Threads.nthreads()
    H_bfg    = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_fg]
    VD_bfg   = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_fg]
    P_bfg    = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_th_fg]
    old_fg   = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Threads.@threads :static for idx in 1:n_outer
        tid     = Threads.threadid()
        H_drift, pwr = ens_pairs[idx]
        for n in 1:n_t
            H_bfg[tid] .= H_drift
            for k in 1:n_ctrl
                @. H_bfg[tid] += pwr * waveform[k, n] * operators[k]
            end
            _expm_neg_i_into!(P_bfg[tid], H_bfg[tid], pulse_dt[n], VD_bfg[tid])
            Ps_cpu[:, :, idx, n] = T.(P_bfg[tid])
        end
    end
    BLAS.set_num_threads(old_fg)
    Ps_gpu = to_gpu(reshape(Ps_cpu, dim, dim, n_outer * n_t))

    fidelity_sum = 0.0
    for s in 1:n_pairs
        ψ0_T    = Vector{T}(rho_init[s])
        ψ_batch = to_gpu(repeat(reshape(ψ0_T, dim, 1), 1, n_outer))  # [dim × n_outer]
        # Forward propagation: one GPU broadcast-reduce per time step.
        # No snapshot storage needed — only the final state is used.
        for n in 1:n_t
            Ps_n    = Ps_gpu[:, :, (n - 1) * n_outer + 1 : n * n_outer]
            ψ_batch = reshape(sum(Ps_n .* reshape(ψ_batch, 1, dim, n_outer); dims = 2), dim, n_outer)
        end
        ψ_final = ComplexF64.(Array(ψ_batch))  # single GPU→CPU transfer
        for i in 1:n_outer
            fidelity_sum += state_fidelity(rho_targ[s], ψ_final[:, i]; type = fidelity_type)
        end
    end
    return fidelity_sum / N_ens
end

# ─── Shared helpers ───────────────────────────────────────────────────────────

# Apply penalty terms; modifies grad in-place, returns updated fidelity scalar.
function _apply_penalties!(fidelity::Float64, grad::Matrix{Float64}, waveform, ctrl)
    for (pen, weight) in zip(ctrl.penalties, ctrl.p_weights)
        pen == :none && continue
        weight ≈ 0.0 && continue
        F_pen, G_pen = penalty_value_and_gradient(waveform;
            type    = pen,
            weight  = Float64(weight),
            l_bound = ctrl.l_bound,
            u_bound = ctrl.u_bound)
        fidelity -= F_pen
        grad     .-= G_pen
    end
    return fidelity
end

# Build control Hamiltonian at time step n.
@inline function _ctrl_hamiltonian(waveform, operators, n, n_ctrl)
    H = waveform[1, n] .* operators[1]
    for k in 2:n_ctrl
        H = H .+ waveform[k, n] .* operators[k]
    end
    return H
end

# _expm_neg_i / _expm_neg_i_into! are defined in Core/Propagators.jl.

# Penalty functions are provided by Core/Fidelity.jl:
#   penalty_value_and_gradient(w; type, weight, l_bound, u_bound, dt)

# ─── Per-sample kernel (for EnsembleObjective per-sample closures) ───────────

"""
    grape_state_kernel_single(waveform, H_drift, pwr, rho_init, rho_targ, ctrl)
        → (F::Float64, grad::Matrix{Float64})

Single-sample Hilbert-space GRAPE kernel for one `(H_drift, pwr, ρ_init, ρ_targ)`
combination. Returns the fidelity and gradient without averaging and without
applying penalties — the caller (an `EnsembleObjective` aggregator) is
responsible for both.

Used by [`build_ensemble_from_mrcontrol`](@ref) to back `:worst_case` / `:cvar`
aggregators that cannot use the batched [`grape_state_kernel`](@ref) fast path.
For `:mean` aggregation prefer `grape_state_kernel` — it is the same computation
but with threaded/GPU batching over all `(drift, pwr, state_pair)` members.
"""
function grape_state_kernel_single(waveform::Matrix{Float64},
                                    H_drift::AbstractMatrix,
                                    pwr::Real,
                                    rho_init::AbstractVector,
                                    rho_targ::AbstractVector,
                                    ctrl)
    n_ctrl = size(waveform, 1)
    n_t    = size(waveform, 2)
    dim    = length(rho_init)

    Ps     = [Matrix{ComplexF64}(undef, dim, dim) for _ in 1:n_t]
    H_buf  = Matrix{ComplexF64}(undef, dim, dim)
    VD_buf = Matrix{ComplexF64}(undef, dim, dim)
    ψ_fwd  = Matrix{ComplexF64}(undef, dim, n_t + 1)
    λ_mat  = Matrix{ComplexF64}(undef, dim, n_t + 1)
    tmp    = Vector{ComplexF64}(undef, dim)

    for n in 1:n_t
        H_buf .= H_drift
        for k in 1:n_ctrl
            @. H_buf += pwr * waveform[k, n] * ctrl.operators[k]
        end
        _expm_neg_i_into!(Ps[n], H_buf, ctrl.pulse_dt[n], VD_buf)
    end

    ψ_fwd[:, 1] .= rho_init
    for n in 1:n_t
        mul!(view(ψ_fwd, :, n + 1), Ps[n], view(ψ_fwd, :, n))
    end

    ψ_T = view(ψ_fwd, :, n_t + 1)
    z   = state_overlap(rho_targ, ψ_T)
    F   = state_fidelity(rho_targ, ψ_T; type = ctrl.fidelity)

    λ_mat[:, n_t + 1] .= rho_targ
    for n in n_t:-1:1
        mul!(view(λ_mat, :, n), Ps[n]', view(λ_mat, :, n + 1))
    end

    grad = zeros(Float64, n_ctrl, n_t)
    for n in 1:n_t
        dt_n = ctrl.pulse_dt[n]
        λ_v  = view(λ_mat, :, n + 1)
        ψ_v  = view(ψ_fwd, :, n)
        for k in 1:n_ctrl
            mul!(tmp, ctrl.operators[k], ψ_v)
            inner = dot(λ_v, tmp)
            grad[k, n] = fidelity_grad_prefactor(
                z, inner, dt_n * pwr; type = ctrl.fidelity)
        end
    end

    return F, grad
end
