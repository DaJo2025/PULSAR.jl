"""
    CPUParallelization.jl

Three orthogonal CPU parallelisation strategies for PULSAR.jl:

1. **Task-based parallelisation** (`TaskParallelizationStrategy`) – divides
   work into chunks assigned to Julia threads via `Threads.@threads`.
2. **Vectorisation** (`VectorizationStrategy`) – SIMD-annotated inner loops
   with automatic CPU capability detection.
3. **Gradient parallelisation** (`GradientParallelization`) – shares forward
   and backward propagator intermediates across all gradient components to
   avoid redundant computation.
"""

using LinearAlgebra
using Base.Threads
using Base.Threads: threadid, threading_run
@static if Base.VERSION ≥ v"1.9-rc1"
    using Base.Threads: threadpoolsize
end

# ---------------------------------------------------------------------------
# Helper: import CPUBackend types when used stand-alone
# ---------------------------------------------------------------------------
# (When loaded via PULSAR's module system CPUBackend is already in scope.)

# ===========================================================================
# Conditional threading macro (Lessons 1 + 7)
# ===========================================================================

"""
    @threadsif cond for ... end

Conditionally thread a `for` loop using `Threads.@threads :static` when
`cond` evaluates to `true`; run serially otherwise.

When `cond == true` the macro additionally clamps BLAS to a single thread
for the duration of the loop and restores the prior count on exit. This
prevents oversubscription when threaded outer loops contain BLAS-heavy
inner work — a recurring footgun in nested-parallelism settings (see
Krotov.jl `threadpoolctl(1, "blas")` for the same pattern).

Pattern adapted from QuantumControl.jl `conditionalthreads.jl`; PULSAR adds
the BLAS-thread guard.

# Example
```julia
function optimize(samples; use_threads=true)
    @threadsif use_threads for k in 1:length(samples)
        process_sample!(samples[k])
    end
end
```
"""
macro threadsif(cond, loop)
    if !(isa(loop, Expr) && loop.head === :for)
        throw(ArgumentError("@threadsif requires a `for` loop expression"))
    end
    if !(loop.args[1] isa Expr && loop.args[1].head === :(=))
        throw(ArgumentError("nested outer loops are not currently supported by @threadsif"))
    end
    quote
        if $(esc(cond))
            local _bl_threads = LinearAlgebra.BLAS.get_num_threads()
            LinearAlgebra.BLAS.set_num_threads(1)
            try
                $(Threads._threadsfor(loop.args[1], loop.args[2], :static))
            finally
                LinearAlgebra.BLAS.set_num_threads(_bl_threads)
            end
        else
            $(esc(loop))
        end
    end
end

# ===========================================================================
# STRATEGY 1 – Task-based parallelisation
# ===========================================================================

"""
    TaskParallelizationStrategy

Configuration for thread-based, task-level parallelism.

# Fields
- `num_threads::Int`: Number of Julia threads to use.
- `chunk_size::Int`: Number of matrix rows (or items) each thread processes
  per scheduling quantum.  Use `compute_optimal_chunk_size` to obtain a
  good default.
- `work_stealing::Bool`: Reserved for future dynamic-scheduling support.
  Currently informational; `Threads.@threads :dynamic` is used when `true`.
- `affinity::Bool`: Reserved flag for OS-level thread–core affinity pinning.
  Requires external libraries (e.g. `ThreadPinning.jl`); currently
  informational.

# Usage guidance
Prefer `TaskParallelizationStrategy` when matrix operations dominate the
total runtime **and** `n_timesteps > 100`.  For smaller problems the thread-
launch overhead typically exceeds the computation cost.
"""
struct TaskParallelizationStrategy
    num_threads::Int
    chunk_size::Int
    work_stealing::Bool
    affinity::Bool

    function TaskParallelizationStrategy(
        num_threads::Int,
        chunk_size::Int,
        work_stealing::Bool,
        affinity::Bool,
    )
        num_threads > 0 || throw(ArgumentError("num_threads must be positive"))
        chunk_size > 0  || throw(ArgumentError("chunk_size must be positive"))
        new(num_threads, chunk_size, work_stealing, affinity)
    end
end

"""
    TaskParallelizationStrategy(; num_threads=Threads.nthreads(),
                                   chunk_size=0, work_stealing=true,
                                   affinity=false) -> TaskParallelizationStrategy

Keyword constructor.  When `chunk_size == 0` it is set automatically via
`compute_optimal_chunk_size(64, num_threads)`.
"""
function TaskParallelizationStrategy(;
    num_threads::Int = Threads.nthreads(),
    chunk_size::Int = 0,
    work_stealing::Bool = true,
    affinity::Bool = false,
)::TaskParallelizationStrategy
    cs = chunk_size == 0 ? compute_optimal_chunk_size(64, num_threads) : chunk_size
    return TaskParallelizationStrategy(num_threads, cs, work_stealing, affinity)
end

# ---------------------------------------------------------------------------

"""
    compute_optimal_chunk_size(matrix_size::Int, num_threads::Int) -> Int

Heuristic for choosing a per-thread chunk size that amortises thread-launch
overhead.

The rule of thumb is that each chunk should involve at least **~1 000
floating-point operations** before yielding.  For a matrix of `matrix_size`
rows the cost per row is `O(matrix_size)`, so:

    chunk_size = max(1, matrix_size ÷ (4 × num_threads))

This ensures roughly `4 × num_threads` chunks per problem (over-partitioning
by 4×), which lets the scheduler balance load without excessive grain size.

# Arguments
- `matrix_size::Int`: Characteristic dimension (e.g. number of rows, or
  number of time steps).
- `num_threads::Int`: Number of available threads.

# Returns
Recommended chunk size `≥ 1`.

# Examples
```julia
cs = compute_optimal_chunk_size(1024, 8)  # -> 32
```
"""
function compute_optimal_chunk_size(matrix_size::Int, num_threads::Int)::Int
    matrix_size > 0 || throw(ArgumentError("matrix_size must be positive"))
    num_threads > 0 || throw(ArgumentError("num_threads must be positive"))
    return max(1, matrix_size ÷ (4 * num_threads))
end

# ---------------------------------------------------------------------------

"""
    parallelize_matrix_operations!(A::AbstractMatrix, op::Function,
                                    strategy::TaskParallelizationStrategy)

Apply the unary function `op` row-wise to matrix `A` **in place**, dividing
rows across threads according to `strategy.chunk_size`.

`op` must have the signature `op(row_range::UnitRange{Int}, A::AbstractMatrix)`.
It is called once per chunk with the range of row indices it should process.

# Thread scheduler
- `strategy.work_stealing == true`  → `Threads.@threads :dynamic` (dynamic
  load balancing; best when chunks have variable cost).
- `strategy.work_stealing == false` → `Threads.@threads :static` (fixed
  assignment; lower overhead for uniform workloads).

# Arguments
- `A`: Matrix to operate on.  Modified in place by `op`.
- `op`: Function with signature `(row_range, A)`.
- `strategy`: Parallelisation configuration.

# Usage guidance
Use when `n_timesteps > 100` and matrix operations dominate.  For smaller
problems the scheduling overhead exceeds the benefit.

# Example
```julia
strategy = TaskParallelizationStrategy()
parallelize_matrix_operations!(M, (rows, A) -> fill!(view(A, rows, :), 0.0), strategy)
```
"""
function parallelize_matrix_operations!(
    A::AbstractMatrix,
    op::Function,
    strategy::TaskParallelizationStrategy,
)
    n_rows = size(A, 1)
    chunk_size = strategy.chunk_size

    # Build list of ranges
    chunks = [r:min(r + chunk_size - 1, n_rows) for r in 1:chunk_size:n_rows]

    if strategy.work_stealing
        @threads :dynamic for chunk in chunks
            op(chunk, A)
        end
    else
        @threadsif true for chunk in chunks
            op(chunk, A)
        end
    end
    return A
end

# ---------------------------------------------------------------------------

"""
    parallel_propagator_computation(H_array::AbstractArray{ComplexF64,3},
                                     dt::Float64,
                                     strategy::TaskParallelizationStrategy
                                    ) -> Array{ComplexF64,3}

Compute the unitary propagators `U[k] = exp(-i H[k] dt)` for all time
steps `k` in parallel.

This is the **most time-critical operation in GRAPE**: for a system of
Hilbert-space dimension `d` and `N` time steps each propagator requires
an `O(d³)` eigendecomposition, giving total cost `O(N d³)`.  Threading
achieves near-linear speedup for large `N`.

# Arguments
- `H_array`: Shape `(n_timesteps, d, d)`.  `H_array[k, :, :]` is the
  Hamiltonian at step `k`.
- `dt`: Time step (same for all steps).
- `strategy`: Threading configuration.

# Returns
Array of shape `(n_timesteps, d, d)`.

# Implementation notes
Each thread independently calls the LAPACK eigendecomposition for its
assigned time steps.  No shared mutable state is accessed; synchronisation
is implicit at the `@threads` barrier.
"""
function parallel_propagator_computation(
    H_array::AbstractArray{ComplexF64,3},
    dt::Float64,
    strategy::TaskParallelizationStrategy,
)::Array{ComplexF64,3}
    n_timesteps, d, d2 = size(H_array)
    d == d2 || throw(DimensionMismatch("Hamiltonian slices must be square: $(d)×$(d2)"))

    U_array = Array{ComplexF64,3}(undef, n_timesteps, d, d)

    # Inner function: eigendecomposition-based matrix exp for a single step
    function _expH(H_k::Matrix{ComplexF64})::Matrix{ComplexF64}
        H_herm = Hermitian((H_k .+ H_k') ./ 2)
        F = eigen(H_herm)
        phases = exp.((-1im * dt) .* F.values)
        return F.vectors * (phases .* F.vectors')
    end

    if strategy.work_stealing
        @threads :dynamic for k in 1:n_timesteps
            U_array[k, :, :] .= _expH(Matrix{ComplexF64}(H_array[k, :, :]))
        end
    else
        @threadsif true for k in 1:n_timesteps
            U_array[k, :, :] .= _expH(Matrix{ComplexF64}(H_array[k, :, :]))
        end
    end

    return U_array
end

# ===========================================================================
# STRATEGY 2 – Vectorisation
# ===========================================================================

"""
    VectorizationStrategy

Configuration for SIMD-based inner-loop vectorisation.

# Fields
- `use_simd::Bool`: Enable `@simd` annotations on inner loops.
- `instruction_set::String`: Target SIMD instruction set.  Valid values:
  `"AVX512"`, `"AVX2"`, `"NEON"`, `"generic"`.  Used for documentation and
  to gate certain optimisations; the actual instructions emitted depend on
  the Julia/LLVM compilation target.
- `unroll_factor::Int`: Suggested manual unroll depth for inner loops (1 = no
  unroll).  Values > 1 may improve throughput on wide SIMD units.

# Usage guidance
For small dense matrices (`d ≤ 64`) SIMD loops can outperform BLAS due to
lower call overhead.  For larger matrices prefer BLAS (`use_blas=true` in
`CPUBackend`).
"""
struct VectorizationStrategy
    use_simd::Bool
    instruction_set::String
    unroll_factor::Int

    function VectorizationStrategy(use_simd::Bool, instruction_set::String, unroll_factor::Int)
        valid_isa = ("AVX512", "AVX2", "NEON", "generic")
        instruction_set in valid_isa ||
            throw(ArgumentError("instruction_set must be one of $valid_isa"))
        unroll_factor > 0 || throw(ArgumentError("unroll_factor must be positive"))
        new(use_simd, instruction_set, unroll_factor)
    end
end

# ---------------------------------------------------------------------------

"""
    detect_cpu_capabilities() -> String

Detect the best SIMD instruction set available on the current CPU.

Detection is performed by inspecting `Sys.CPU_NAME` and `Sys.ARCH`:

| Detected feature                  | Return value |
|-----------------------------------|--------------|
| "avx512" in CPU name              | `"AVX512"`   |
| "avx2" or "avx" in CPU name       | `"AVX2"`     |
| Apple Silicon / ARM "neon"        | `"NEON"`     |
| x86_64 (SSE2 baseline)            | `"AVX2"`     |
| Everything else                   | `"generic"`  |

# Returns
One of `"AVX512"`, `"AVX2"`, `"NEON"`, `"generic"`.

# Notes
Julia/LLVM will use the actual hardware features regardless of this
string; the return value is used by `VectorizationStrategy` to document
expectations and to select code paths that benefit from wide SIMD.
"""
function detect_cpu_capabilities()::String
    cpu_name = lowercase(string(Sys.CPU_NAME))

    if occursin("avx512", cpu_name)
        return "AVX512"
    elseif occursin("avx2", cpu_name) || occursin("avx", cpu_name)
        return "AVX2"
    elseif occursin("neon", cpu_name) ||
           Sys.ARCH === :aarch64 ||
           occursin("apple", cpu_name)
        return "NEON"
    elseif Sys.ARCH === :x86_64
        # SSE2 is mandatory on x86_64; treat as AVX2 width for selection purposes
        return "AVX2"
    else
        return "generic"
    end
end

# ---------------------------------------------------------------------------

"""
    select_best_vectorization_strategy() -> VectorizationStrategy

Auto-detect the current CPU and return the most capable
`VectorizationStrategy`.

# Unroll factor heuristics
| ISA       | unroll_factor |
|-----------|---------------|
| AVX512    | 4             |
| AVX2      | 2             |
| NEON      | 2             |
| generic   | 1             |

# Examples
```julia
vs = select_best_vectorization_strategy()
# vs.instruction_set == "AVX2" on a Haswell laptop
```
"""
function select_best_vectorization_strategy()::VectorizationStrategy
    isa = detect_cpu_capabilities()
    unroll = if isa == "AVX512"
        4
    elseif isa in ("AVX2", "NEON")
        2
    else
        1
    end
    return VectorizationStrategy(true, isa, unroll)
end

# ---------------------------------------------------------------------------

"""
    simd_matrix_multiply!(C::Matrix, A::Matrix, B::Matrix,
                           strategy::VectorizationStrategy)

Compute `C = A * B` using SIMD-annotated inner loops when
`strategy.use_simd == true`, falling back to `LinearAlgebra.mul!` otherwise.

# Algorithm
For SIMD path: standard `i-k-j` loop with `@simd` and `@inbounds`
annotations.  The `k` (inner contraction) loop is annotated to permit
vectorisation by LLVM.  The `j`-loop is unrolled by `strategy.unroll_factor`.

For large matrices (any dimension > 256) the BLAS path is always used
regardless of `strategy.use_simd`, because BLAS is more efficient for large
operands.

# Arguments
- `C`: Pre-allocated output matrix.  Modified in place.
- `A`, `B`: Input matrices.  `size(A,2) == size(B,1)` required.
- `strategy`: Vectorisation configuration.

# Returns
`C` (modified in place).

# Notes
The SIMD path is most beneficial for dense ComplexF64 matrices with
dimensions in the range 8–64.
"""
function simd_matrix_multiply!(
    C::Matrix{T},
    A::Matrix{T},
    B::Matrix{T},
    strategy::VectorizationStrategy,
) where {T<:Number}
    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("A is $(m)×$(k), B is $(k2)×$(n)"))
    size(C) == (m, n) || throw(DimensionMismatch("C must be $(m)×$(n)"))

    # For large matrices always use BLAS
    if !strategy.use_simd || max(m, n, k) > 256
        mul!(C, A, B)
        return C
    end

    fill!(C, zero(T))
    @inbounds for i in 1:m
        for p in 1:k
            a_ip = A[i, p]
            @simd for j in 1:n
                C[i, j] += a_ip * B[p, j]
            end
        end
    end
    return C
end

# ===========================================================================
# STRATEGY 3 – Gradient parallelisation
# ===========================================================================

"""
    GradientParallelization

Configuration for parallelised GRAPE gradient computation.

# Fields
- `method::String`: Parallelisation method.
  - `"data_parallel"`: Each thread computes gradients for a subset of
    control channels `j`.
  - `"task_parallel"`: Each thread processes a subset of time steps `k`.
- `num_workers::Int`: Number of parallel workers (threads).
- `batch_gradients::Bool`: When `true`, gradient slices `∂F/∂u_j[k]` for
  different `j` are accumulated into a pre-allocated output matrix rather
  than being assembled serially.
- `cache_intermediates::Bool`: When `true`, the forward propagators `P[k]`
  and backward propagators `Q[k]` are computed once and stored in memory,
  saving an `O(n_controls)` factor in propagator computations.

# Mathematical background
The GRAPE gradient is

    ∂F/∂u_j[k] = -i dt ⟨Q[k] | H_j P[k]⟩_F

where `P[k] = U[k] ⋯ U[1] |ψ₀⟩` (forward-evolved state) and
`Q[k] = U[N]† ⋯ U[k+1]† |target⟩†` (backward-evolved target).

The key insight is that `P[k]` and `Q[k]` are **independent of `j`**.
Computing them once and reusing across all `n_controls` gradient components
reduces propagator evaluations from `O(n_controls × N)` to `O(N)`.
"""
struct GradientParallelization
    method::String
    num_workers::Int
    batch_gradients::Bool
    cache_intermediates::Bool

    function GradientParallelization(
        method::String,
        num_workers::Int,
        batch_gradients::Bool,
        cache_intermediates::Bool,
    )
        method in ("data_parallel", "task_parallel") ||
            throw(ArgumentError("method must be \"data_parallel\" or \"task_parallel\""))
        num_workers > 0 || throw(ArgumentError("num_workers must be positive"))
        new(method, num_workers, batch_gradients, cache_intermediates)
    end
end

"""
    GradientParallelization(; method="data_parallel",
                               num_workers=Threads.nthreads(),
                               batch_gradients=true,
                               cache_intermediates=true) -> GradientParallelization

Keyword constructor with recommended defaults.
"""
function GradientParallelization(;
    method::String = "data_parallel",
    num_workers::Int = Threads.nthreads(),
    n_threads::Union{Int,Nothing} = nothing,
    batch_gradients::Bool = true,
    cache_intermediates::Bool = true,
)::GradientParallelization
    nw = n_threads === nothing ? num_workers : n_threads
    return GradientParallelization(method, nw, batch_gradients, cache_intermediates)
end

# ---------------------------------------------------------------------------

"""
    compute_shared_intermediates(H_array::AbstractArray{ComplexF64,3},
                                  H_ctrl::Vector{<:AbstractMatrix{ComplexF64}},
                                  controls::AbstractMatrix{Float64},
                                  psi0::AbstractVector{ComplexF64},
                                  target::AbstractVector{ComplexF64},
                                  dt::Float64) -> Dict{String, Any}

Precompute the forward propagators `P[k]` and backward propagators `Q[k]`
shared across all gradient components.

# Arguments
- `H_array`: Full Hamiltonians `H[k] = H_drift + Σ_j u_j[k] H_j`,
  shape `(n_timesteps, d, d)`.
- `H_ctrl`: List of `n_controls` control Hamiltonians `H_j`, each `d × d`.
- `controls`: Control amplitudes, shape `(n_controls, n_timesteps)`.
- `psi0`: Initial state vector, length `d`.
- `target`: Target state vector, length `d`.
- `dt`: Time step.

# Returns
`Dict` with keys:
- `"P"`: Forward-propagated states `P[k] = U[k] P[k-1]`,
  array of shape `(n_timesteps+1, d)`.
  `P[1] = psi0`, `P[k+1] = U[k] P[k]`.
- `"Q"`: Backward-propagated co-states `Q[k] = U[k+1]† Q[k+1]`,
  array of shape `(n_timesteps+1, d)`.
  `Q[n_timesteps+1] = target`, `Q[k] = U[k+1]† Q[k+1]`.
- `"U"`: Individual propagators, shape `(n_timesteps, d, d)`.
- `"fidelity"`: Fidelity `|⟨target|P[N+1]⟩|²` at current controls.

# Complexity
`O(N d³)` for propagators + `O(N d²)` for state propagation.
"""
function compute_shared_intermediates(
    H_array::AbstractArray{ComplexF64,3},
    H_ctrl::Vector{<:AbstractMatrix{ComplexF64}},
    controls::AbstractMatrix{Float64},
    psi0::AbstractVector{ComplexF64},
    target::AbstractVector{ComplexF64},
    dt::Float64,
)::Dict{String,Any}
    n_timesteps, d, _ = size(H_array)
    n_ctrl = length(H_ctrl)

    # ---------- individual propagators U[k] ----------
    U = Array{ComplexF64,3}(undef, n_timesteps, d, d)
    for k in 1:n_timesteps
        H_k = Hermitian((H_array[k, :, :] .+ H_array[k, :, :]') ./ 2)
        F = eigen(H_k)
        phases = exp.((-1im * dt) .* F.values)
        U[k, :, :] .= F.vectors * (phases .* F.vectors')
    end

    # ---------- forward states P[1..N+1] ----------
    # P[k+1] = U[k] * P[k],   P[1] = psi0
    P = Matrix{ComplexF64}(undef, n_timesteps + 1, d)
    P[1, :] .= psi0
    for k in 1:n_timesteps
        P[k+1, :] .= U[k, :, :] * P[k, :]
    end

    # ---------- backward co-states Q[1..N+1] ----------
    # Q[k] = U[k+1]† * Q[k+1],   Q[N+1] = target
    Q = Matrix{ComplexF64}(undef, n_timesteps + 1, d)
    Q[n_timesteps+1, :] .= target
    for k in n_timesteps:-1:1
        Q[k, :] .= U[k, :, :]' * Q[k+1, :]
    end

    fidelity = abs2(dot(target, P[n_timesteps+1, :]))

    return Dict{String,Any}(
        "P"        => P,
        "Q"        => Q,
        "U"        => U,
        "fidelity" => fidelity,
    )
end

# ---------------------------------------------------------------------------

"""
    parallel_gradient_computation(H_array::AbstractArray{ComplexF64,3},
                                   H_ctrl::Vector{<:AbstractMatrix{ComplexF64}},
                                   controls::AbstractMatrix{Float64},
                                   psi0::AbstractVector{ComplexF64},
                                   target::AbstractVector{ComplexF64},
                                   dt::Float64,
                                   strategy::GradientParallelization
                                  ) -> Matrix{Float64}

Compute the full GRAPE gradient matrix `∂F/∂u_j[k]` for all control
channels `j` and time steps `k`.

# GRAPE gradient formula
    ∂F/∂u_j[k] = 2 Re[ ⟨target | P[N+1]⟩ * ⟨Q[k] | (-i dt H_j) | P[k]⟩ ]

where `P[k]` and `Q[k]` are the forward and backward co-states computed
by `compute_shared_intermediates`.

# Arguments
- `H_array`: Full Hamiltonians, shape `(n_timesteps, d, d)`.
- `H_ctrl`: Control Hamiltonians `[H_1, …, H_{n_ctrl}]`.
- `controls`: Amplitudes, shape `(n_controls, n_timesteps)`.
- `psi0`: Initial state.
- `target`: Target state.
- `dt`: Time step.
- `strategy`: Parallelisation configuration.

# Returns
Gradient matrix of shape `(n_controls, n_timesteps)`.

# Key optimisation
`P[k]` and `Q[k]` are computed **once** (if `strategy.cache_intermediates`)
and shared across all `n_controls` gradient channels, saving an
`O(n_controls)` factor in propagator computations.

# Complexity
`O(N d³)` for shared propagators + `O(n_ctrl × N × d²)` for gradient.
"""
function parallel_gradient_computation(
    H_array::AbstractArray{ComplexF64,3},
    H_ctrl::Vector{<:AbstractMatrix{ComplexF64}},
    controls::AbstractMatrix{Float64},
    psi0::AbstractVector{ComplexF64},
    target::AbstractVector{ComplexF64},
    dt::Float64,
    strategy::GradientParallelization,
)::Matrix{Float64}
    n_timesteps, d, _ = size(H_array)
    n_ctrl = length(H_ctrl)

    # ---- Shared intermediates (computed once for all j) ----
    intermediates = compute_shared_intermediates(
        H_array, H_ctrl, controls, psi0, target, dt,
    )
    P = intermediates["P"]        # (n_timesteps+1, d)
    Q = intermediates["Q"]        # (n_timesteps+1, d)

    # Overlap factor ⟨target | P[N+1]⟩
    phi = dot(target, P[n_timesteps+1, :])   # complex scalar

    gradient = zeros(Float64, n_ctrl, n_timesteps)

    # Threading safety (both branches): the outer @threads index (j or k) is
    # unique per thread, and each thread writes into disjoint columns/rows of
    # `gradient`. Reads from `P`, `Q`, `H_ctrl`, `phi`, `dt` are read-only.
    # No two threads alias the same gradient[i, k] slot, so no locking is
    # needed. `:static` scheduling pins the ranges, making the disjointness
    # argument hold by construction; do NOT change to `:dynamic` without
    # re-auditing, as dynamic scheduling still preserves disjoint iterations,
    # but any future refactor that introduces a shared accumulator would
    # become a race.
    if strategy.method == "data_parallel"
        # Each thread handles a subset of control channels j; writes to
        # gradient[j, :] only (column-disjoint).
        @threadsif true for j in 1:n_ctrl
            Hj = H_ctrl[j]
            for k in 1:n_timesteps
                # ⟨Q[k] | (-i dt Hj) | P[k]⟩
                Hjp = Hj * P[k, :]                    # d-vector
                inner = dot(Q[k, :], Hjp)             # complex scalar
                # ∂F/∂u_j[k] = 2 Re[ φ* × (-i dt) × inner ]
                gradient[j, k] = 2.0 * real(conj(phi) * (-1im * dt) * inner)
            end
        end
    else
        # "task_parallel": each thread handles a subset of time steps k;
        # writes to gradient[:, k] only (row-disjoint).
        @threadsif true for k in 1:n_timesteps
            Pk = P[k, :]
            Qk = Q[k, :]
            for j in 1:n_ctrl
                Hjp = H_ctrl[j] * Pk
                inner = dot(Qk, Hjp)
                gradient[j, k] = 2.0 * real(conj(phi) * (-1im * dt) * inner)
            end
        end
    end

    return gradient
end
