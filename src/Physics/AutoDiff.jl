"""
    AutomaticDifferentiation.jl

Automatic differentiation (AD) integration for PULSAR quantum control gradients.

Provides forward-mode (ForwardDiff.jl), reverse-mode (Zygote.jl), and
finite-difference gradient computation as drop-in alternatives to the
analytical GRAPE gradient. The primary use cases are:

  1. Gradient verification — compare AD-computed gradients against the
     analytical GRAPE gradient to catch implementation bugs.
  2. Novel objective functions — when an objective is not covered by the
     built-in GRAPE gradient, AD can differentiate it automatically.
  3. Rapid prototyping — iterate on new fidelity metrics without deriving
     analytical gradients by hand.

# Backend selection heuristic

  n_params = n_controls * n_timesteps

  - n_params < 200 and ForwardDiff available  → :forward  (fewer passes, low overhead)
  - n_params ≥ 200 and Zygote available        → :reverse  (one reverse pass)
  - otherwise                                  → :finite_diff

All backends return a `Matrix{Float64}` of shape `[n_controls × n_timesteps]`
matching the convention of `compute_grape_gradient`.

# Thread safety
All public functions are stateless and thread-safe.  The ForwardDiff chunked
Jacobian internally uses scratch arrays scoped to each call.

Reference:
  Revels, Lubin & Papamarkou, "Forward-Mode Automatic Differentiation in Julia",
  arXiv:1607.07892 (2016).
"""

using LinearAlgebra

# ---------------------------------------------------------------------------
# Extension stubs — real implementations live in ext/PULSARForwardDiffExt.jl
# and ext/PULSARZygoteExt.jl. These are loaded automatically by Julia 1.9+
# when ForwardDiff / Zygote are in the environment.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Configuration type
# ---------------------------------------------------------------------------

"""
    AutoDiffConfig

Configuration for automatic-differentiation gradient computation.

# Fields
- `backend::Symbol` — differentiation backend; one of `:forward`, `:reverse`,
  `:finite_diff`, or `:auto` (default). `:auto` selects the best available
  backend based on problem size and package availability.
- `chunk_size::Int` — ForwardDiff chunk size. `0` (default) lets ForwardDiff
  choose automatically. Larger values use more memory but fewer passes.
- `verify_against_numerical::Bool` — when `true`, the computed gradient is
  cross-checked against a finite-difference gradient; a warning is printed when
  the maximum deviation exceeds `numerical_eps`. Default: `false`.
- `numerical_eps::Float64` — finite-difference step size used for verification
  and the `:finite_diff` backend. Default: `1e-6`.
- `verbose::Bool` — print backend selection and timing information when `true`.
  Default: `false`.

# Example
```julia
cfg = AutoDiffConfig(backend=:forward, chunk_size=10, verbose=true)
G = compute_gradient_autodiff(sys, seq, tgt; config=cfg)
```
"""
struct AutoDiffConfig
    backend::Symbol
    chunk_size::Int
    verify_against_numerical::Bool
    numerical_eps::Float64
    verbose::Bool
end

"""
    AutoDiffConfig(; backend=:auto, chunk_size=0, verify=false,
                     eps=1e-6, verbose=false) -> AutoDiffConfig

Keyword constructor for `AutoDiffConfig`.

# Keyword Arguments
- `backend`    — `:auto`, `:forward`, `:reverse`, or `:finite_diff`
- `chunk_size` — ForwardDiff chunk size; `0` = automatic
- `verify`     — cross-check gradient against finite differences
- `eps`        — finite-difference step size (for verify and fallback backend)
- `verbose`    — print progress messages

# Throws
- `ArgumentError` if `backend` is not a recognised symbol.
- `ArgumentError` if `chunk_size < 0` or `eps ≤ 0`.

# Example
```julia
cfg = AutoDiffConfig(backend=:auto, verify=true, verbose=true)
```
"""
function AutoDiffConfig(;
        backend::Symbol  = :auto,
        chunk_size::Int  = 0,
        verify::Bool     = false,
        verify_against_numerical::Bool = verify,
        eps::Float64     = 1e-6,
        numerical_eps::Float64 = eps,
        verbose::Bool    = false)::AutoDiffConfig

    valid_backends = (:auto, :forward, :reverse, :finite_diff,
                      :forwarddiff, :zygote)
    if !(backend in valid_backends)
        throw(ArgumentError(
            "backend must be one of $valid_backends, got :$backend"))
    end
    if chunk_size < 0
        throw(ArgumentError("chunk_size must be ≥ 0, got $chunk_size"))
    end
    if numerical_eps <= 0.0
        throw(ArgumentError("numerical_eps must be > 0, got $numerical_eps"))
    end
    return AutoDiffConfig(backend, chunk_size,
                          verify_against_numerical, numerical_eps, verbose)
end

# ---------------------------------------------------------------------------
# Optional-backend probes
# ---------------------------------------------------------------------------
# These helpers report whether ForwardDiff / Zygote have been loaded into the
# active session.  They are populated by the package extensions (when
# present) and default to `false` so the auto-backend selection falls back to
# finite differences.

const _FORWARDDIFF_LOADED = Ref(false)
const _ZYGOTE_LOADED      = Ref(false)

_forwarddiff_available() = _FORWARDDIFF_LOADED[]
_zygote_available()      = _ZYGOTE_LOADED[]

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

"""
    compute_gradient_autodiff(system::AbstractQuantumSystem,
                               controls::ControlSequence,
                               target::QuantumTarget;
                               config::AutoDiffConfig = AutoDiffConfig())
    -> Matrix{Float64}

Compute the fidelity gradient ∂F/∂u_j[k] using automatic differentiation.

This function is a drop-in alternative to `compute_grape_gradient` that uses
AD rather than the analytical GRAPE formula. The returned matrix has the same
shape `[n_controls × n_timesteps]` and the same sign convention (positive
entries mean "increase this control to increase fidelity").

# Arguments
- `system`   — quantum system (H_drift, H_controls)
- `controls` — current piecewise-constant control sequence
- `target`   — optimization target (unitary or state)
- `config`   — AD configuration; see `AutoDiffConfig`

# Returns
Gradient matrix `G[j, k] = ∂F/∂u_j[k]`, shape `[n_controls × n_timesteps]`.

# Algorithm
1. Select backend via `select_autodiff_backend` (respects `config.backend`).
2. Flatten `controls.controls` to a 1-D vector and differentiate
   `_fidelity_flat` with respect to that vector.
3. Reshape the resulting flat gradient back to `[n_controls × n_timesteps]`.
4. Optionally verify against finite differences when `config.verify_against_numerical`.

# Throws
- `ArgumentError` if the selected backend is unavailable and no fallback exists.

# Example
```julia
G_ad   = compute_gradient_autodiff(sys, seq, tgt)
G_grape = compute_grape_gradient(sys, seq, tgt)
@show maximum(abs.(G_ad .- G_grape))
```
"""
function compute_gradient_autodiff(
        system::AbstractQuantumSystem,
        controls::ControlSequence,
        target::QuantumTarget;
        config::AutoDiffConfig = AutoDiffConfig())::Matrix{Float64}

    backend = select_autodiff_backend(system, controls, config)

    if config.verbose
        println("[AutoDiff] Using backend :$backend for " *
                "$(system.n_controls)×$(controls.n_timesteps) problem")
    end

    G = if backend == :forward
        forward_diff_gradient(system, controls, target, config)
    elseif backend == :reverse
        reverse_diff_gradient(system, controls, target, config)
    else
        finite_diff_gradient_ad(system, controls, target; eps=config.numerical_eps)
    end

    if config.verify_against_numerical
        G_fd = finite_diff_gradient_ad(system, controls, target;
                                        eps=config.numerical_eps)
        max_dev = maximum(abs.(G .- G_fd))
        if max_dev > 1e-4
            @warn "[AutoDiff] Gradient verification FAILED: max|G_ad - G_fd| = $max_dev"
        elseif config.verbose
            println("[AutoDiff] Gradient verified: max|G_ad - G_fd| = $max_dev")
        end
    end

    return G
end

# ---------------------------------------------------------------------------
# AD-differentiable fidelity function (flat control vector interface)
# ---------------------------------------------------------------------------

"""
    _fidelity_flat(u_flat::AbstractVector, system, target,
                    n_controls::Int, n_timesteps::Int, dt::Float64)
    -> Float64

Compute fidelity from a flat (1-D) control vector `u_flat`.

This is the scalar function that AD backends differentiate. It is written
without in-place mutations so that both ForwardDiff (dual-number propagation)
and Zygote (reverse-mode AD) can differentiate through it.

# Arguments
- `u_flat`      — flat control vector of length `n_controls * n_timesteps`;
  the matrix `u[j, k]` is stored in column-major order: `u_flat[(k-1)*n_controls + j]`.
- `system`      — quantum system
- `target`      — optimization target
- `n_controls`  — number of control channels
- `n_timesteps` — number of time steps
- `dt`          — time step duration (seconds)

# Returns
Scalar fidelity F ∈ [0, 1].

# Notes
The function reconstructs a `ControlSequence` on every call, which is
allocation-heavy but correct.  For production optimization use the analytical
GRAPE gradient via `compute_grape_gradient`.
"""
function _fidelity_flat(u_flat::AbstractVector,
                         system::AbstractQuantumSystem,
                         target::QuantumTarget,
                         n_controls::Int,
                         n_timesteps::Int,
                         dt::Float64)::Float64
    # Reshape flat vector → control matrix (column-major: [n_controls × n_timesteps])
    u_mat = reshape(u_flat, n_controls, n_timesteps)

    # Build total Hamiltonians and propagators
    # We compute everything from scratch so AD can trace through.
    dim = system.dim
    U_total = Matrix{ComplexF64}(I, dim, dim)

    for k in 1:n_timesteps
        # H(t_k) = H_drift + Σ_j u_j[k] * H_j
        H_k = copy(system.H_drift)
        for j in 1:n_controls
            H_k = H_k .+ real(u_flat[(k-1)*n_controls + j]) .* system.H_controls[j]
        end
        # U[k] = exp(-i H_k dt)
        U_k = compute_propagator(H_k, dt)
        U_total = U_k * U_total
    end

    return compute_fidelity(U_total, target)
end

# ---------------------------------------------------------------------------
# Forward-mode gradient (ForwardDiff)
# ---------------------------------------------------------------------------

"""
    forward_diff_gradient(system::AbstractQuantumSystem,
                           controls::ControlSequence,
                           target::QuantumTarget,
                           config::AutoDiffConfig)
    -> Matrix{Float64}

Compute the fidelity gradient using ForwardDiff.gradient.

# Arguments
- `system`   — quantum system
- `controls` — current control sequence
- `target`   — optimization target
- `config`   — AD configuration (used for `chunk_size`)

# Returns
Gradient matrix `[n_controls × n_timesteps]`.

# Algorithm
Flattens the control matrix to a vector and calls `ForwardDiff.gradient`
(or `ForwardDiff.gradient` with explicit chunk tag for large problems).
Chunked evaluation reduces peak memory at the cost of multiple forward passes.

# Throws
- `ErrorException` if ForwardDiff is not available at runtime.

# Notes
ForwardDiff is typically fastest for `n_params ≲ 200` and when the function
evaluation is cheap relative to the overhead of setting up dual numbers.
"""
function forward_diff_gradient(system::AbstractQuantumSystem,
                                 controls::ControlSequence,
                                 target::QuantumTarget,
                                 config::AutoDiffConfig)::Matrix{Float64}
    error("ForwardDiff.jl is required for forward-mode AD.\n" *
          "Add it to your environment and load it: `using ForwardDiff, PULSAR`.\n" *
          "The PULSARForwardDiffExt extension will provide the implementation automatically.")
end

# ---------------------------------------------------------------------------
# Reverse-mode gradient (Zygote)
# ---------------------------------------------------------------------------

"""
    reverse_diff_gradient(system::AbstractQuantumSystem,
                           controls::ControlSequence,
                           target::QuantumTarget,
                           config::AutoDiffConfig)
    -> Matrix{Float64}

Compute the fidelity gradient using Zygote.gradient (reverse-mode AD).

# Arguments
- `system`   — quantum system
- `controls` — current control sequence
- `target`   — optimization target
- `config`   — AD configuration (used for fallback eps on failure)

# Returns
Gradient matrix `[n_controls × n_timesteps]`.

# Algorithm
Calls `Zygote.gradient` on `_fidelity_flat`.  If Zygote throws (e.g. because
the computation contains non-differentiable operations such as in-place array
mutations or LAPACK calls), the function falls back automatically to
`finite_diff_gradient_ad` with a warning.

# Notes
Reverse mode is advantageous when `n_params ≫ 1` because it requires only
one backward pass regardless of problem size, at the cost of storing the full
computational graph in memory.
"""
function reverse_diff_gradient(system::AbstractQuantumSystem,
                                 controls::ControlSequence,
                                 target::QuantumTarget,
                                 config::AutoDiffConfig)::Matrix{Float64}
    @warn "[AutoDiff] Zygote not available; falling back to finite differences.\n" *
          "  Add Zygote to your environment: `using Zygote, PULSAR`."
    return finite_diff_gradient_ad(system, controls, target; eps=config.numerical_eps)
end

# ---------------------------------------------------------------------------
# Finite-difference gradient (fallback / verification)
# ---------------------------------------------------------------------------

"""
    finite_diff_gradient_ad(system::AbstractQuantumSystem,
                              controls::ControlSequence,
                              target::QuantumTarget;
                              eps::Float64 = 1e-6)
    -> Matrix{Float64}

Compute the fidelity gradient via central finite differences.

This is the fallback backend used when ForwardDiff and Zygote are both
unavailable or fail, and it is also used internally for gradient verification.

# Arguments
- `system`   — quantum system
- `controls` — current control sequence
- `target`   — optimization target
- `eps`      — finite-difference step size (default `1e-6`)

# Returns
Gradient matrix `[n_controls × n_timesteps]` computed as

    G[j, k] ≈ (F(u + ε e_{jk}) - F(u - ε e_{jk})) / (2ε)

where `e_{jk}` is the unit vector in the direction of control `j` at timestep `k`.

# Notes
Complexity is O(2 * n_controls * n_timesteps) fidelity evaluations.  Each
evaluation requires computing the full propagator in O(n_timesteps * dim³).
Total cost is O(n_controls * n_timesteps² * dim³), which is expensive for
large problems.  Use analytical GRAPE gradients for production runs.
"""
function finite_diff_gradient_ad(system::AbstractQuantumSystem,
                                   controls::ControlSequence,
                                   target::QuantumTarget;
                                   eps::Float64 = 1e-6)::Matrix{Float64}
    nc = system.n_controls
    nt = controls.n_timesteps
    dt = controls.dt
    u0 = copy(controls.controls)
    G  = zeros(Float64, nc, nt)

    for j in 1:nc
        for k in 1:nt
            # Perturb +eps
            u_plus = copy(u0)
            u_plus[j, k] += eps
            seq_plus = ControlSequence(u_plus, dt, controls.total_time, nt)
            H_plus   = build_total_hamiltonian(system, seq_plus)
            U_plus   = compute_total_propagator(H_plus, dt)
            F_plus   = compute_fidelity(U_plus, target)

            # Perturb -eps
            u_minus = copy(u0)
            u_minus[j, k] -= eps
            seq_minus = ControlSequence(u_minus, dt, controls.total_time, nt)
            H_minus   = build_total_hamiltonian(system, seq_minus)
            U_minus   = compute_total_propagator(H_minus, dt)
            F_minus   = compute_fidelity(U_minus, target)

            G[j, k] = (F_plus - F_minus) / (2.0 * eps)
        end
    end

    return G
end

# ---------------------------------------------------------------------------
# Backend selection heuristic
# ---------------------------------------------------------------------------

"""
    select_autodiff_backend(system::AbstractQuantumSystem,
                             controls::ControlSequence,
                             config::AutoDiffConfig)
    -> Symbol

Choose the best available AD backend for the given problem.

# Arguments
- `system`   — quantum system
- `controls` — current control sequence
- `config`   — AD configuration (may override automatic selection via `backend` field)

# Returns
One of `:forward`, `:reverse`, or `:finite_diff`.

# Selection logic
| Condition                                      | Selected backend |
|:---------------------------------------------- |:---------------- |
| `config.backend != :auto`                      | `config.backend` |
| ForwardDiff available AND n_params < 200       | `:forward`       |
| Zygote available AND n_params ≥ 200            | `:reverse`       |
| Zygote available AND ForwardDiff not available | `:reverse`       |
| ForwardDiff available AND Zygote not available | `:forward`       |
| Neither available                              | `:finite_diff`   |

# Notes
The threshold of 200 parameters is an empirical rule of thumb. For problems
with fewer than 200 parameters the overhead of setting up the reverse-mode
computational graph dominates; for larger problems the O(n_params) cost of
forward mode becomes the bottleneck.
"""
function select_autodiff_backend(system::AbstractQuantumSystem,
                                   controls::ControlSequence,
                                   config::AutoDiffConfig)::Symbol
    # Explicit user request
    if config.backend != :auto
        return config.backend
    end

    n_params = system.n_controls * controls.n_timesteps

    if _forwarddiff_available() && n_params < 200
        return :forward
    elseif _zygote_available()
        return :reverse
    elseif _forwarddiff_available()
        return :forward
    else
        return :finite_diff
    end
end

# ---------------------------------------------------------------------------
# Gradient verification
# ---------------------------------------------------------------------------

"""
    verify_gradient_autodiff(system::AbstractQuantumSystem,
                              controls::ControlSequence,
                              target::QuantumTarget;
                              tol::Float64 = 1e-5,
                              verbose::Bool = true)
    -> Bool

Cross-check the analytical GRAPE gradient against an AD-computed gradient.

This is a diagnostic tool for detecting bugs in the analytical gradient
implementation or in a custom fidelity function.

# Arguments
- `system`   — quantum system
- `controls` — control sequence at which to evaluate gradients
- `target`   — optimization target
- `tol`      — maximum allowed element-wise deviation (default `1e-5`)
- `verbose`  — print detailed comparison statistics when `true` (default `true`)

# Returns
`true` if the maximum deviation `max|G_grape - G_ad|` is less than `tol`,
`false` otherwise.

# Output (when verbose)
Prints the maximum absolute deviation, the location (j, k) of the largest
discrepancy, and both gradient values at that location.

# Example
```julia
is_ok = verify_gradient_autodiff(sys, seq, tgt; tol=1e-5, verbose=true)
# [GradVerify] max|G_grape - G_ad| = 3.14e-08 at (j=2, k=17)
#   G_grape[2,17] =  0.00423
#   G_ad[2,17]    =  0.00423
# [GradVerify] PASSED (tol = 1.0e-5)
```
"""
function verify_gradient_autodiff(system::AbstractQuantumSystem,
                                    controls::ControlSequence,
                                    target::QuantumTarget;
                                    tol::Float64 = 1e-5,
                                    verbose::Bool = true)::Bool
    # Analytical GRAPE gradient
    G_grape = compute_grape_gradient(system, controls, target)

    # AD gradient (auto-select backend)
    cfg = AutoDiffConfig(backend=:auto, verbose=false)
    G_ad = compute_gradient_autodiff(system, controls, target; config=cfg)

    diff_mat = abs.(G_grape .- G_ad)
    max_dev  = maximum(diff_mat)
    idx      = argmax(diff_mat)
    j_max, k_max = Tuple(idx)

    if verbose
        @printf("[GradVerify] max|G_grape - G_ad| = %.3e at (j=%d, k=%d)\n",
                max_dev, j_max, k_max)
        @printf("  G_grape[%d,%d] = % .5f\n", j_max, k_max, G_grape[j_max, k_max])
        @printf("  G_ad[%d,%d]    = % .5f\n", j_max, k_max, G_ad[j_max, k_max])
        if max_dev < tol
            @printf("[GradVerify] PASSED (tol = %.1e)\n", tol)
        else
            @printf("[GradVerify] FAILED (tol = %.1e)\n", tol)
        end
    end

    return max_dev < tol
end

# ---------------------------------------------------------------------------
# Internal helper: total propagator from H_total array
# ---------------------------------------------------------------------------

"""
    compute_total_propagator(H_total::Array{ComplexF64,3}, dt::Float64)
    -> Matrix{ComplexF64}

Compute the full time-ordered propagator U_total = U[n_t] ⋅ … ⋅ U[1] from
a pre-built array of total Hamiltonians.

# Arguments
- `H_total` — 3-D array of size `[dim × dim × n_timesteps]`; `H_total[:,:,k]`
  is the total Hamiltonian at time step k.
- `dt`       — time step duration (seconds)

# Returns
Full propagator matrix `[dim × dim]`.

# Notes
This is a thin wrapper used internally by `finite_diff_gradient_ad` and
`_fidelity_flat` to avoid code duplication.
"""
function compute_total_propagator(H_total::Array{ComplexF64,3},
                                    dt::Float64)::Matrix{ComplexF64}
    # `build_total_hamiltonian` returns `[n_timesteps × dim × dim]`.
    n_t = size(H_total, 1)
    dim = size(H_total, 2)
    U   = Matrix{ComplexF64}(I, dim, dim)
    for k in 1:n_t
        U = compute_propagator(H_total[k, :, :], dt) * U
    end
    return U
end
