"""
    AdaptiveStepSize.jl

Adaptive step-size schedules and preconditioning for PULSAR optimizers.

Provides:

- **Adam** (Kingma & Ba, 2015) — adaptive moment estimation optimizer with
  bias correction and per-parameter learning rates.  Effective for noisy
  fidelity landscapes.

- **Backtracking line search** — Armijo sufficient-decrease backtracking along
  an arbitrary search direction in flattened control space.

- **Strong Wolfe line search** — zoom algorithm that satisfies both the Armijo
  sufficient-decrease and the strong curvature conditions.  Required by
  BFGS/L-BFGS for proper Hessian updates.

- **Diagonal preconditioning** — scales the gradient by the reciprocal of a
  diagonal Hessian approximation, accelerating convergence in ill-conditioned
  problems.

- **Hessian diagonal estimation** — finite-difference estimate of the diagonal
  Hessian elements used by the diagonal preconditioner.

References:
  Kingma & Ba, "Adam: A method for stochastic optimization", ICLR 2015.
  Nocedal & Wright, "Numerical Optimization", 2nd ed., Springer (2006), ch. 3.
"""

using LinearAlgebra
using Printf

# ============================================================================
# Adam state and configuration
# ============================================================================

"""
    AdamState

Mutable state for the Adam optimizer, updated at each iteration.

# Fields
- `m::Vector{Float64}` — first moment estimate (exponential moving average of gradient)
- `v::Vector{Float64}` — second moment estimate (exponential moving average of g²)
- `t::Int` — time step counter (starts at 0, incremented before each update)

Initialize via `adam_init(n_params)`.
"""
mutable struct AdamState
    m::Vector{Float64}   # First moment
    v::Vector{Float64}   # Second moment (element-wise squared gradient EMA)
    t::Int               # Time step
end

"""
    AdamConfig

Hyperparameters for the Adam optimizer.

# Fields
- `learning_rate::Float64` — step size η (default 0.01)
- `beta1::Float64` — first moment decay rate β₁ (default 0.9)
- `beta2::Float64` — second moment decay rate β₂ (default 0.999)
- `epsilon::Float64` — numerical stability term ε (default 1e-8)

# Example
```julia
cfg = AdamConfig(learning_rate=0.001, beta1=0.9, beta2=0.999)
```
"""
struct AdamConfig
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
end

"""
    AdamConfig(; kwargs...) -> AdamConfig

Construct an `AdamConfig` with keyword arguments and default values.

# Keyword Arguments
- `learning_rate = 0.01`
- `beta1 = 0.9`
- `beta2 = 0.999`
- `epsilon = 1e-8`
"""
function AdamConfig(;
    learning_rate::Float64 = 0.01,
    beta1::Float64         = 0.9,
    beta2::Float64         = 0.999,
    epsilon::Float64       = 1e-8
)::AdamConfig
    learning_rate > 0  || throw(ArgumentError("learning_rate must be positive"))
    0 <= beta1 < 1     || throw(ArgumentError("beta1 must be in [0, 1)"))
    0 <= beta2 < 1     || throw(ArgumentError("beta2 must be in [0, 1)"))
    epsilon > 0        || throw(ArgumentError("epsilon must be positive"))

    return AdamConfig(learning_rate, beta1, beta2, epsilon)
end

# ============================================================================
# Adam: initialization and update
# ============================================================================

"""
    adam_init(n_params::Int) -> AdamState

Initialize the Adam optimizer state for `n_params` parameters.

Creates zeroed first-moment `m`, second-moment `v` vectors and sets the time
step counter `t = 0`.

# Arguments
- `n_params` — total number of scalar parameters (n_controls * n_timesteps)

# Returns
A fresh `AdamState` ready for the first update call.

# Example
```julia
state = adam_init(n_controls * n_timesteps)
```
"""
function adam_init(n_params::Int)::AdamState
    n_params >= 1 || throw(ArgumentError("n_params must be ≥ 1"))
    return AdamState(zeros(Float64, n_params), zeros(Float64, n_params), 0)
end

"""
    adam_update!(controls::Matrix{Float64},
                  gradient::Matrix{Float64},
                  state::AdamState,
                  config::AdamConfig) -> Matrix{Float64}

Apply one Adam update step to the control matrix (gradient *ascent* on fidelity).

The update rules are:

    t   ← t + 1
    m   ← β₁ m + (1 - β₁) g
    v   ← β₂ v + (1 - β₂) g²
    m̂   = m / (1 - β₁ᵗ)         (bias-corrected first moment)
    v̂   = v / (1 - β₂ᵗ)         (bias-corrected second moment)
    u   ← u + η * m̂ / (√v̂ + ε)  (ascent step)

# Arguments
- `controls` — current control amplitudes `[n_controls × n_timesteps]`, updated in-place
- `gradient` — fidelity gradient ∂F/∂u_j[k], same shape as `controls`
- `state`    — mutable `AdamState` (modified in-place)
- `config`   — `AdamConfig` with hyperparameters

# Returns
Updated `controls` matrix (same object, modified in-place).

# Notes
The gradient should be the *ascent* direction (i.e. ∂F/∂u, positive = increase
fidelity). Adam divides by the RMS of recent gradients, giving larger steps in
directions of small curvature and smaller steps where the gradient has been
historically large (adaptive learning rate).

# Example
```julia
state  = adam_init(n_c * n_t)
config = AdamConfig(learning_rate=0.005)
for iter in 1:max_iter
    g = compute_grape_gradient(sys, seq, target)
    adam_update!(controls, g, state, config)
end
```
"""
function adam_update!(controls::Matrix{Float64},
                       gradient::Matrix{Float64},
                       state::AdamState,
                       config::AdamConfig)::Matrix{Float64}
    if size(controls) != size(gradient)
        throw(DimensionMismatch(
            "controls size $(size(controls)) ≠ gradient size $(size(gradient))"))
    end
    n = length(controls)
    if length(state.m) != n
        throw(DimensionMismatch(
            "AdamState has $(length(state.m)) params but controls has $n elements"))
    end

    # Increment time step
    state.t += 1
    t = state.t

    β1 = config.beta1
    β2 = config.beta2
    η  = config.learning_rate
    ε  = config.epsilon

    # Bias-correction factors
    bc1 = 1.0 - β1^t
    bc2 = 1.0 - β2^t

    # Flat view to avoid repeated reshape
    g_flat = vec(gradient)
    u_flat = vec(controls)

    @inbounds for i in 1:n
        gi        = g_flat[i]
        state.m[i] = β1 * state.m[i] + (1 - β1) * gi
        state.v[i] = β2 * state.v[i] + (1 - β2) * gi * gi
        m_hat      = state.m[i] / bc1
        v_hat      = state.v[i] / bc2
        u_flat[i]  = u_flat[i] + η * m_hat / (sqrt(v_hat) + ε)
    end

    # u_flat is a view into controls, so it is already updated.
    # Return for convenience chaining.
    return controls
end

"""
    adam_optimize(system::AbstractQuantumSystem,
                  target::QuantumTarget,
                  controls_init::ControlSequence;
                  config::AdamConfig = AdamConfig(),
                  max_iter::Int = 1000,
                  convergence_tol::Float64 = 1e-8,
                  verbose::Bool = false,
                  print_interval::Int = 100) -> OptimizationResult

Run the Adam optimizer on a quantum control problem.

Internally calls `adam_update!` at each iteration and records fidelity and
gradient norm history.  Returns the standard `OptimizationResult`.

# Arguments
- `system`           — quantum system
- `target`           — optimization target
- `controls_init`    — initial control sequence
- `config`           — Adam hyperparameters (default `AdamConfig()`)
- `max_iter`         — maximum iterations (default 1000)
- `convergence_tol`  — |ΔF| convergence threshold (default 1e-8)
- `verbose`          — print progress (default false)
- `print_interval`   — logging frequency (default 100)

# Returns
`OptimizationResult`.

# Example
```julia
result = adam_optimize(sys, target, seq;
             config = AdamConfig(learning_rate=0.005),
             max_iter = 2000,
             verbose  = true)
```
"""
function adam_optimize(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls_init::ControlSequence;
    config::AdamConfig          = AdamConfig(),
    max_iter::Int               = 1000,
    convergence_tol::Float64    = 1e-8,
    verbose::Bool               = false,
    print_interval::Int         = 100
)::OptimizationResult

    t_start = time()
    dt      = controls_init.dt
    n_c     = system.n_controls
    n_t     = controls_init.n_timesteps
    n       = n_c * n_t

    controls      = copy(controls_init.controls)
    best_controls = copy(controls)

    state = adam_init(n)

    fid_hist   = Float64[]
    gnorm_hist = Float64[]
    n_fid      = 0
    n_grad     = 0
    F_best     = -Inf
    converged  = false
    reason     = "maximum iterations reached"

    for iter in 1:max_iter
        seq_iter = ControlSequence(controls, dt, dt * n_t, n_t)

        # Fidelity
        H_total = build_total_hamiltonian(system, seq_iter)
        U_steps = compute_propagators(H_total, dt)
        U_total = compute_total_propagator(U_steps)
        F_curr  = compute_fidelity(U_total, target)
        n_fid  += 1

        # GRAPE gradient
        G = compute_grape_gradient(system, seq_iter, target)
        n_grad += 1
        gnorm  = norm(G)

        push!(fid_hist,   F_curr)
        push!(gnorm_hist, gnorm)

        if F_curr > F_best
            F_best = F_curr
            best_controls .= controls
        end

        if verbose && (iter % print_interval == 0 || iter == 1)
            @printf("[Adam] iter=%5d  F=%.8f  |∇F|=%.3e  η=%.3e\n",
                    iter, F_curr, gnorm, config.learning_rate)
        end

        # Convergence
        n_h = length(fid_hist)
        if n_h >= 2 && abs(fid_hist[n_h] - fid_hist[n_h-1]) < convergence_tol
            converged = true
            reason    = "fidelity change < $(convergence_tol)"
            break
        end
        if F_curr >= 1.0 - 1e-12
            converged = true
            reason    = "perfect fidelity"
            break
        end

        # Adam update (in-place, ascent)
        adam_update!(controls, G, state, config)
    end

    return OptimizationResult(
        best_controls,
        F_best,
        fid_hist,
        gnorm_hist,
        length(fid_hist),
        converged,
        reason,
        time() - t_start,
        n_fid,
        n_grad,
        Dict{String,Any}(
            "algorithm"     => "Adam",
            "learning_rate" => config.learning_rate,
            "beta1"         => config.beta1,
            "beta2"         => config.beta2,
        )
    )
end

# ============================================================================
# Backtracking line search
# ============================================================================

"""
    backtracking_line_search(f::Function,
                              u::Vector{Float64},
                              direction::Vector{Float64},
                              grad::Vector{Float64};
                              c::Float64     = 0.1,
                              rho::Float64   = 0.5,
                              alpha0::Float64 = 1.0,
                              max_iter::Int  = 50) -> Float64

Armijo backtracking line search for a scalar objective function (fidelity
maximization convention).

Finds the largest `α ∈ {alpha0, alpha0·ρ, …}` satisfying the Armijo condition:

    f(u + α·d) ≥ f(u) + c·α·⟨grad, direction⟩

# Arguments
- `f`         — objective function `Vector{Float64} → Float64` (fidelity to maximize)
- `u`         — current iterate (flattened controls)
- `direction` — search direction (should satisfy ⟨grad, direction⟩ > 0 for ascent)
- `grad`      — gradient of `f` at `u` (used for slope computation)
- `c`         — Armijo constant ∈ (0, 1), default 1e-4 (standard in Nocedal & Wright)
- `rho`       — backtracking factor ∈ (0, 1), default 0.5
- `alpha0`    — initial step size, default 1.0
- `max_iter`  — maximum backtracking steps, default 50

# Returns
Accepted step size α.  If no step satisfies Armijo within `max_iter`
reductions, returns the smallest tried step size (the search degrades to a
tiny step rather than failing).

# Example
```julia
f = u -> fidelity(sys, target, reshape(u, n_c, n_t), dt)
α = backtracking_line_search(f, u, direction, grad; c=1e-4, rho=0.5)
u = u .+ α .* direction
```
"""
function backtracking_line_search(f::Function,
                                   u::Vector{Float64},
                                   direction::Vector{Float64},
                                   grad::Vector{Float64};
                                   c::Float64       = 1e-4,
                                   rho::Float64     = 0.5,
                                   alpha0::Float64  = 1.0,
                                   max_iter::Int    = 50,
                                   check_invariants::Bool = false)::Float64
    length(u) == length(direction) == length(grad) ||
        throw(DimensionMismatch("u, direction, grad must all have the same length"))
    c > 0 && c < 1   || throw(ArgumentError("c must be in (0, 1)"))
    rho > 0 && rho < 1 || throw(ArgumentError("rho must be in (0, 1)"))
    alpha0 > 0         || throw(ArgumentError("alpha0 must be positive"))

    f0    = f(u)
    slope = dot(grad, direction)

    α = alpha0
    for _ in 1:max_iter
        f_new = f(u .+ α .* direction)
        if f_new >= f0 + c * α * slope
            if check_invariants
                # Map ascent Armijo (f_new ≥ f0 + c·α·slope, slope>0) to the
                # minimisation form used by check_armijo.
                ok, msg = check_armijo(-f_new, -f0, c, α, -slope)
                _assert_invariant(ok, msg, :armijo,
                                  (; α=α, slope=slope, f0=f0, f_new=f_new))
            end
            return α
        end
        α *= rho
    end
    return α  # return smallest tried step
end

# ============================================================================
# Strong Wolfe line search with zoom
# ============================================================================

"""
    strong_wolfe_line_search(f::Function,
                              grad_f::Function,
                              u::Vector{Float64},
                              direction::Vector{Float64};
                              c1::Float64        = 1e-4,
                              c2::Float64        = 0.9,
                              alpha_max::Float64 = 10.0) -> Float64

Strong Wolfe conditions line search.

Finds a step size α satisfying:

    1. f(u + α d) ≥ f(u) + c1 α ⟨g₀, d⟩         (sufficient increase for maximization)
    2. |⟨∇f(u + α d), d⟩| ≤ c2 |⟨g₀, d⟩|         (curvature condition)

Uses the bracket-and-zoom strategy from Nocedal & Wright, Algorithm 3.5–3.6,
adapted for maximization.

# Arguments
- `f`         — objective (fidelity to maximize), `Vector{Float64} → Float64`
- `grad_f`    — gradient function, `Vector{Float64} → Vector{Float64}`
- `u`         — current iterate
- `direction` — ascent direction; must satisfy ⟨∇f(u), d⟩ > 0
- `c1`        — sufficient-increase constant ∈ (0, c2), default 1e-4
- `c2`        — curvature constant ∈ (c1, 1), default 0.9
- `alpha_max` — upper bound for the search interval, default 10.0

# Returns
Step size α satisfying (approximately) the strong Wolfe conditions.

# Example
```julia
α = strong_wolfe_line_search(f, grad_f, u, d; c1=1e-4, c2=0.9)
```
"""
function strong_wolfe_line_search(f::Function,
                                   grad_f::Function,
                                   u::Vector{Float64},
                                   direction::Vector{Float64};
                                   c1::Float64        = 1e-4,
                                   c2::Float64        = 0.9,
                                   alpha_max::Float64 = 10.0,
                                   check_invariants::Bool = false)::Float64
    0 < c1 < c2 < 1   || throw(ArgumentError("Need 0 < c1 < c2 < 1"))
    alpha_max > 0      || throw(ArgumentError("alpha_max must be positive"))

    f0  = f(u)
    g0  = grad_f(u)
    φ0  = dot(g0, direction)   # > 0 for ascent direction

    # ---- Phase 1: bracket search -------------------------------------------
    α_prev = 0.0
    f_prev = f0
    φ_prev = φ0
    α_curr = min(1.0, alpha_max)

    max_outer = 25

    for _ in 1:max_outer
        f_curr = f(u .+ α_curr .* direction)
        g_curr = grad_f(u .+ α_curr .* direction)
        φ_curr = dot(g_curr, direction)

        # Armijo violated or worse than previous step
        if (f_curr < f0 + c1 * α_curr * φ0) || (f_curr < f_prev)
            # Bracket found: zoom between α_prev and α_curr
            α_star = _wolfe_zoom(f, grad_f, u, direction, f0, φ0,
                                  α_prev, α_curr, f_prev, f_curr, c1, c2)
            return α_star
        end

        # Strong curvature condition met
        if abs(φ_curr) <= c2 * abs(φ0)
            if check_invariants
                ok, msg = check_wolfe_curvature(φ_curr, φ0, c2)
                _assert_invariant(ok, msg, :wolfe_curvature,
                                  (; α=α_curr, φ_curr=φ_curr, φ0=φ0))
            end
            return α_curr
        end

        # Derivative at α_curr is positive → bracket on other side
        if φ_curr <= 0.0
            α_star = _wolfe_zoom(f, grad_f, u, direction, f0, φ0,
                                  α_curr, α_prev, f_curr, f_prev, c1, c2)
            return α_star
        end

        # Expand interval
        α_prev = α_curr
        f_prev = f_curr
        φ_prev = φ_curr
        α_curr = min(2 * α_curr, alpha_max)

        if α_curr >= alpha_max
            break
        end
    end

    return α_curr
end

"""
    _wolfe_zoom(f, grad_f, u, d, f0, φ0, α_lo, α_hi,
                f_lo, f_hi, c1, c2) -> Float64

Zoom phase of the strong Wolfe line search.  Called when a bracket [α_lo, α_hi]
containing the desired α is known.  Uses bisection to narrow the bracket.
"""
function _wolfe_zoom(f, grad_f, u, d, f0, φ0,
                      α_lo, α_hi, f_lo, f_hi, c1, c2)::Float64
    for _ in 1:20
        α_j = (α_lo + α_hi) / 2.0
        f_j = f(u .+ α_j .* d)
        g_j = grad_f(u .+ α_j .* d)
        φ_j = dot(g_j, d)

        if (f_j < f0 + c1 * α_j * φ0) || (f_j < f_lo)
            α_hi = α_j
            f_hi = f_j
        else
            if abs(φ_j) <= c2 * abs(φ0)
                return α_j
            end
            if φ_j * (α_hi - α_lo) >= 0.0
                α_hi = α_lo
                f_hi = f_lo
            end
            α_lo = α_j
            f_lo = f_j
        end

        if abs(α_hi - α_lo) < 1e-14
            break
        end
    end
    return (α_lo + α_hi) / 2.0
end

# ============================================================================
# Diagonal preconditioning
# ============================================================================

"""
    apply_diagonal_preconditioning(gradient::Matrix{Float64},
                                    hessian_diag::Vector{Float64}) -> Matrix{Float64}

Apply diagonal preconditioning to a gradient matrix.

Computes the preconditioned gradient:

    g_prec[j, k] = g[j, k] / max(|h_ii|, ε)

where `h_ii` is the diagonal Hessian element corresponding to parameter (j, k)
and ε = 1e-14 prevents division by near-zero.

The preconditioned gradient is the Newton direction in the subspace approximated
by the diagonal Hessian.  For problems with large differences in curvature
across parameters, this can significantly accelerate convergence.

# Arguments
- `gradient`      — gradient matrix `[n_controls × n_timesteps]`
- `hessian_diag`  — diagonal Hessian elements, flattened to length `n_controls * n_timesteps`

# Returns
Preconditioned gradient matrix of the same shape as `gradient`.

# Notes
For gradient *ascent* (fidelity maximization), the Hessian of the *fidelity*
should be negative semi-definite at a maximum, so `|h_ii|` effectively normalizes
by the (positive) curvature magnitude.

# Example
```julia
h_diag = estimate_hessian_diagonal(sys, seq, target)
g_prec = apply_diagonal_preconditioning(G, h_diag)
controls .+= step_size .* g_prec
```
"""
function apply_diagonal_preconditioning(gradient::Matrix{Float64},
                                         hessian_diag::Vector{Float64})::Matrix{Float64}
    n_c, n_t = size(gradient)
    n = n_c * n_t
    if length(hessian_diag) != n
        throw(DimensionMismatch(
            "hessian_diag length $(length(hessian_diag)) ≠ " *
            "n_controls * n_timesteps = $n"))
    end

    g_prec = copy(gradient)
    g_flat = vec(g_prec)

    @inbounds for i in 1:n
        denom     = max(abs(hessian_diag[i]), 1e-14)
        g_flat[i] /= denom
    end

    return g_prec
end

# ============================================================================
# Hessian diagonal estimation
# ============================================================================

"""
    estimate_hessian_diagonal(system::AbstractQuantumSystem,
                               controls::ControlSequence,
                               target::QuantumTarget;
                               eps::Float64 = 1e-4) -> Vector{Float64}

Estimate the diagonal elements of the fidelity Hessian via finite differences.

Uses the second-order central-difference formula for each parameter (j, k):

    h_ii ≈ (F(u + ε eᵢ) - 2 F(u) + F(u - ε eᵢ)) / ε²

where eᵢ is the unit vector along parameter i.

Cost: `2 * n_controls * n_timesteps + 1` fidelity evaluations.

# Arguments
- `system`   — quantum system
- `controls` — control sequence at which to estimate the Hessian diagonal
- `target`   — optimization target
- `eps`      — finite-difference step size (default 1e-4)

# Returns
Vector of length `n_controls * n_timesteps` with diagonal Hessian estimates.

# Notes
- At a maximum of the fidelity, diagonal Hessian elements should be non-positive.
- Elements near zero may indicate flat directions (poor conditioning).
- Use `apply_diagonal_preconditioning` to apply the result.

# Example
```julia
h_diag = estimate_hessian_diagonal(sys, seq, target; eps=1e-4)
println("Condition estimate: ", maximum(abs, h_diag) / (minimum(abs, h_diag) + 1e-14))
```
"""
function estimate_hessian_diagonal(system::AbstractQuantumSystem,
                                    controls::ControlSequence,
                                    target::QuantumTarget;
                                    eps::Float64 = 1e-4)::Vector{Float64}
    eps > 0 || throw(ArgumentError("eps must be positive"))

    dt  = controls.dt
    n_c = system.n_controls
    n_t = controls.n_timesteps
    n   = n_c * n_t
    u0  = vec(copy(controls.controls))

    # Evaluate fidelity at center
    f = u_v -> begin
        seq = ControlSequence(reshape(u_v, n_c, n_t), dt, dt * n_t, n_t)
        H   = build_total_hamiltonian(system, seq)
        U   = compute_propagators(H, dt)
        compute_fidelity(compute_total_propagator(U), target)
    end

    F0 = f(u0)
    h_diag = zeros(Float64, n)

    @inbounds for i in 1:n
        u_p    = copy(u0);  u_p[i] += eps
        u_m    = copy(u0);  u_m[i] -= eps
        h_diag[i] = (f(u_p) - 2*F0 + f(u_m)) / (eps^2)
    end

    return h_diag
end

# ============================================================================
# Barzilai-Borwein step size
# ============================================================================

"""
    barzilai_borwein_step(s::Vector{Float64}, y::Vector{Float64};
                           method::Symbol = :long) -> Float64

Compute the Barzilai-Borwein (BB) step size from consecutive iterates.

The BB step size provides a quasi-Newton-like curvature estimate without
forming the Hessian:

- `:long`  (BB1): α = s⊤s / s⊤y
- `:short` (BB2): α = s⊤y / y⊤y

where s = u_new - u_old and y = g_new - g_old are the parameter and gradient
differences between consecutive iterations.

# Arguments
- `s`      — parameter change vector (u_new - u_old)
- `y`      — gradient change vector (g_new - g_old)
- `method` — `:long` (default) or `:short`

# Returns
Positive step size α.  Falls back to `1.0` if the denominator is near zero.

# Notes
BB steps are often used in the spectral projected gradient (SPG) method.
They can be non-monotone (fidelity may decrease at individual steps) but
achieve superlinear convergence on quadratic objectives.

# Example
```julia
α = barzilai_borwein_step(u_new - u_old, g_new - g_old; method=:long)
```
"""
function barzilai_borwein_step(s::Vector{Float64},
                                y::Vector{Float64};
                                method::Symbol = :long)::Float64
    length(s) == length(y) ||
        throw(DimensionMismatch("s and y must have the same length"))
    method in (:long, :short) ||
        throw(ArgumentError("method must be :long or :short"))

    ss = dot(s, s)
    sy = dot(s, y)
    yy = dot(y, y)

    if method == :long
        denom = sy
        return abs(denom) < 1e-14 ? 1.0 : abs(ss / denom)
    else
        denom = yy
        return abs(denom) < 1e-14 ? 1.0 : abs(sy / denom)
    end
end

# ============================================================================
# Step size schedule utilities
# ============================================================================

"""
    cosine_decay_schedule(iter::Int, max_iter::Int, lr_init::Float64;
                           lr_min::Float64 = 0.0) -> Float64

Cosine annealing learning rate schedule.

    η(t) = η_min + (η_init - η_min) * (1 + cos(π t / T)) / 2

where t = `iter`, T = `max_iter`.

Provides gradual learning rate warmdown, often helpful when fine-tuning near
a high-fidelity solution.

# Arguments
- `iter`      — current iteration (1-based)
- `max_iter`  — total number of iterations
- `lr_init`   — initial learning rate η_init
- `lr_min`    — minimum learning rate η_min (default 0.0)

# Returns
Learning rate for iteration `iter`.

# Example
```julia
for iter in 1:1000
    η = cosine_decay_schedule(iter, 1000, 0.01; lr_min=1e-5)
    # use η as the step size
end
```
"""
function cosine_decay_schedule(iter::Int,
                                max_iter::Int,
                                lr_init::Float64;
                                lr_min::Float64 = 0.0)::Float64
    iter >= 1     || throw(ArgumentError("iter must be ≥ 1"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1"))
    lr_init > 0   || throw(ArgumentError("lr_init must be positive"))
    lr_min >= 0   || throw(ArgumentError("lr_min must be ≥ 0"))
    lr_min < lr_init || throw(ArgumentError("lr_min must be < lr_init"))

    t = clamp(iter - 1, 0, max_iter - 1)
    return lr_min + (lr_init - lr_min) * (1.0 + cos(π * t / max_iter)) / 2.0
end

"""
    polynomial_decay_schedule(iter::Int, lr_init::Float64, decay_rate::Float64;
                               power::Float64 = 1.0) -> Float64

Polynomial decay learning rate schedule.

    η(t) = η_init / (1 + decay_rate * t)^power

# Arguments
- `iter`        — current iteration (1-based)
- `lr_init`     — initial learning rate
- `decay_rate`  — decay coefficient (must be positive)
- `power`       — exponent for polynomial decay (default 1.0 = inverse decay)

# Returns
Learning rate at iteration `iter`.

# Example
```julia
η = polynomial_decay_schedule(iter, 0.01, 0.01; power=0.75)
```
"""
function polynomial_decay_schedule(iter::Int,
                                    lr_init::Float64,
                                    decay_rate::Float64;
                                    power::Float64 = 1.0)::Float64
    iter >= 1      || throw(ArgumentError("iter must be ≥ 1"))
    lr_init > 0    || throw(ArgumentError("lr_init must be positive"))
    decay_rate > 0 || throw(ArgumentError("decay_rate must be positive"))
    power > 0      || throw(ArgumentError("power must be positive"))

    t = iter - 1
    return lr_init / (1.0 + decay_rate * t)^power
end
