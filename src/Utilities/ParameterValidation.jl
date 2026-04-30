# ============================================================
# Pulsar.jl — Parameter Validation
# Pulse Design Library for Spin Control Algorithms and Rollout
# ============================================================
#
# Pre-flight checks that catch common physics/programming errors
# before they manifest as silent NaN or incorrect results.
# ============================================================

using LinearAlgebra

# ──────────────────────────────────────────────────────────────
# Matrix property checks
# ──────────────────────────────────────────────────────────────

"""
    check_hermitian(M; tol) -> Bool

Return true if M is Hermitian: ‖M - M†‖_F < tol·‖M‖_F.
"""
function check_hermitian(M::AbstractMatrix; tol::Float64 = 1e-10)::Bool
    nM = norm(M)
    nM < eps() && return true          # zero matrix is Hermitian
    return norm(M - M') < tol * max(nM, 1.0)
end

"""
    check_unitary(U; tol) -> Bool

Return true if U is unitary: ‖U†U - I‖_F < tol.
"""
function check_unitary(U::AbstractMatrix; tol::Float64 = 1e-8)::Bool
    n = size(U, 1)
    size(U, 2) == n || return false
    return norm(U' * U - I(n)) < tol * n
end

"""
    check_normalized(v; tol) -> Bool

Return true if the vector v has unit norm: | ‖v‖ - 1 | < tol.
"""
function check_normalized(v::AbstractVector; tol::Float64 = 1e-10)::Bool
    return abs(norm(v) - 1.0) < tol
end

"""
    check_positive_semidefinite(M; tol) -> Bool

Return true if M is positive semidefinite (all eigenvalues ≥ -tol).
"""
function check_positive_semidefinite(M::AbstractMatrix; tol::Float64 = 1e-10)::Bool
    check_hermitian(M) || return false
    λ = real(eigvals(Hermitian(M)))
    return minimum(λ) >= -tol
end

# ──────────────────────────────────────────────────────────────
# System validation
# ──────────────────────────────────────────────────────────────

"""
    validate_system(system)

Check that a quantum system is physically valid.

# Checks performed
1. H_drift is a square matrix
2. H_drift is Hermitian (‖H - H†‖_F / ‖H‖_F < 1e-10)
3. Each H_control is Hermitian and the same size as H_drift
4. `system.dim` matches size(H_drift, 1)
5. `system.n_controls` matches length(system.H_controls)

Throws `ArgumentError` with a descriptive message if any check fails.
"""
function validate_system(system::AbstractQuantumSystem)
    errors = String[]

    # Square H_drift
    sz = size(system.H_drift)
    sz[1] == sz[2] || push!(errors,
        "H_drift must be square; got size $(sz)")

    # Hermitian H_drift
    check_hermitian(system.H_drift) || push!(errors,
        "H_drift is not Hermitian: ‖H-H†‖=$(norm(system.H_drift - system.H_drift'))")

    # dim consistency
    system.dim == sz[1] || push!(errors,
        "system.dim=$(system.dim) does not match size(H_drift,1)=$(sz[1])")

    # n_controls consistency
    system.n_controls == length(system.H_controls) || push!(errors,
        "system.n_controls=$(system.n_controls) does not match " *
        "length(H_controls)=$(length(system.H_controls))")

    # Each H_control
    for (j, Hc) in enumerate(system.H_controls)
        size(Hc) == sz || push!(errors,
            "H_controls[$j] has size $(size(Hc)); expected $(sz)")
        check_hermitian(Hc) || push!(errors,
            "H_controls[$j] is not Hermitian: ‖H-H†‖=$(norm(Hc - Hc'))")
    end

    isempty(errors) || throw(ArgumentError(
        "Invalid quantum system:\n" * join("  • " .* errors, "\n")))
    nothing
end

# ──────────────────────────────────────────────────────────────
# Controls validation
# ──────────────────────────────────────────────────────────────

"""
    validate_controls(controls)

Check that a ControlSequence is consistent and finite.

# Checks performed
1. dt > 0
2. n_timesteps ≥ 1
3. size(controls.controls, 2) == n_timesteps
4. No NaN or Inf values
5. total_time ≈ dt * n_timesteps  (relative tolerance 1e-6)
"""
function validate_controls(controls::ControlSequence)
    errors = String[]

    controls.dt > 0 || push!(errors, "dt must be positive; got dt=$(controls.dt)")
    controls.n_timesteps >= 1 || push!(errors,
        "n_timesteps must be ≥ 1; got $(controls.n_timesteps)")

    sz2 = size(controls.controls, 2)
    sz2 == controls.n_timesteps || push!(errors,
        "controls matrix has $(sz2) columns but n_timesteps=$(controls.n_timesteps)")

    any(isnan, controls.controls) && push!(errors, "NaN detected in controls matrix")
    any(isinf, controls.controls) && push!(errors, "Inf detected in controls matrix")

    expected_T = controls.dt * controls.n_timesteps
    if abs(controls.total_time - expected_T) > 1e-6 * expected_T
        push!(errors, "total_time=$(controls.total_time) ≠ dt×n_timesteps=$(expected_T)")
    end

    isempty(errors) || throw(ArgumentError(
        "Invalid ControlSequence:\n" * join("  • " .* errors, "\n")))
    nothing
end

# ──────────────────────────────────────────────────────────────
# Target validation
# ──────────────────────────────────────────────────────────────

"""
    validate_target(target, system)

Check that an optimization target is physically valid for the given system.

# Checks performed
- **State target**: state_vector is normalised, length == system.dim
- **Unitary target**: matrix is unitary, size == (system.dim, system.dim)
"""
function validate_target(target::QuantumTarget, system::AbstractQuantumSystem)
    errors = String[]
    d = system.dim

    if target.type == "state"
        if isnothing(target.target_state)
            push!(errors, "target_state is nothing for a state target")
        else
            length(target.target_state) == d || push!(errors,
                "target_state has length $(length(target.target_state)); expected $d")
            check_normalized(target.target_state) || push!(errors,
                "target_state is not normalised: ‖ψ‖=$(norm(target.target_state))")
        end

    elseif target.type == "unitary"
        if isnothing(target.target_unitary)
            push!(errors, "target_unitary is nothing for a unitary target")
        else
            sz = size(target.target_unitary)
            sz == (d, d) || push!(errors,
                "target_unitary has size $(sz); expected ($d, $d)")
            check_unitary(target.target_unitary) || push!(errors,
                "target_unitary is not unitary: ‖U†U-I‖=$(norm(target.target_unitary' * target.target_unitary - I(d)))")
        end

    else
        push!(errors, "Unknown target type '$(target.type)'; expected \"state\" or \"unitary\"")
    end

    isempty(errors) || throw(ArgumentError(
        "Invalid QuantumTarget:\n" * join("  • " .* errors, "\n")))
    nothing
end

# ──────────────────────────────────────────────────────────────
# Optimization result validation
# ──────────────────────────────────────────────────────────────

"""
    validate_optimization_result(result) -> Vector{String}

Check an OptimizationResult for anomalies.

Returns a (possibly empty) vector of warning strings rather than throwing.
"""
function validate_optimization_result(result::OptimizationResult)::Vector{String}
    warnings = String[]

    (0.0 <= result.fidelity <= 1.0 + 1e-8) || push!(warnings,
        "Fidelity $(result.fidelity) is outside [0, 1]")

    any(isnan, result.controls) && push!(warnings,
        "NaN detected in optimised controls")
    any(isinf, result.controls) && push!(warnings,
        "Inf detected in optimised controls")

    isempty(result.fidelity_history) || begin
        if result.fidelity_history[end] < result.fidelity_history[1] - 0.05
            push!(warnings, "Final fidelity is significantly lower than initial fidelity")
        end
    end

    return warnings
end

# ──────────────────────────────────────────────────────────────
# Convenience: validate everything at once
# ──────────────────────────────────────────────────────────────

"""
    validate_all(system, target, controls)

Run all pre-flight validation checks in a single call.
Throws on first critical error; prints warnings for soft issues.

# Example
```julia
validate_all(system, target, controls_init)
result = grape_optimize(system, target, controls_init)
```
"""
function validate_all(
    system::AbstractQuantumSystem,
    target::QuantumTarget,
    controls::ControlSequence,
)
    validate_system(system)
    validate_target(target, system)
    validate_controls(controls)
    # Cross-check dimensions
    size(controls.controls, 1) == system.n_controls || throw(ArgumentError(
        "controls matrix has $(size(controls.controls,1)) rows but " *
        "system.n_controls=$(system.n_controls)"))
    nothing
end
