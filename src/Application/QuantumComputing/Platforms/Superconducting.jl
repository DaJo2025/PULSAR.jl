# Application/QuantumComputing/Platforms/Superconducting.jl
# Optimal control entry point for superconducting transmon qubits.
#
# Provides:
#   optimcon(sys::TransmonSystem, ...)  — GRAPE with DRAG + leakage suppression
#   (QuantumSystem assembly lives in Platforms/Common.jl as _build_platform_qs)

using LinearAlgebra

# QuantumSystem assembly (including Lindblad branch) is shared with the other
# platforms via `_build_platform_qs` in `Common.jl`.

# ============================================================================
# optimcon overload for TransmonSystem
# ============================================================================

"""
    optimcon(sys::TransmonSystem, target::QuantumTarget, ctrl::ControlSequence;
             config, use_lindblad, leakage_weight, drag_beta) -> OptimizationResult

Superconducting-qubit optimal control with automatic:
  1. DRAG-inspired leakage penalty (if `leakage_weight > 0` and leakage levels exist)
  2. Optional Lindblad open-system propagation (if `use_lindblad=true` and T1/T2 are set)

The optimiser used is GRAPE with the physics-hook interface.

# Arguments
- `sys`             — `TransmonSystem` (from `transmon_system(...)`)
- `target`          — `QuantumTarget` (from `unitary_target` or `state_target`)
- `ctrl`            — initial `ControlSequence`
- `config`          — `GRAPEConfig` (default: `GRAPEConfig()`)
- `use_lindblad`    — use Lindblad propagation (T₁/T₂ decoherence)
- `leakage_weight`  — weight λ for leakage penalty in the objective
  F_total = F_gate − λ × P_leak  (default 0.1)
- `drag_beta`       — if set (Float64 **seconds**), apply DRAG correction to the
  initial guess waveform before running GRAPE (pre-conditioning only).
  Convention: `Ωy += (−β) dΩx/dt` where β has units of seconds — i.e. the
  textbook factor `1/Δ` (Δ = anharmonicity in rad/s) is already absorbed into
  β. Typical choice: `β = 1 / (2π · anharm_hz)`.

# Returns
`OptimizationResult`

# Example
```julia
sys    = transmon_system(5.0e9, -200e6; T1_s=50e-6, T2_s=30e-6)
target = unitary_target([0 1; 1 0] .+ 0im)  # X gate on computational subspace
ctrl   = ControlSequence(0.01 .* randn(sys.n_controls, 200), 5e-9, 1e-6, 200)
result = optimcon(sys, target, ctrl; leakage_weight=0.1)
```
"""
function optimcon(sys            :: TransmonSystem,
                  target         :: QuantumTarget,
                  ctrl           :: ControlSequence;
                  config         :: GRAPEConfig = GRAPEConfig(),
                  use_lindblad   :: Bool         = false,
                  leakage_weight :: Float64      = 0.1,
                  drag_beta      :: Union{Float64, Nothing} = nothing)::OptimizationResult

    # Apply DRAG pre-conditioning to x-channel of initial guess
    w0 = copy(ctrl.controls)
    if !isnothing(drag_beta)
        for q in 1:sys.n_qubits
            xi = 2q - 1    # x-channel index for qubit q
            yi = 2q        # y-channel index
            dΩ = _numeric_derivative(w0[xi, :], ctrl.dt)
            w0[yi, :] .+= (-drag_beta) .* dΩ
        end
    end
    ctrl0 = ControlSequence(w0, ctrl.dt, ctrl.total_time, ctrl.n_timesteps)

    # Build penalty hooks if leakage levels exist
    if leakage_weight > 0.0 && !isempty(sys.leakage_indices)
        leak_idx = sys.leakage_indices
        lw       = leakage_weight

        penalty_fn = function(system, c, tgt)
            # Forward propagate to get U_total
            Us    = compute_propagators(system, c)
            U_tot = foldl((A, B) -> B * A,
                          [Us[k, :, :] for k in 1:c.n_timesteps])
            return leakage_penalty(U_tot, leak_idx; weight=lw)
        end

        penalty_grad_fn = function(system, c, tgt)
            Us    = compute_propagators(system, c)
            P     = compute_forward_propagators(Us)
            U_tot = P[c.n_timesteps + 1, :, :]
            PiL   = _diagonal_projector(system.dim, leak_idx)
            Q     = compute_backward_propagators(Us, U_tot' * PiL)
            return leakage_gradient(P, Q, system.H_controls, leak_idx,
                                    c.dt; weight=lw)
        end

        return _platform_grape(sys, target, ctrl0;
                               config           = config,
                               use_lindblad     = use_lindblad,
                               penalty_fns      = [penalty_fn],
                               penalty_grad_fns = [penalty_grad_fn])
    else
        return _platform_grape(sys, target, ctrl0;
                               config = config, use_lindblad = use_lindblad)
    end
end
