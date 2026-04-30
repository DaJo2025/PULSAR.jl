"""
    MR/GRAPEPhase.jl

Phase-only GRAPE kernel for state-transfer optimisation.

Wraps `grape_state_kernel` with a polar→Cartesian conversion so that the
optimizer works directly in phase space while the amplitude profile is held
fixed.  Polar parameterisation following Goodwin & Kuprov.

## Parameterisation

Controls are grouped into `n_channels = length(ctrl.operators) ÷ 2` (x,y)
pairs.  For channel `ch` at step `n`:

    w[2ch−1, n] = amplitudes[ch, n] × cos(φ[ch, n])   # Bx
    w[2ch,   n] = amplitudes[ch, n] × sin(φ[ch, n])   # By

The amplitude profile is supplied by the caller and kept constant throughout
optimisation.  When omitted it defaults to 1.0 (normalised full power) at
every step.

## Gradient (chain rule)

    ∂F/∂φ[ch, n] = −G_x[ch,n] × sin(φ[ch,n]) + G_y[ch,n] × cos(φ[ch,n])

where `G_x`, `G_y` are the Cartesian gradients returned by
`grape_state_kernel`.  The amplitude gradient is not computed — the caller
is responsible for any amplitude penalty terms handled separately.
"""

"""
    grape_phase_kernel(phi_profile, ctrl; amplitudes) → (F, G_phase)

Phase-only GRAPE kernel.  Returns the ensemble-averaged fidelity and its
gradient with respect to the phase profile.

# Arguments
- `phi_profile :: Matrix{Float64}` — shape `[n_channels × n_steps]`, phases
  in radians.  One row per (Bx, By) channel pair.
- `ctrl`                           — `MRControl` problem definition (same object used with
  `grape_state_kernel`).  Must have an even number of operators.

# Keyword argument
- `amplitudes  :: Matrix{Float64}` — shape `[n_channels × n_steps]`, normalised
  amplitudes in `[0, 1]`.  Defaults to `ones(n_channels, n_steps)` (constant
  full power).

# Returns
- `F        :: Float64`            — ensemble-averaged fidelity ∈ [0, 1].
- `G_phase  :: Matrix{Float64}`    — gradient `∂F/∂φ`, shape
  `[n_channels × n_steps]`.
"""
function grape_phase_kernel(
    phi_profile :: Matrix{Float64},
    ctrl;
    amplitudes  :: Matrix{Float64} = ones(size(phi_profile)),
)
    n_channels, n_steps = size(phi_profile)

    if length(ctrl.operators) != 2 * n_channels
        throw(ArgumentError(
            "grape_phase_kernel: ctrl has $(length(ctrl.operators)) operators " *
            "but phi_profile has $n_channels rows — expected " *
            "length(operators) == 2 × n_channels."))
    end
    if size(amplitudes) != size(phi_profile)
        throw(ArgumentError(
            "grape_phase_kernel: amplitudes $(size(amplitudes)) must match " *
            "phi_profile $(size(phi_profile))."))
    end
    if any(amplitudes .< 0)
        throw(ArgumentError("grape_phase_kernel: all amplitudes must be ≥ 0."))
    end

    # Polar → Cartesian waveform  [n_ctrl × n_steps]  (n_ctrl = 2 × n_channels)
    w = zeros(Float64, 2 * n_channels, n_steps)
    for ch in 1:n_channels
        @views @. w[2ch-1, :] = amplitudes[ch, :] * cos(phi_profile[ch, :])
        @views @. w[2ch,   :] = amplitudes[ch, :] * sin(phi_profile[ch, :])
    end

    # Cartesian GRAPE kernel
    F, G = grape_state_kernel(w, ctrl)

    # Chain rule: ∂F/∂φ[ch,n] = −G_x[ch,n]·sin(φ[ch,n]) + G_y[ch,n]·cos(φ[ch,n])
    G_phase = zeros(Float64, n_channels, n_steps)
    for ch in 1:n_channels
        @views @. G_phase[ch, :] = -G[2ch-1, :] * sin(phi_profile[ch, :]) +
                                     G[2ch,   :] * cos(phi_profile[ch, :])
    end

    return F, G_phase
end
