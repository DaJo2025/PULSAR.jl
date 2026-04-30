# MRI — magnetic resonance imaging

MRI pulse design works in the classical Bloch-vector regime: track
magnetization `M = (Mx, My, Mz)` per isochromat / voxel rather than the
full density matrix.

Pulsar's MRI support is in
[`src/Types/BlochSystem.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Types/BlochSystem.jl),
[`src/Computation/BlochPropagator.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Computation/BlochPropagator.jl),
and
[`src/Application/MR/MRI/MRIOptControl.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/MR/MRI/MRIOptControl.jl).

## Building a Bloch system

`BlochIsochromat` carries a position, T1, T2, ΔB0, equilibrium density,
and an initial magnetization `M0`:

```julia
isos = [
    BlochIsochromat(
        (x, 0.0, 0.0),                # position (m)
        1.0,                          # T1 (s)
        0.1,                          # T2 (s)
        0.0,                          # ΔB0 (T)
        1.0,                          # ρ₀ (a.u.)
        (0.0, 0.0, 1.0),              # M0
    )
    for x in range(-0.05, 0.05; length = 256)
]

grad = GradientSystem(40e-3, 200.0, 4e-6)   # (G_max [T/m], slew [T/m/s], raster [s])
sys  = bloch_system(isos;
                    gamma            = 2.675e8,         # ¹H γ in rad/s/T
                    gradient_system  = grad,
                    B1_max_tesla     = 50e-6,
                    SAR_limit_Wkg    = 4.0)
```

## MRI control sequence

`MRIControlSequence` packages an RF waveform `B1[2 × n_steps]`, a gradient
waveform `G[3 × n_steps]`, time step, total time, number of steps, and
the legacy `amplitudes` matrix:

```julia
B1 = randn(2, n_steps) .* 5e-6                # Bx, By in tesla
G  = zeros(3, n_steps)                         # Gx, Gy, Gz in T/m
ctrl = MRIControlSequence(B1, G, dt, dt * n_steps, n_steps, B1)
```

## Bloch propagator

```julia
M_final = bloch_forward_pass(sys, ctrl)        # 3 × n_isochromats
F       = bloch_fidelity(sys, M_final, M_target)
F_slice = slice_profile_fidelity(sys, M_final, M_target)
```

`bloch_adjoint_pass` produces the adjoint variables for analytic
gradient-based optimisation; `bloch_fidelity` is `<M_final, M_target>`
weighted by `ρ₀`, `slice_profile_fidelity` is the same with magnetization
restricted to the transverse plane.

## High-level optimisation

```julia
result = optimcon(sys, M_target, ctrl;
                  config        = LBFGSConfig(max_iter = 200),
                  sigma_Sm      = 0.5,                  # tissue conductivity
                  rho_kgm3      = 1000.0,
                  lambda_SAR    = 1e-2,
                  lambda_slew   = 1e-3)
```

## SAR and slew-rate penalties

| Function | Signature | Purpose |
|---|---|---|
| `sar_penalty`        | `sar_penalty(ctrl::MRIControlSequence, sigma_Sm, rho_kgm3)` | SAR cost `(1/2σρ) · ‖B1‖²·dt` |
| `sar_gradient`       | `sar_gradient(ctrl, sigma_Sm, rho_kgm3)`         | Analytic gradient w.r.t. B1 |
| `slew_rate_penalty`  | `slew_rate_penalty(G::Matrix, dt, G_max_slew)`    | Slew-rate cost on gradient channels |
| `slew_rate_gradient` | `slew_rate_gradient(G, dt, G_max_slew)`           | Analytic gradient w.r.t. G |

These plug into the `optimcon(::BlochSystem, …)` wrapper through
`lambda_SAR` / `lambda_slew`, or can be combined with `EnergyPenalty` and
`SmoothnessPenalty` for soft hardware-limit shaping.

## Common patterns

- **B1-robust slice excitation** — multi-isochromat Bloch ensemble +
  robust optimisation over a measured B1 map.
- **Spectrally-spatially selective** — joint RF + gradient design with
  `slice_profile_fidelity` over an offset × position grid.
- **VERSE post-processing** — peak-amplitude reduction with preserved
  profile via `verse(...)` (see
  [Analytic pulses](../algorithms/analytic.md)).
- **SLR initial guesses** — `slr_1d(...)` for strong starting points
  before gradient optimisation.
