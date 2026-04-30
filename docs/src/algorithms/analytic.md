# Analytic pulses

Closed-form pulse families that achieve targeted fidelities by construction —
no iterative optimization needed. Source:
[`src/Optimization/Analytic/`](https://github.com/DaJo2025/PULSAR.jl/tree/main/src/Optimization/Analytic).

## Composite pulses

Composite pulses chain several rectangular RF segments at carefully chosen
phases / durations to cancel specific error terms (off-resonance, B1 inhomogeneity).

| Function | Family | Compensates |
|---|---|---|
| `bb1` | BB1 (Wimperis) | B1 amplitude error |
| `scrofulous` | SCROFULOUS | B1 + offset |
| `sk1` | Solovay–Kitaev 1 | Systematic over-rotation |
| `corpse` | CORPSE | Off-resonance |
| `short_corpse` | Short CORPSE | Off-resonance, fewer segments |
| `f1` | F1 | First-order detuning |
| `g1` | G1 | First-order amplitude |
| `corpse_in_bb1` | CORPSE-in-BB1 | Joint amplitude + offset |

Each returns an `AnalyticPulse` consisting of `CompositePulseSegment`s. They
can be sampled to a `ControlSequence` for direct use or comparison.

## DRAG

Derivative-removal-by-adiabatic-gate pulses for reducing leakage in
multi-level qubits (transmons, trapped ions).

```julia
pulse = drag_pulse(amplitude, duration; α=anharmonicity)
```

## Shortcuts to adiabaticity (STA)

`sta_fourier_1d` ([`src/Optimization/Analytic/STA.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Analytic/STA.jl))
returns 1-D Fourier-basis shortcuts for adiabatic-like state transfer.

## SLR

Shinnar–Le Roux pulse design for MRI selective excitation
([`src/Optimization/Analytic/SLR.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Analytic/SLR.jl)).

```julia
pulse = slr_1d(bandwidth_hz, duration_s; ripple, ...)
```

## VERSE

Variable-rate selective excitation. Reshapes an existing pulse to satisfy a
peak-amplitude or acoustic-noise constraint while preserving slice profile
([`src/Optimization/Analytic/VERSE.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Analytic/VERSE.jl)).

| Function | Use |
|---|---|
| `verse` | Standard VERSE rescaling |
| `verse_min_time` | Minimum-time VERSE |
| `verse_acoustic_noise` | Acoustic-noise-aware VERSE |

## When to use analytic pulses

- **Robustness without iteration**: a SCROFULOUS or CORPSE produces a
  parameter-tolerant pulse instantly
- **Initial guess** for GRAPE / Krotov on hard problems — analytic seeds often
  beat random initialization
- **Comparison baseline** when benchmarking optimal-control methods

For comparison-grade workflows, sample the analytic pulse to a
`ControlSequence` then drive it through `compute_fidelity` or any optimizer
for refinement.
