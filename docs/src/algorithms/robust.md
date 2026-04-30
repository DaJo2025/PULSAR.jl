# Robust optimization

Real systems suffer parameter drift (B0, B1, T2, J-couplings) and pulses
must remain high-fidelity across the resulting parameter ensemble.
PULSAR's `robust_optimize` (in
[`src/Optimization/Robust/RobustOpt.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/Robust/RobustOpt.jl))
optimizes a chosen aggregate of fidelities over a parametric ensemble.

## API

```julia
robust_cfg = RobustConfig(
    uncertainty_type      = "parametric",      # or "amplitude", "frequency"
    uncertainty_magnitude = 0.10,              # ±10 %
    robustness_measure    = "mean",            # or "worst_case"
    n_samples             = 21,
    base_method           = "lbfgs",
    verbose               = false,
)

result = robust_optimize(sys, target, ctrl;
                          config       = robust_cfg,
                          lbfgs_config = LBFGSConfig(...),
                          penalty_fns  = penalty_fns)
```

## Aggregate measures

| `robustness_measure` | Aggregate |
|---|---|
| `"mean"` | `(1/N) Σᵢ Fᵢ` — average fidelity |
| `"worst_case"` | `min Fᵢ` — minimax / worst-case |

`"mean"` produces a smooth gradient and is the default for moderate
uncertainty. `"worst_case"` yields more conservative but flatter pulses; the
gradient is the gradient of the worst single member at each iterate.

## Uncertainty types

| `uncertainty_type` | Perturbs |
|---|---|
| `"parametric"` | A user-chosen field (offset, J, T2, …) |
| `"amplitude"` | Multiplicative scaling of each control channel |
| `"frequency"` | Resonance-offset distribution |

The actual ensemble is built by drawing `n_samples` from a uniform / Gaussian
grid (see source); for custom distributions, build the ensemble yourself with
`ensemble_fidelity` and feed it to GRAPE-ensemble directly.

## Worked example — frequency-robust 1-qubit X gate

```julia
robust_cfg = RobustConfig(
    uncertainty_type      = "frequency",
    uncertainty_magnitude = 0.05,           # ±5 %
    robustness_measure    = "mean",
    n_samples             = 15,
    base_method           = "lbfgs",
)

result = robust_optimize(sys, target, ctrl;
                          config = robust_cfg,
                          lbfgs_config = LBFGSConfig(max_iter=400, verbose=false))

# Independent ensemble re-evaluation
ctrl_opt = ControlSequence(Matrix(result.controls'), ctrl.dt, ctrl.n_steps)
F_avg    = ensemble_fidelity(sys, ctrl_opt, target;
                              uncertainty_type="frequency",
                              uncertainty_magnitude=0.05,
                              n_samples=15)
@assert F_avg ≥ result.fidelity - 1e-6
```

## QC platform-specific noise models

For superconducting / trapped-ion / neutral-atom systems, the
[`Application/QuantumComputing/NoiseModels/`](https://github.com/DaJo2025/PULSAR.jl/tree/main/src/Application/QuantumComputing/NoiseModels)
helpers provide `QuasiStaticNoise`, `MarkovianNoise`, and
`NoiseSpectrum` types that integrate directly with `robust_optimcon_qs`,
`lindblad_optimcon`, and `optimcon_ff` (filter function) wrappers.
