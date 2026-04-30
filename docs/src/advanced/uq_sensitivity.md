# Uncertainty quantification & sensitivity

PULSAR provides post-hoc analysis tools for understanding how robust an
optimized pulse is — both globally (UQ) and per-parameter (sensitivity).

## Uncertainty quantification (UQ)

Source: [`src/Physics/UncertaintyQuantification.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Physics/UncertaintyQuantification.jl).

```julia
uq_cfg = UQConfig(
    parameters       = [:b1_amplitude, :offset_freq],
    distribution     = :uniform,                # or :gaussian
    magnitude        = [0.10, 50.0],            # ±10 % B1, ±50 Hz offset
    n_samples        = 256,
    aggregation      = :mean,                   # also :worst_case, :percentile
)

uq::UncertaintyResult = estimate_uncertainty(sys, target, result.controls;
                                              config=uq_cfg)

uq.mean_fidelity         # Float64
uq.std_fidelity          # Float64
uq.percentile_low        # 5th percentile
uq.percentile_high       # 95th percentile
uq.histogram             # Bin counts
```

Use this *after* optimization to validate the pulse against a richer ensemble
than the optimization saw.

## Sensitivity analysis

Source: [`src/Physics/Sensitivity.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Physics/Sensitivity.jl).

```julia
sens_cfg = SensitivityConfig(
    parameters = [:b1_amplitude, :offset_freq, :T1, :T2],
    method     = :finite_difference,            # or :sobol, :morris
    delta      = 1e-3,
)

sens::SensitivityResult = compute_sensitivity(sys, target, result.controls;
                                               config=sens_cfg)

sens.first_order   # ∂F/∂param vector
sens.total_order   # Total-effect Sobol indices (if :sobol)
sens.ranking       # Sorted by magnitude
```

Use sensitivity to identify which uncertain parameters dominate the residual
infidelity — those become candidates for inclusion in the next robust
optimization pass.

## Visualization

```julia
using Plots
plot_sensitivity_heatmap(sens)      # 2-D parameter heatmap
```

(Requires `pkg> add Plots`.)

## Recommended workflow

1. Optimize with `grape_optimize` or `robust_optimize`
2. Run UQ to get a fidelity distribution under realistic noise
3. Run sensitivity to rank the parameters by impact
4. Re-optimize with `RobustConfig` covering the top 1–3 sensitive parameters
5. Repeat 2–3 to confirm the ensemble fidelity meets the target

## Algorithm recommendation

For a problem-aware optimizer suggestion (skipping the manual choice):

```julia
rec = recommend_optimizer(sys, target, ctrl)
auto_optimize(sys, target, ctrl)        # uses rec internally
println(describe_recommendation(rec))
```

See [`Runtime/AlgorithmSelection.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Runtime/AlgorithmSelection.jl).
