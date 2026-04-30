# API — Utilities

Validation and visualization helpers.
Source: [`src/Utilities/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Utilities).

## Parameter validation

```@docs
validate_system
validate_controls
validate_target
validate_all
```

## Visualization

The plotting helpers require the `Plots` package extension
(`pkg> add Plots`).

```@docs
plot_convergence
plot_controls
plot_bloch_trajectory
plot_sensitivity_heatmap
plot_pareto_front
create_optimization_report
```
