# Multi-objective optimization

When several objectives compete (high fidelity *and* low energy *and* smooth
waveform *and* low peak amplitude), Pareto-front exploration is more
informative than a single weighted-sum optimum.

Source: [`src/Optimization/MultiObjective/MultiObjectiveOptimization.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/MultiObjective/MultiObjectiveOptimization.jl).

## API

```julia
config = MultiObjectiveConfig(
    method      = "nsga2",                  # or "weighted_sum", "epsilon"
    population  = 80,
    max_iter    = 200,
    objectives  = [
        fidelity_objective(sys, target),
        energy_objective(),
        smoothness_objective(),
        peak_amplitude_objective(),
    ],
    verbose     = false,
)

result::MultiObjectiveResult = multi_objective_optimize(ctrl; config=config)
```

## Built-in objective constructors

| Constructor | Returns |
|---|---|
| `fidelity_objective(sys, target)` | `1 − F(w)` |
| `energy_objective()` | `Σ wᵢ²` |
| `smoothness_objective()` | `Σ (wᵢ₊₁ − wᵢ)²` |
| `peak_amplitude_objective()` | `max |wᵢ|` |

For a custom objective, supply `(name, evaluator, ∇evaluator?)` tuples.

## Result structure

```julia
result.pareto_front       # Vector of MultiObjectiveSolution
result.pareto_objectives  # n_solutions × n_objectives
result.history
```

Each `MultiObjectiveSolution` carries the controls and objective values; you
choose the trade-off you want.

## Visualization

```julia
using Plots
plot_pareto_front(result)
```

(Requires the `Plots` extension — `pkg> add Plots`.)

## When to use multi-objective

- **Hardware budget**: comparing fidelity vs energy vs amplitude trade-offs
- **Reporting**: picking a representative pulse for a paper figure
- **Initial scoping**: when you don't yet know reasonable penalty weights —
  a Pareto front gives weights for free
