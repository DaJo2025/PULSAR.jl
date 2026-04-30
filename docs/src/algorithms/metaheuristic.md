# Metaheuristic methods

Global / population-based stochastic optimizers, useful when the objective is
multi-modal or the gradient is unreliable. All metaheuristic optimizers live
under [`src/Optimization/Metaheur/`](https://github.com/DaJo2025/PULSAR.jl/tree/main/src/Optimization/Metaheur).

## Population-based

| Function | Algorithm | Source |
|---|---|---|
| `ga_optimize` | Genetic algorithm | `GA.jl` |
| `pso_optimize` | Particle swarm | `Swarm.jl` |
| `de_optimize` | Differential evolution | `Swarm.jl` |
| `cmaes_optimize` | CMA-ES (covariance-matrix adaptation) | `CMAES.jl` |
| `pscmaes_optimize` | Parallel-restart CMA-ES | `CMAES.jl` |

## Annealing / Monte-Carlo

| Function | Algorithm | Source |
|---|---|---|
| `sa_optimize` | Simulated annealing | `SA.jl` |
| `mcsa_optimize` | Markov-chain SA | `SA.jl` |
| `ssmc_optimize` | Sequential-state Monte-Carlo | `SA.jl` |
| `mc_random_search` | Pure Monte-Carlo / uniform | `MC.jl` |
| `grid_search` | Cartesian grid sweep | `MC.jl` |

## Basin hopping

`basin_hopping_optimize` (in `BasinHopping.jl`) chains a global random-step
exploration with a deterministic local optimizer (default: L-BFGS). Excellent
for finding multiple high-quality solutions across distinct basins.

## QOC-domain dispatchers

| Function | Source |
|---|---|
| `cmaes_optimize`, `pso_optimize`, `nelder_mead_optimize` (QOC signatures) | [`src/Optimization/DirectSearchMethods.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Optimization/DirectSearchMethods.jl) |

These take `(sys, target, ctrl)` directly and internally call the generic
optimizer with the right objective wrapper.

## Choosing among them

- **CMA-ES**: very strong default for ≤ ~500 parameters; self-adapts step
  sizes
- **DE**: good for noisy objectives, slightly easier to tune than GA
- **PSO**: fast initial progress, less robust late
- **GA**: high diversity, expensive
- **Basin hopping**: when you suspect many local optima
- **Simulated annealing**: classic baseline; rarely beats CMA-ES in practice

## Tuning tips

- Population size scales as `O(n^{0.5})` to `O(n)` of the parameter count.
  CMA-ES default is `4 + ⌊3 ln n⌋`.
- Always re-evaluate the final fidelity with `compute_fidelity` — metaheuristic
  optimizers track noisy estimates internally and may overstate the result.
- Wrap with `Threads.@threads` over independent restarts when you have
  multiple cores.

## Example

```julia
config = CMAESConfig(
    max_iter        = 500,
    population_size = 30,
    sigma_init      = 0.3,
    seed            = 42,
)
result = cmaes_optimize(sys, target, ctrl; config=config)

# Independent re-simulation
ctrl_opt = ControlSequence(Matrix(result.controls'), ctrl.dt, ctrl.n_steps)
F_check  = compute_fidelity(sys, ctrl_opt, target)
```
