# Parallelism

PULSAR exposes two orthogonal CPU parallelism strategies, defined in
[`src/Backend/Parallelism/CPUParallelization.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Backend/Parallelism/CPUParallelization.jl).

## Task-level parallelism

`TaskParallelizationStrategy` distributes independent work units across Julia
threads or distributed workers. Use this for:

- **Ensemble fidelity** — each ensemble member is independent
- **Powder averaging** — each Euler triple is independent
- **Batch optimization** — each restart / parameter sweep is independent

Best when each unit performs ≥ a few milliseconds of work, so thread-spawn
overhead is amortized.

## Vectorization

`VectorizationStrategy` reorganizes inner loops for SIMD via Julia's
`@simd` / `@inbounds` directives in performance-critical kernels (forward
propagator, gradient accumulation). This is automatic and requires no user
configuration.

## Gradient parallelization

`GradientParallelization` parallelizes the GRAPE gradient computation itself
across time slices when the per-slice work is large (high-dimensional
systems, many controls).

## Choosing how many threads

Start Julia with `julia --threads=N` (or `JULIA_NUM_THREADS=N`). Inside,
`Threads.nthreads()` reports the count.

```bash
julia --threads=auto --project=. my_pulse_design.jl
```

For ensemble GRAPE the speedup is approximately linear up to `N = n_ensemble`.

## Distributed parallelism

For multi-node sweeps:

```julia
using Distributed
addprocs(8)
@everywhere using PULSAR
# … then drive ensemble work across workers
```

The hybrid planner (`plan_hybrid_execution`) understands distributed workers
when CUDA / Metal devices are also present.

## Performance monitoring

`PerformanceMonitor` ([`Runtime/PerformanceMonitoring.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Runtime/PerformanceMonitoring.jl)) records per-iteration
wall time, gradient norm, fidelity, and memory usage. Wire it through the
`callback` keyword on any optimizer:

```julia
mon = PerformanceMonitor()
config = GRAPEConfig(... , callback = (i, F; kw...) -> record_iteration!(mon, i, F))
result = grape_optimize(sys, target, ctrl; config=config)
print_progress(mon)
```
