# API — Runtime

Performance monitoring, iteration callbacks, and algorithm selection.
Source: [`src/Runtime/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Runtime).

## Performance monitoring

```@docs
PerformanceMonitor
record_iteration!
get_summary
print_progress
detect_stagnation
get_memory_usage_mb
```

## Iteration callback

```@docs
IterationCallback
iteration_callback
```

## Algorithm selection

```@docs
recommend_optimizer
auto_optimize
describe_recommendation
AlgorithmRecommendation
```
