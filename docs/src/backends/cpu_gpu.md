# CPU / GPU backends

PULSAR's compute kernels target three hardware backends:

| Backend | Module | Activation |
|---|---|---|
| CPU | `CPUBackend` | Always available |
| CUDA | `CUDABackend` | `pkg> add CUDA` |
| Metal | `MetalBackend` | `pkg> add Metal` (Apple Silicon) |

Sources:
[`src/Backend/Hardware/CPUBackend.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Backend/Hardware/CPUBackend.jl),
[`CUDABackend.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Backend/Hardware/CUDABackend.jl),
[`MetalBackend.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Backend/Hardware/MetalBackend.jl).

## Constructors

```julia
cpu  = cpu_backend()
cuda = CUDABackend()        # falls back to CPU with @warn if CUDA unavailable
mtl  = MetalBackend()       # falls back to CPU with @warn if Metal unavailable
```

## Selecting a backend

The device registry tracks the active backend per scope:

| Function | Effect |
|---|---|
| `set_device!(backend)` | Set global default |
| `get_device()` | Query current device |
| `available_devices()` | Vector of all detected backends |
| `with_device(backend) do ... end` | Scoped override |

```julia
with_device(CUDABackend()) do
    result = grape_optimize(sys, target, ctrl; config=config)
end
# global device unchanged here
```

## Hybrid execution

For large sweeps with heterogeneous workload (some members CPU-cheap, some
better on GPU), `HybridExecutionPlanner` distributes work across available
backends.

| Function | Source | Use |
|---|---|---|
| `plan_hybrid_execution(work, backends)` | [`HybridExecution.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Backend/Scheduling/HybridExecution.jl) | Build a plan |
| `adaptive_backend_selection(...)` | same | Pick best backend per item |
| `estimate_operation_time(op, backend)` | same | Cost model |

## Graceful fallback

If a GPU backend is requested but the package isn't available (or the runtime
hasn't loaded it), PULSAR falls back to CPU and emits a `@warn`. This is
controlled by the `_CUDA_LOADED` and `_METAL_LOADED` `Ref{Bool}`s in
[`Backend/Scheduling/DeviceRegistry.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/Backend/Scheduling/DeviceRegistry.jl).

## Practical guidance

- **Single small problem (dim ≤ 8, n_steps ≤ 200)**: CPU is faster than
  GPU due to launch overhead.
- **Ensemble / batch sweep**: GPU shines for `n_members × n_steps × dim²`
  large enough to saturate the device.
- **MAS powder averaging**: tile the powder grid across GPU threads
  (`compute_grape_gradient_powder`).
- **Apple Silicon laptops**: `MetalBackend` is competitive with CPU for
  medium problem sizes and offers significant power savings.
