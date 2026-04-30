# ============================================================================
# Backend/DeviceRegistry.jl
# Package-level compute device registry — thread-safe via task-local storage.
#
# Usage:
#   PULSAR.set_device!(:cpu)         # set default for this task (and new tasks)
#   PULSAR.get_device()              # returns device for current task
#   PULSAR.available_devices()       # returns list of currently usable devices
#   PULSAR.with_device(:metal) do    # scoped override — restores previous on exit
#       grape_optimize(...)
#   end
# ============================================================================

# Global default device — used when a task has no task-local override.
# All writes go through set_device!; reads go through get_device().
const _DEVICE_DEFAULT = Ref{Symbol}(:cpu)

# Task-local storage key
const _DEVICE_KEY = :pulsar_device

# ============================================================================
# Core API
# ============================================================================

"""
    PULSAR.set_device!(device::Symbol)

Set the compute device for the **current task** and update the global default
so that newly spawned tasks also inherit this choice.

Accepted values:
- `:cpu`   — Julia multi-threading (`Threads.@threads`). Default.
             Scale with `julia -t N` or `JULIA_NUM_THREADS=N`.
- `:cuda`  — NVIDIA GPU via CUDA.jl.
             Requires `import CUDA` **before** `using PULSAR`.
- `:metal` — Apple Silicon GPU via Metal.jl.
             Requires `import Metal` **before** `using PULSAR`.

If the requested GPU backend is not loaded, emits a warning and keeps the
current device unchanged.

# Example
```julia
import Metal
using PULSAR

PULSAR.set_device!(:metal)
PULSAR.available_devices()   # → [:cpu, :metal]
```
"""
function set_device!(device::Symbol)
    device ∈ (:cpu, :cuda, :metal) ||
        throw(ArgumentError(
            "Unknown device :$device. Choose :cpu, :cuda, or :metal."))
    if device === :cuda && !_CUDA_LOADED[]
        @warn "set_device!(:cuda): CUDA.jl is not loaded. " *
              "Add `import CUDA` before `using PULSAR` to enable GPU support. " *
              "Keeping current device :$(get_device())."
        return nothing
    end
    if device === :metal && !_METAL_LOADED[]
        @warn "set_device!(:metal): Metal.jl is not loaded. " *
              "Add `import Metal` before `using PULSAR` to enable GPU support. " *
              "Keeping current device :$(get_device())."
        return nothing
    end
    # Update both the task-local value and the global default
    task_local_storage(_DEVICE_KEY, device)
    _DEVICE_DEFAULT[] = device
    return nothing
end

"""
    PULSAR.get_device() → Symbol

Return the compute device for the **current task**.

Falls back to the global default (`_DEVICE_DEFAULT`) when the task has no
task-local override.  This means tasks spawned before `set_device!` was called
see the previous setting, and newly spawned tasks inherit the most recent
global default.

Thread-safe: each task has its own independent storage slot.
"""
function get_device()::Symbol
    return get!(task_local_storage(), _DEVICE_KEY, _DEVICE_DEFAULT[])::Symbol
end

"""
    PULSAR.available_devices() → Vector{Symbol}

Return the list of devices that are currently usable.

Always includes `:cpu`. GPU devices appear only when their package
(`CUDA.jl` or `Metal.jl`) was loaded before `using PULSAR`.

# Example
```julia
import Metal; using PULSAR
PULSAR.available_devices()  # → [:cpu, :metal]
```
"""
function available_devices()::Vector{Symbol}
    devs = Symbol[:cpu]
    _CUDA_LOADED[]  && push!(devs, :cuda)
    _METAL_LOADED[] && push!(devs, :metal)
    return devs
end

# ============================================================================
# Scoped device override
# ============================================================================

"""
    PULSAR.with_device(f, device::Symbol)

Execute `f()` with the compute device temporarily overridden to `device` for
the **current task**, then restore the previous device on exit (even if `f`
throws).

This is the recommended way to run a section of code on a specific device
without permanently changing the global setting.  It is also composable:
nested `with_device` calls correctly restore the outer device.

# Arguments
- `f`      — zero-argument callable
- `device` — `:cpu`, `:cuda`, or `:metal`

# Example
```julia
result_cpu = with_device(:cpu) do
    grape_optimize(sys, tgt, ctrl_init)
end

result_gpu = with_device(:metal) do
    grape_optimize(sys, tgt, ctrl_init)
end
```
"""
function with_device(f, device::Symbol)
    device ∈ (:cpu, :cuda, :metal) ||
        throw(ArgumentError(
            "Unknown device :$device. Choose :cpu, :cuda, or :metal."))
    prev = get_device()
    task_local_storage(_DEVICE_KEY, device)
    try
        return f()
    finally
        task_local_storage(_DEVICE_KEY, prev)
    end
end
