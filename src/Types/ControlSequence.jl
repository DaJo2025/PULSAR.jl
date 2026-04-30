# Types/ControlSequence.jl
# ControlSequence type and constructor functions.
# Extracted from Core/QuantumSystem.jl.

using Random

# ============================================================================
# Control sequence type
# ============================================================================

"""
    ControlSequence

Piecewise-constant control amplitudes for a quantum control experiment.

# Fields
- `amplitudes :: Matrix{Float64}` — control amplitudes with size
  `[n_steps × n_controls]`; `amplitudes[k, j]` is the amplitude of control j
  during the k-th time slice (rad/s if multiplied into H_controls which are in rad/s).
  Julia column-major layout: inner loop over controls (j) is stride-1for fixed k.
- `dt         :: Float64` — duration of each time slice (seconds, > 0)
- `n_steps    :: Int`  — number of discrete time slices

# Backward-compatible properties
The following computed properties are available for code written against the
previous API and will forward to the new fields:
- `controls`     — `[n_controls × n_steps]` adjoint view of `amplitudes`
- `n_timesteps`  — alias for `n_steps`
- `total_time`   — `dt * n_steps` (seconds)

# Convention
Control amplitudes u_j[k] are in units that, when multiplied by the corresponding
H_controls[j] (in rad/s), give a Hamiltonian in rad/s.  For NMR the conventional
amplitude unit is Hz, and H_controls carries a factor of 2π so that the product
has the correct dimension.
"""
struct ControlSequence
    amplitudes  :: Matrix{Float64}
    dt          :: Float64
    n_steps     :: Int

    # Inner constructor: validates shape only.  Numeric checks on `dt` and
    # `n_steps` are deferred to `validate_controls` so the user can construct
    # a ControlSequence with placeholder values for inspection/repair.
    function ControlSequence(amplitudes::Matrix{Float64}, dt::Float64, n_steps::Int)
        if dt < 0.0
            throw(ArgumentError("dt must be non-negative, got $dt"))
        end
        if n_steps < 0
            throw(ArgumentError("n_steps must be non-negative, got $n_steps"))
        end
        if size(amplitudes, 1) != n_steps
            throw(DimensionMismatch(
                "amplitudes has $(size(amplitudes, 1)) rows but n_steps = $n_steps; " *
                "expected layout [n_steps × n_controls]"))
        end
        new(amplitudes, dt, n_steps)
    end
end

# ============================================================================
# Backward-compatible 4-argument outer constructor
# ============================================================================

"""
    ControlSequence(controls, dt, total_time, n_timesteps) -> ControlSequence

Backward-compatible constructor that accepts a `[n_controls × n_timesteps]`
matrix (old layout) and automatically transposes it to the new
`[n_timesteps × n_controls]` internal layout.

This form is recognised by four positional arguments where the third is
`total_time::Float64` (ignored; `total_time = dt * n_timesteps` is used).
All existing code that constructs `ControlSequence(mat, dt, total_time, n_t)`
continues to work without modification.
"""
function ControlSequence(controls::Matrix{Float64},
                          dt::Float64,
                          total_time::Float64,
                          n_timesteps::Int)
    n_c, n_t = size(controls)
    if n_t != n_timesteps
        throw(DimensionMismatch(
            "controls has $n_t columns but n_timesteps = $n_timesteps; " *
            "expected [n_controls × n_timesteps] layout"))
    end
    amplitudes = Matrix(controls')   # transpose: [n_c × n_t] → [n_t × n_c]
    return ControlSequence(amplitudes, dt, n_timesteps)
end

# ============================================================================
# Backward-compatible property access
# ============================================================================

"""
Provide backward-compatible property aliases so that existing code using
`ctrl.controls`, `ctrl.n_timesteps`, and `ctrl.total_time` continues to work.
"""
function Base.getproperty(ctrl::ControlSequence, sym::Symbol)
    if sym === :controls
        # [n_controls × n_steps] view — backward-compatible read access
        return getfield(ctrl, :amplitudes)'
    elseif sym === :n_timesteps
        return getfield(ctrl, :n_steps)
    elseif sym === :total_time
        return getfield(ctrl, :dt) * getfield(ctrl, :n_steps)
    else
        return getfield(ctrl, sym)
    end
end

Base.propertynames(::ControlSequence, private::Bool=false) =
    (:amplitudes, :dt, :n_steps, :controls, :n_timesteps, :total_time)

# ============================================================================
# Control sequence factories
# ============================================================================

"""
    random_controls(system::AbstractQuantumSystem, total_time::Float64,
                    n_timesteps::Int; amplitude::Float64=1.0,
                    rng::AbstractRNG=Random.GLOBAL_RNG) -> ControlSequence

Generate a `ControlSequence` with uniformly distributed random amplitudes.

# Arguments
- `system`      — quantum system (used to determine `n_controls`)
- `total_time`  — total pulse duration in seconds (> 0)
- `n_timesteps` — number of discrete time steps (> 0)
- `amplitude`   — half-range of uniform distribution; amplitudes drawn from
  `Uniform(-amplitude, +amplitude)`
- `rng`         — random number generator (default: `Random.GLOBAL_RNG`)

# Returns
A `ControlSequence` with `amplitudes` drawn from U(-amplitude, +amplitude).

# Example
```julia
seq = random_controls(sys, 1e-3, 100; amplitude=2π*1000.0)
```
"""
function random_controls(system::AbstractQuantumSystem, total_time::Real,
                         n_timesteps::Int;
                         amplitude::Real=1.0,
                         rng::AbstractRNG=Random.GLOBAL_RNG)::ControlSequence
    total_time = Float64(total_time)
    amplitude  = Float64(amplitude)
    if total_time <= 0
        throw(ArgumentError("total_time must be positive, got $total_time"))
    end
    if n_timesteps <= 0
        throw(ArgumentError("n_timesteps must be positive, got $n_timesteps"))
    end
    dt = total_time / n_timesteps
    # Allocate [n_steps × n_controls] — cache-friendly for inner j (control) loop
    amplitudes = amplitude .* (2 .* rand(rng, n_timesteps, system.n_controls) .- 1)
    return ControlSequence(amplitudes, dt, n_timesteps)
end

"""
    zero_controls(system::AbstractQuantumSystem, total_time::Float64,
                  n_timesteps::Int) -> ControlSequence

Generate a `ControlSequence` with all control amplitudes set to zero.

# Arguments
- `system`      — quantum system (used to determine `n_controls`)
- `total_time`  — total pulse duration in seconds (> 0)
- `n_timesteps` — number of discrete time steps (> 0)

# Returns
A `ControlSequence` with `amplitudes` identically zero.

# Notes
Zero controls correspond to free evolution under H_drift alone.  This is
often a useful starting point for optimization when the drift already produces
the desired gate to first order.

# Example
```julia
seq = zero_controls(sys, 1e-3, 100)
```
"""
function zero_controls(system::AbstractQuantumSystem, total_time::Real,
                       n_timesteps::Int)::ControlSequence
    total_time = Float64(total_time)
    if total_time <= 0
        throw(ArgumentError("total_time must be positive, got $total_time"))
    end
    if n_timesteps <= 0
        throw(ArgumentError("n_timesteps must be positive, got $n_timesteps"))
    end
    dt = total_time / n_timesteps
    # Allocate [n_steps × n_controls]
    amplitudes = zeros(Float64, n_timesteps, system.n_controls)
    return ControlSequence(amplitudes, dt, n_timesteps)
end

# ============================================================================
# Utility: pretty printing
# ============================================================================

function Base.show(io::IO, seq::ControlSequence)
    n_c = size(seq.amplitudes, 2)
    print(io, "ControlSequence(n_controls=$n_c, " *
              "n_steps=$(seq.n_steps), total_time=$(seq.total_time) s, " *
              "dt=$(seq.dt) s)")
end
