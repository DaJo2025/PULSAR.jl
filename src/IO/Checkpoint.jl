# ============================================================
# Pulsar.jl — Checkpoint and Resume
# Pulse Design Library for Spin Control Algorithms and Rollout
# ============================================================
#
# Save / load checkpointing for long optimization runs so they can
# be interrupted and resumed without losing progress.
#
# A single unified `Checkpoint <: AbstractCheckpoint` type subsumes
# warm-start (MR / QC) and resume-state payloads.  Storage is Julia
# `Serialization` (.jls) only.
#
# `MRCheckpoint` and `QCCheckpoint` are retained as private structs
# (not exported) so existing `.jls` files written by earlier Pulsar
# versions still deserialise; `load_checkpoint` auto-converts them
# to `Checkpoint` on read.
# ============================================================

using Dates, LinearAlgebra, Printf, Serialization

# ──────────────────────────────────────────────────────────────
# Type hierarchy
# ──────────────────────────────────────────────────────────────

"""
    AbstractCheckpoint

Supertype for Pulsar checkpoint snapshots.  The concrete subtype is
[`Checkpoint`](@ref).
"""
abstract type AbstractCheckpoint end

"""
    Checkpoint <: AbstractCheckpoint

Unified pulse-design checkpoint snapshot.  Carries enough state for
both warm-start (the dominant MR/QC use case) and full resume.

# Fields

Warm-start core:
- `w_opt`        — best waveform `[n_controls × n_timesteps]`
- `F_opt`        — best fidelity at time of save
- `n_controls`   — number of control channels
- `n_timesteps`  — number of time steps

Optional resume payload (defaults: 0 / empty / `:generic`):
- `iteration`             — iteration index when saved (0 if none)
- `fidelity_history`      — fidelity trace up to `iteration`
- `gradient_norm_history` — gradient-norm trace up to `iteration`
- `optimizer_state`       — algorithm-specific dict (e.g. BFGS Hessian)
- `system_dim`            — Hilbert-space dim (for resume validation)

Domain tags (informational):
- `domain`        — `:mr`, `:qc`, or `:generic`
- `drive_max_hz`  — peak drive amplitude (Hz)
- `T_pulse`       — total pulse duration (s)
- `system_kind`   — platform symbol (`:transmon`, `:nmr_1H`, …)

Common:
- `timestamp`     — ISO 8601 string when written
- `metadata`      — free-form user dict
"""
struct Checkpoint <: AbstractCheckpoint
    w_opt                 :: Matrix{Float64}
    F_opt                 :: Float64
    n_controls            :: Int
    n_timesteps           :: Int
    iteration             :: Int
    fidelity_history      :: Vector{Float64}
    gradient_norm_history :: Vector{Float64}
    optimizer_state       :: Dict{String,Any}
    system_dim            :: Int
    domain                :: Symbol
    drive_max_hz          :: Float64
    T_pulse               :: Float64
    system_kind           :: Symbol
    timestamp             :: String
    metadata              :: Dict{String,Any}
end

"""
    Checkpoint(w_opt, F_opt, n_controls, n_timesteps; kwargs...) -> Checkpoint

Warm-start-friendly outer constructor.  All resume / domain fields
default to empty / 0 / `:generic` so a minimal call records just the
waveform + fidelity.
"""
function Checkpoint(
    w_opt::AbstractMatrix, F_opt::Real,
    n_controls::Integer, n_timesteps::Integer;
    iteration::Integer            = 0,
    fidelity_history              = Float64[],
    gradient_norm_history         = Float64[],
    optimizer_state               = Dict{String,Any}(),
    system_dim::Integer           = 0,
    domain::Symbol                = :generic,
    drive_max_hz::Real            = 0.0,
    T_pulse::Real                 = 0.0,
    system_kind::Symbol           = :generic,
    timestamp::AbstractString     = string(now()),
    metadata                      = Dict{String,Any}(),
)
    return Checkpoint(
        Matrix{Float64}(w_opt),
        Float64(F_opt),
        Int(n_controls),
        Int(n_timesteps),
        Int(iteration),
        Float64.(collect(fidelity_history)),
        Float64.(collect(gradient_norm_history)),
        Dict{String,Any}(string(k) => v for (k, v) in optimizer_state),
        Int(system_dim),
        domain,
        Float64(drive_max_hz),
        Float64(T_pulse),
        system_kind,
        String(timestamp),
        Dict{String,Any}(string(k) => v for (k, v) in metadata),
    )
end

# ──────────────────────────────────────────────────────────────
# Construction helper
# ──────────────────────────────────────────────────────────────

"""
    create_checkpoint(iter, controls, fidelity, fidelity_history,
                       gradient_norms, optimizer_state, system,
                       metadata=Dict()) -> Checkpoint

Build a [`Checkpoint`](@ref) from the current optimizer state.
"""
function create_checkpoint(
    iter::Int,
    controls::Matrix{Float64},
    fidelity::Float64,
    fidelity_history::Vector{Float64},
    gradient_norms::Vector{Float64},
    optimizer_state::Dict{String,Any},
    system::AbstractQuantumSystem,
    metadata::Dict{String,Any} = Dict{String,Any}(),
)::Checkpoint
    return Checkpoint(
        copy(controls), fidelity, system.n_controls, size(controls, 2);
        iteration             = iter,
        fidelity_history      = copy(fidelity_history),
        gradient_norm_history = copy(gradient_norms),
        optimizer_state       = optimizer_state,
        system_dim            = system.dim,
        timestamp             = string(now()),
        metadata              = metadata,
    )
end

# ──────────────────────────────────────────────────────────────
# Save  (Julia Serialization, atomic temp + rename)
# ──────────────────────────────────────────────────────────────

"""
    save_checkpoint(filepath, ckpt::Checkpoint) -> nothing

Atomically write a [`Checkpoint`](@ref) to disk via Julia
`Serialization` (temp file + rename).
"""
function save_checkpoint(filepath::String, ckpt::Checkpoint)
    tmp = filepath * ".tmp"
    open(tmp, "w") do io
        serialize(io, ckpt)
    end
    mv(tmp, filepath; force = true)
    @info "Checkpoint saved: $filepath  (iter=$(ckpt.iteration), F=$(round(ckpt.F_opt, digits=6)))"
    return nothing
end

# ──────────────────────────────────────────────────────────────
# Load  (returns Checkpoint regardless of payload)
# ──────────────────────────────────────────────────────────────

"""
    load_checkpoint(filepath) -> Checkpoint

Load a checkpoint from disk via Julia `Serialization`.  If the
deserialised payload is a legacy `MRCheckpoint` or `QCCheckpoint`
(written by an earlier Pulsar release), it is auto-converted to a
[`Checkpoint`](@ref).
"""
function load_checkpoint(filepath::String)::Checkpoint
    isfile(filepath) ||
        throw(ArgumentError("Checkpoint file not found: $filepath"))
    raw = open(filepath, "r") do io
        deserialize(io)
    end
    if raw isa Checkpoint
        return raw
    elseif raw isa AbstractCheckpoint
        return Checkpoint(raw)
    else
        throw(ArgumentError(
            "File $filepath does not contain a Checkpoint or compatible " *
            "AbstractCheckpoint subtype (got $(typeof(raw)))."))
    end
end

# ──────────────────────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────────────────────

"""
    resume_optimization(checkpoint, system, target;
                         additional_iterations, optimizer, verbose)
        -> OptimizationResult

Resume an optimization run from a saved [`Checkpoint`](@ref).

# Validation
- When `checkpoint.system_dim > 0`, requires `system.dim == checkpoint.system_dim`.
- Always requires `system.n_controls == checkpoint.n_controls`.
"""
function resume_optimization(
    checkpoint::Checkpoint,
    system::AbstractQuantumSystem,
    target::QuantumTarget;
    additional_iterations::Int = 500,
    optimizer::Symbol          = :grape,
    verbose::Bool              = true,
)::OptimizationResult

    if checkpoint.system_dim > 0 && system.dim != checkpoint.system_dim
        throw(ArgumentError(
            "System dimension mismatch: checkpoint has dim=$(checkpoint.system_dim), " *
            "provided system has dim=$(system.dim)"))
    end
    if system.n_controls != checkpoint.n_controls
        throw(ArgumentError(
            "n_controls mismatch: checkpoint=$(checkpoint.n_controls), " *
            "system=$(system.n_controls)"))
    end

    if verbose
        @printf("Resuming from checkpoint (iter=%d, F=%.6f, saved %s)\n",
                checkpoint.iteration, checkpoint.F_opt, checkpoint.timestamp)
    end

    dt_val = get(checkpoint.metadata, "dt", nothing)
    if isnothing(dt_val)
        dt_val = 10e-9
        verbose && @warn "dt not stored in checkpoint; assuming dt=10 ns"
    end
    dt_use = Float64(dt_val)

    controls_cs = ControlSequence(
        copy(checkpoint.w_opt),
        dt_use,
        dt_use * checkpoint.n_timesteps,
        checkpoint.n_timesteps,
    )

    result = if optimizer == :lbfgs
        lbfgs_optimize(system, target, controls_cs;
                       config = LBFGSConfig(max_iter = additional_iterations,
                                             verbose = verbose))
    else
        grape_optimize(system, target, controls_cs;
                       config = GRAPEConfig(max_iter = additional_iterations,
                                             verbose = verbose))
    end

    full_history = vcat(checkpoint.fidelity_history,      result.fidelity_history)
    full_gnorm   = vcat(checkpoint.gradient_norm_history, result.gradient_norm_history)

    return OptimizationResult(
        result.controls,
        result.fidelity,
        full_history,
        full_gnorm,
        checkpoint.iteration + result.n_iterations,
        result.converged,
        result.termination_reason,
        result.total_time,
        result.n_fidelity_evaluations,
        result.n_gradient_evaluations,
        merge(result.metadata,
              Dict{String,Any}("resumed_from_iter" => checkpoint.iteration)),
    )
end

# ──────────────────────────────────────────────────────────────
# Auto-checkpoint callback
# ──────────────────────────────────────────────────────────────

"""
    auto_checkpoint_callback(filepath, interval) -> Function

Create a callback that saves a [`Checkpoint`](@ref) every `interval`
iterations.  The returned function has signature

    (iter, controls, fidelity, grad_norm, system) -> nothing
"""
function auto_checkpoint_callback(filepath::String, interval::Int)::Function
    return function callback(iter::Int, controls::Matrix{Float64},
                              fidelity::Float64, grad_norm::Float64,
                              system::AbstractQuantumSystem)
        if iter % interval == 0
            chk = create_checkpoint(
                iter, controls, fidelity,
                Float64[fidelity], Float64[grad_norm],
                Dict{String,Any}(), system,
                Dict{String,Any}("callback_interval" => interval),
            )
            save_checkpoint(filepath, chk)
        end
    end
end

# ──────────────────────────────────────────────────────────────
# Compatibility predicate
# ──────────────────────────────────────────────────────────────

"""
    checkpoint_compatible(ckpt::Checkpoint, n_controls, n_timesteps) -> Bool

Return `true` when the saved waveform matches the current problem
dimensions exactly.  Prints a descriptive message and returns `false`
on mismatch so callers can fall back to a default initialisation.
"""
function checkpoint_compatible(
    ckpt::Checkpoint, n_controls::Int, n_timesteps::Int,
)::Bool
    ok = (ckpt.n_controls == n_controls) && (ckpt.n_timesteps == n_timesteps)
    if !ok
        @printf("  [checkpoint] Dimension mismatch: checkpoint [%d × %d] ≠ current [%d × %d].\n",
                ckpt.n_controls, ckpt.n_timesteps, n_controls, n_timesteps)
        println("               Falling back to default warm start.")
    end
    return ok
end

# ──────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────

"""
    list_checkpoints(directory) -> Vector{String}

Return all `.jls` checkpoint files in `directory`, sorted by filename.
"""
function list_checkpoints(directory::String)::Vector{String}
    isdir(directory) || return String[]
    files = filter(f -> endswith(f, ".jls"), readdir(directory))
    return sort([joinpath(directory, f) for f in files])
end

"""
    checkpoint_summary(filepath) -> String

Return a one-line summary of a checkpoint file.
"""
function checkpoint_summary(filepath::String)::String
    isfile(filepath) || return "File not found: $filepath"
    try
        chk = load_checkpoint(filepath)
        return @sprintf("%s | iter=%d | F=%.6f | saved %s",
                        basename(filepath), chk.iteration,
                        chk.F_opt, chk.timestamp)
    catch e
        return "Could not read checkpoint: $e"
    end
end

# ──────────────────────────────────────────────────────────────
# Legacy structs (private) — kept so on-disk `.jls` files written
# by earlier releases still deserialise.  Not exported.
# ──────────────────────────────────────────────────────────────

struct MRCheckpoint <: AbstractCheckpoint
    w_opt      :: Matrix{Float64}
    F_opt      :: Float64
    N_TS       :: Int
    N_CTRL     :: Int
    RF_MAX_HZ  :: Float64
    T_PULSE    :: Float64
    timestamp  :: String
    metadata   :: Dict{String,Any}
end

struct QCCheckpoint <: AbstractCheckpoint
    w_opt        :: Matrix{Float64}
    F_opt        :: Float64
    n_timesteps  :: Int
    n_controls   :: Int
    drive_max_hz :: Float64
    T_pulse      :: Float64
    system_kind  :: Symbol
    timestamp    :: String
    metadata     :: Dict{String,Any}
end

Checkpoint(c::MRCheckpoint) = Checkpoint(
    c.w_opt, c.F_opt, c.N_CTRL, c.N_TS;
    domain       = :mr,
    drive_max_hz = c.RF_MAX_HZ,
    T_pulse      = c.T_PULSE,
    timestamp    = c.timestamp,
    metadata     = c.metadata,
)

Checkpoint(c::QCCheckpoint) = Checkpoint(
    c.w_opt, c.F_opt, c.n_controls, c.n_timesteps;
    domain       = :qc,
    drive_max_hz = c.drive_max_hz,
    T_pulse      = c.T_pulse,
    system_kind  = c.system_kind,
    timestamp    = c.timestamp,
    metadata     = c.metadata,
)
