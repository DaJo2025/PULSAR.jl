# Checkpointing

Pulsar uses a **unified** checkpoint format. The single `Checkpoint` type
(subtype of `AbstractCheckpoint`) covers MR, QC, and generic optimizations,
and all file I/O goes through one set of save/load functions.

Source: [`src/IO/Checkpoint.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/IO/Checkpoint.jl).

## File format

- **Encoding**: Julia `Serialization` with the `.jls` extension
- **Atomicity**: writes go to a temporary file then `mv` to the final path
- **Compatibility**: layout is version-stable for v0.1.x

## Saving a checkpoint

```julia
save_checkpoint(
    "my_run.jls",
    Checkpoint(
        w_opt, F_opt, n_controls, n_timesteps;
        domain       = :mr,                    # or :qc / :generic
        drive_max_hz = RF_MAX_HZ,
        T_pulse      = T_PULSE,
        metadata     = Dict("note" => "broadband 180°, ±6 kHz, 21 isochromats"),
    ),
)
```

## Loading a checkpoint

```julia
ckpt = load_checkpoint("my_run.jls")     # → Checkpoint

ckpt.w_opt
ckpt.F_opt
ckpt.n_controls
ckpt.n_timesteps
ckpt.iteration
ckpt.fidelity_history
ckpt.gradient_norm_history
ckpt.optimizer_state
ckpt.domain
ckpt.drive_max_hz
ckpt.T_pulse
ckpt.system_kind
ckpt.timestamp
ckpt.metadata
```

## Compatibility check

Before resuming, verify dimensions match the current problem:

```julia
if checkpoint_compatible(ckpt, n_controls, n_timesteps)
    w_init = ckpt.w_opt
else
    w_init = random_initialization(n_controls, n_timesteps)
end
```

## Resuming an optimization

```julia
result = resume_optimization(ckpt, sys, target;
                              config = GRAPEConfig(...))
```

This restores the optimizer state (history, momentum, line-search state, …)
where supported, otherwise hot-starts from `ckpt.w_opt`.

## Auto-checkpointing during a run

```julia
cb     = auto_checkpoint_callback("my_run.jls", every_n_iters=50)
config = GRAPEConfig(... , callback=cb)
result = grape_optimize(sys, target, ctrl; config=config)
```

For MR-layer optimizations (which use `MRControl` rather than the generic
optimizer), wire the callback through `MRControl`:

```julia
cb = (iter, F; grad=NaN, evals=0) -> begin
    iter % 50 == 0 && save_checkpoint(
        "buss.jls",
        Checkpoint(w_ref[], F, N_CTRL, N_TS;
                   domain=:mr, drive_max_hz=RF_MAX_HZ, T_pulse=T_PULSE),
    )
end
ctrl   = MRControl(..., callback=cb)
result = optimcon(ctrl, guess)
```

## Three-tier warm-start

A common production pattern:

1. **Checkpoint** (`load_checkpoint` + `checkpoint_compatible`)
2. **Exported pulse file** (custom Bruker / CSV reader)
3. **Random seed** fallback

This keeps long-running optimizations resilient to crashes while remaining
runnable on a fresh machine without prior state.

## Inspection helpers

| Function | Purpose |
|---|---|
| `list_checkpoints(dir)` | Enumerate `.jls` files in a directory |
| `checkpoint_summary(ckpt)` | One-line description (timestamp, F_opt, dims) |
| `create_checkpoint(...)` | Lower-level constructor |
