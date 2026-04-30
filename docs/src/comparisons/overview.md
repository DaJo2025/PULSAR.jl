# Cross-package benchmark — overview

PULSAR ships a self-contained framework for benchmarking its optimizers
against other quantum optimal control packages on a shared problem
definition. The framework lives in
[`comparisons/`](https://github.com/DaJo2025/PULSAR.jl/tree/main/comparisons)
and is independent of the package itself — nothing in `src/PULSAR.jl`
depends on it, and you do not need any external solver installed to use
PULSAR.

## What is shipped

Only the framework code is distributed:

| Path | Contents |
|---|---|
| `comparisons/Drivers/` | One driver per external solver, plus the abstract interface |
| `comparisons/Translator/` | Cross-package physics translator (capabilities, emitters, subprocess helpers) |
| `comparisons/Report/` | Fixed-width result table and JSON writer |
| `comparisons/run_comparisons.jl` | CLI entry point |

Benchmark **problem definitions** are not bundled. You define your own
`BenchmarkProblem` instances and register them — see
[Defining your own problem](your_problem.md).

## How a run works

`run_comparisons.jl` performs four steps:

1. Includes the driver files and (optionally) `comparisons/Problems/all_problems.jl`
   if it exists in your local working tree.
2. For every `(driver, problem)` pair selected on the command line, calls
   `run_driver(driver, problem)` and collects a `BenchmarkResult`.
3. Prints a per-problem report and a cross-driver summary table.
4. Writes a timestamped JSON file to `comparisons/Results/`.

```bash
# All problems with all installed drivers
julia --project=. comparisons/run_comparisons.jl

# A subset
julia --project=. comparisons/run_comparisons.jl --problems P_BB180,P_INEPT
julia --project=. comparisons/run_comparisons.jl --packages PULSAR_lbfgs,Krotov
```

CLI flags:

| Flag | Meaning |
|---|---|
| `--problems`  | Comma-separated `BenchmarkProblem.id` values (default: all registered) |
| `--packages`  | Comma-separated driver keys from `ALL_DRIVERS` (default: all) |

## Fairness invariants

The framework is designed so that every solver is judged by the *same*
ruler:

- **Canonical fidelity re-evaluation.** Every driver's reported fidelity is
  recomputed through PULSAR's `grape_state_kernel` (closed-system) or
  `grape_lindblad_kernel` (open-system) after the external solver returns
  its waveform. The number that appears in the table is always PULSAR's
  re-simulation, never the external solver's self-report.
- **Same initial guess.** All drivers receive the same `problem.guess_seed`
  and use it to seed an `Random.MersenneTwister` for the random initial
  waveform.
- **Advisory time budget.** `problem.time_limit_s` is passed to the
  external solver where supported; not all packages honor it strictly.
- **Open-system problems** require a `LindbladMRControl`-typed `problem.ctrl`.
  Drivers that do not support open systems return an error rather than a
  silent (closed-system) result.

## Result type

Every driver returns a `BenchmarkResult` defined in
[`Drivers/driver_interface.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/comparisons/Drivers/driver_interface.jl):

```julia
struct BenchmarkResult
    driver_name       :: String           # e.g. "PULSAR/:lbfgs"
    problem_id        :: String           # matches BenchmarkProblem.id
    fidelity          :: Float64          # canonical re-evaluation
    wall_time_s       :: Float64
    n_iterations      :: Int
    converged         :: Bool
    controls          :: Union{Matrix{Float64}, Nothing}
    fidelity_history  :: Vector{Float64}
    available         :: Bool             # false ⇒ external package missing
    unavailable_msg   :: String           # install hint when !available
    metadata          :: Dict{String,Any} # driver-specific
end
```

Two helpers cover the failure paths:

- `not_available_result(driver_name, problem_id, msg)` — returned when the
  external toolchain is missing. Surfaces in the report as `NOT AVAILABLE`.
- `error_result(driver_name, problem_id, err)` — returned when the package
  was available but the run threw. Stores `string(err)` in `metadata["error"]`.

Drivers must never let an exception escape `run_driver` — wrap all external
calls in `try/catch` and return one of these on failure.

## Output

A run prints a per-problem block to the console and writes a JSON file
to `comparisons/Results/`:

```
══════════════════════════════════════════════════════════════════════
  P_BB180  Broadband 180° inversion, ±6 kHz, 500 µs  |  target ≥ 0.99
══════════════════════════════════════════════════════════════════════
Package/Method               Fidelity       Time    Iters  Status
──────────────────────────────────────────────────────────────────────
  PULSAR/:lbfgs              0.9987       1.24 s      312  converged
  PULSAR/:cmaes              0.9941       8.73 s      500  max_iter
  QuantumControl/GRAPE       0.9983       1.89 s      298  converged
  QuTiP/GRAPE          NOT AVAILABLE — install PythonCall + pip install qutip
──────────────────────────────────────────────────────────────────────
```

The JSON output preserves every field of every `BenchmarkResult`,
including `fidelity_history` and `metadata`, so downstream analysis
scripts have full per-iteration trajectories to work with.

## Where to go next

- [Drivers](drivers.md) — the `AbstractSolverDriver` interface and the
  full list of shipped drivers, including external dependencies.
- [Defining your own problem](your_problem.md) — how to write a
  `BenchmarkProblem` and register it for `run_comparisons.jl`.
