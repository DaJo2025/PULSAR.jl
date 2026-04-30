# Defining your own problem

Benchmark problem definitions are not bundled with the public release.
You supply your own `BenchmarkProblem` instances and either drop them into
`comparisons/Problems/all_problems.jl` (which `run_comparisons.jl` picks
up automatically) or `include` them from a script of your own.

## The `BenchmarkProblem` struct

The shipped drivers reference five fields on `problem`:
`id`, `description`, `ctrl`, `guess_seed`, and (optionally) `time_limit_s`
/ `target_fidelity`. A minimal definition:

```julia
struct BenchmarkProblem
    id              :: String
    description     :: String
    sys             :: Any            # QubitSystem / MRSpinSystem / TransmonSystem / …
    target          :: QuantumTarget
    ctrl            :: Any            # ControlSequence or MRControl / LindbladMRControl
    guess_seed      :: Int
    target_fidelity :: Float64        # used by the report for the "target ≥ X" header
    time_limit_s    :: Float64        # advisory budget passed to external solvers
end
```

Place the definition in `comparisons/Problems/problem_types.jl` and load
it from `comparisons/Problems/all_problems.jl` alongside your problem
constructors. `run_comparisons.jl` already includes that file
conditionally, so as soon as it exists the problems become visible.

## A complete example: broadband ¹H 180°

```julia
using Pulsar

function bm_broadband_180()
    sys = mr_system(spins              = ["1H"],
                    chemical_shifts_hz = collect(range(-6e3, 6e3; length=21)),
                    b0_tesla           = 14.1)

    target = state_target(spin_state(sys, [-1.0]);
                           psi_init = spin_state(sys, [+1.0]))

    ctrl = MRControl(sys; n_steps      = 400,
                          total_time_s = 500e-6,
                          drive_max_hz = 20e3)

    return BenchmarkProblem(
        id              = "P_BB180",
        description     = "Broadband 180° inversion, ±6 kHz, 500 µs",
        sys             = sys,
        target          = target,
        ctrl            = ctrl,
        guess_seed      = 2026,
        target_fidelity = 0.99,
        time_limit_s    = 120.0,
    )
end
```

## Registering the problem

Build an `ALL_PROBLEMS` vector in `comparisons/Problems/all_problems.jl`:

```julia
include("problem_types.jl")
include("BM_broadband_180.jl")
include("BM_inept.jl")

const ALL_PROBLEMS = BenchmarkProblem[
    bm_broadband_180(),
    bm_inept(),
]
```

`run_comparisons.jl` includes this file automatically if it exists. The
`--problems` CLI flag then accepts the IDs you defined:

```bash
julia --project=. comparisons/run_comparisons.jl --problems P_BB180
```

## What `ctrl` should be

The driver dispatches on the type of `problem.ctrl`:

| `ctrl` type | Domain | Drivers that accept it |
|---|---|---|
| `ControlSequence`    | Generic / QC closed system | All Pulsar drivers, QuantumControl, Krotov, QuTiP, qopt |
| `MRControl`          | NMR / EPR / DNP closed system | Pulsar (MR layer), Spinach, SIMPSON |
| `LindbladMRControl`  | NMR / EPR open system (T₁/T₂) | Pulsar Lindblad path; drivers that lack open-system support return an error |
| `MRIControlSequence` | Bloch / MRI | Pulsar Bloch propagator |

A driver that does not support the supplied `ctrl` type returns an
`error_result` with a clear message rather than silently producing a
closed-system answer.

## Open-system problems

For Lindblad benchmarks, build a `LindbladMRControl` and provide jump
operators alongside the drift Hamiltonian:

```julia
ctrl = LindbladMRControl(sys; n_steps      = 200,
                              total_time_s = 1.0,
                              drive_max_hz = 5e3,
                              jump_ops     = [√(1/T1) * spin_op(sys, 1, :-)],
                              decay_rates  = [1.0])
```

The shared `grape_lindblad_kernel` is then used for canonical fidelity
re-evaluation across all drivers — see [Propagators](../theory/propagators.md)
for the Liouville-space convention.

## Reproducibility

`problem.guess_seed` is the only randomness in the framework. Every
driver builds its initial waveform from
`Random.MersenneTwister(problem.guess_seed)`, so two runs of the same
problem produce identical initial controls. Pin the seed once per
problem and you get bit-identical starting points across solvers, which
is what makes the per-iteration `fidelity_history` traces directly
comparable.

## Tips

- **Keep problems cheap.** Aim for individual driver runs in the
  10-second to 2-minute range. Long-running benchmarks make iteration
  on the framework painful and external-solver subprocesses can be hard
  to interrupt.
- **One physical effect per problem.** Resist bundling broadband + power
  cap + selective + open system into one `BenchmarkProblem`. Separate
  problems make the per-effect comparison legible in the summary table.
- **Set `target_fidelity` honestly.** It only affects the report header,
  but readers use it as a "did this solver actually solve the problem?"
  cutoff.
- **Use `metadata` on `BenchmarkResult`** to surface solver-specific
  diagnostics (wall-clock breakdown, gradient norms, line-search step
  count). The JSON output preserves it verbatim.
