# PULSAR Cross-Package Optimal Control Benchmark

This directory contains a driver-based framework for benchmarking PULSAR
against other quantum optimal control packages on user-supplied pulse-design
problems.

> **What is shipped:** the framework code only — drivers, the cross-package
> translator, the report formatter, and the CLI entry point. Benchmark problem
> definitions are **not** distributed with the package; you supply your own
> `BenchmarkProblem` instances (see [Defining problems](#defining-problems)).
> Results are written to `comparisons/Results/` at runtime; that directory is
> `.gitignore`'d.

---

## Layout

```
comparisons/
  Drivers/             # One file per external solver
    driver_interface.jl       # AbstractSolverDriver, BenchmarkResult,
                              # not_available_result, error_result
    pulsar_driver.jl          # PULSAR (lbfgs, cmaes, grape, lbfgsb, cg)
    quantumcontrol_driver.jl  # QuantumControl.jl
    krotov_driver.jl          # Krotov.jl
    qutip_driver.jl           # QuTiP (Python via PythonCall)
    qopt_driver.jl            # qopt (Python via PythonCall)
    spinach_driver.jl         # Spinach (MATLAB)
    simpson_driver.jl         # SIMPSON (C subprocess)
    quandary_driver.jl        # Quandary (C++ subprocess)
  Translator/          # Cross-package physics translation
  Report/              # Result table + JSON export
  run_comparisons.jl   # CLI entry point
```

---

## Quick Start

You must define at least one `BenchmarkProblem` and register your problems
into `run_comparisons.jl` (or import them from your own module). Once that's
done:

```bash
# Run all defined problems with all installed drivers
julia --project=. comparisons/run_comparisons.jl

# Subset of problems / drivers
julia --project=. comparisons/run_comparisons.jl --problems P_BB180,P_INEPT
julia --project=. comparisons/run_comparisons.jl --packages PULSAR_lbfgs,Krotov
julia --project=. comparisons/run_comparisons.jl --problems P_BB180 \
                                                  --packages PULSAR_lbfgs,QuantumControl
```

Results are printed as fixed-width tables and saved to
`comparisons/Results/results_YYYY-MM-DD_HHMMSS.json`.

---

## Defining problems

A `BenchmarkProblem` (defined in `Drivers/driver_interface.jl`) is a struct
that bundles the system, target, control template, fidelity threshold, and
time limit. A minimal example for a broadband ¹H 180° inversion:

```julia
using PULSAR

function my_broadband_180()
    sys = mr_system(spins=["1H"],
                     chemical_shifts_hz=collect(range(-6e3, 6e3; length=21)),
                     b0_tesla=14.1)
    target = state_target(spin_state(sys, [-1.0]))
    ctrl   = MRControl(sys; n_steps=400, total_time_s=500e-6,
                       drive_max_hz=20e3)
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

Then register it in `run_comparisons.jl` next to the existing `ALL_PROBLEMS`
dictionary.

The bundled drivers re-evaluate fidelity through PULSAR's canonical
`grape_state_kernel` (or `grape_lindblad_kernel` for open-system problems)
regardless of what the external solver reports — this is the apples-to-apples
fairness invariant.

---

## Available Drivers

| Driver key       | Package                | Method            |
|------------------|------------------------|-------------------|
| `PULSAR_lbfgs`   | PULSAR (built-in)      | L-BFGS            |
| `PULSAR_cmaes`   | PULSAR (built-in)      | CMA-ES            |
| `PULSAR_grape`   | PULSAR (built-in)      | GRAPE             |
| `PULSAR_lbfgsb`  | PULSAR (built-in)      | L-BFGS-B          |
| `PULSAR_cg`      | PULSAR (built-in)      | Nonlinear CG      |
| `QuantumControl` | QuantumControl.jl      | GRAPE             |
| `Krotov`         | Krotov.jl              | Krotov            |
| `QuTiP`          | QuTiP (Python)         | GRAPE             |
| `qopt`           | qopt (Python)          | GRAPE             |
| `Spinach`        | Spinach (MATLAB)       | GRAPE (optimcon)  |
| `SIMPSON`        | SIMPSON (C binary)     | GRAPE (optcontrol)|
| `Quandary`       | Quandary (C++ binary)  | gradient-based    |

Drivers fail gracefully when their backing toolchain is not installed —
`run_comparisons.jl` reports `NOT AVAILABLE` rather than aborting.

---

## Installing External Packages

### QuantumControl.jl / Krotov.jl

```julia
using Pkg
Pkg.add(["QuantumControl", "Krotov"])
```

### QuTiP / qopt (Python via PythonCall.jl)

```julia
using Pkg; Pkg.add("PythonCall")
```

```bash
pip install qutip qopt
```

### Spinach (MATLAB)

1. Install MATLAB.
2. Download Spinach from <https://spindynamics.org/Spinach.php>.
3. Add Spinach to your MATLAB path.
4. Optional: `Pkg.add("MATLAB")` for the Julia–MATLAB bridge. Without it,
   the driver falls back to launching `matlab` as a subprocess.

### SIMPSON

Download and build from <http://inano.au.dk/about/nmr-methods-and-software/simpson/>
and ensure `simpson` is on your `PATH`.

### Quandary

```bash
git clone https://github.com/LLNL/quandary
cd quandary && cmake . && make
export PATH="$PATH:$(pwd)/build"
```

---

## Adding a New Driver

1. Create `comparisons/Drivers/mypackage_driver.jl`:
   - Define `struct MyPackageDriver <: AbstractSolverDriver`.
   - Implement `run_driver(::MyPackageDriver, ::BenchmarkProblem) → BenchmarkResult`.
   - Probe availability at the top of `run_driver`; return
     `not_available_result(...)` if missing.
   - Wrap the optimization call in `try/catch`; return `error_result(...)`
     on failure.
   - Always re-evaluate fidelity using `grape_state_kernel(waveform, ctrl)`
     (or `grape_lindblad_kernel` for open-system problems).
2. Include the new file in `run_comparisons.jl`.
3. Register it in the `ALL_DRIVERS` dict.

---

## Output Format

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

JSON results are written to `comparisons/Results/results_*.json` —
that directory is `.gitignore`'d so private benchmark data stays local.

---

## Fairness invariants

- **Canonical fidelity re-evaluation.** Every driver's reported fidelity is
  recomputed through PULSAR's `grape_state_kernel` /
  `grape_lindblad_kernel` after the external solver returns.
- **Same initial guess.** All drivers receive the same `guess_seed`.
- **`time_limit_s`** is advisory.
- **Open-system problems** require `LindbladMRControl`; drivers that do not
  support open systems return an error rather than a silent (closed-system)
  result.
