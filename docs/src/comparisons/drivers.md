# Drivers

A *driver* is a thin adapter that takes a `BenchmarkProblem`, hands it to an
external optimizer in that solver's native idiom, and returns a
`BenchmarkResult`. All drivers implement the same one-method contract.

## The `AbstractSolverDriver` interface

```julia
abstract type AbstractSolverDriver end

run_driver(driver::ConcreteDriver, problem::BenchmarkProblem) → BenchmarkResult
```

Implementation rules (enforced by every shipped driver):

1. **Probe availability first.** If the backing package / binary is not
   installed, return `not_available_result(driver_name, problem.id, msg)`
   with a one-line install hint. Never throw on missing dependencies.
2. **Wrap the optimizer call in `try/catch`.** Return `error_result(...)`
   on any thrown exception so a single broken solver does not abort the
   whole run.
3. **Re-evaluate fidelity through Pulsar.** Whatever the external solver
   reports, the `BenchmarkResult.fidelity` field must be the value
   returned by `grape_state_kernel(waveform, ctrl)` (closed-system) or
   `grape_lindblad_kernel(waveform, ctrl)` (open-system). This is the
   apples-to-apples invariant.
4. **Honor the shared seed.** Build the initial waveform from
   `Random.MersenneTwister(problem.guess_seed)` so all drivers start from
   the same control vector.
5. **Use the canonical layout.** Return `controls` as `[n_ctrl × n_t]`
   (the optimizer-internal layout), not the `ControlSequence`-facing
   `[n_t × n_ctrl]` layout.

## Shipped drivers

Driver keys are the strings used in `--packages` on the CLI. The keys map
to driver instances in the `ALL_DRIVERS` dict in
[`run_comparisons.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/comparisons/run_comparisons.jl).

| Driver key       | Backend                      | Method            | External requirement |
|------------------|------------------------------|-------------------|----------------------|
| `Pulsar_lbfgs`   | Pulsar built-in              | L-BFGS            | none                 |
| `Pulsar_lbfgsb`  | Pulsar built-in              | L-BFGS-B          | `LBFGSB.jl` (extension) |
| `Pulsar_cg`      | Pulsar built-in              | Nonlinear CG      | none                 |
| `Pulsar_grape`   | Pulsar built-in              | First-order GRAPE | none                 |
| `Pulsar_cmaes`   | Pulsar built-in              | CMA-ES            | none                 |
| `QuantumControl` | [`QuantumControl.jl`](https://github.com/JuliaQuantumControl/QuantumControl.jl) | GRAPE | `Pkg.add("QuantumControl")` |
| `Krotov`         | [`Krotov.jl`](https://github.com/JuliaQuantumControl/Krotov.jl) | Krotov | `Pkg.add("Krotov")` |
| `QuTiP`          | [QuTiP](https://qutip.org/) (Python) | GRAPE | `Pkg.add("PythonCall")` + `pip install qutip` |
| `qopt`           | [qopt](https://github.com/qutech/qopt) (Python) | GRAPE | `Pkg.add("PythonCall")` + `pip install qopt` |
| `Spinach`        | [Spinach](https://spindynamics.org/Spinach.php) (MATLAB) | GRAPE (`optimcon`) | MATLAB + Spinach on path |
| `SIMPSON`        | [SIMPSON](http://inano.au.dk/about/nmr-methods-and-software/simpson/) (C) | GRAPE (`optcontrol`) | `simpson` on `PATH` |
| `Quandary`       | [Quandary](https://github.com/LLNL/quandary) (C++) | gradient-based | `quandary` binary on `PATH` |

When a driver's backing toolchain is missing, the row appears as
`NOT AVAILABLE` in the report with the install hint shown above; the
overall run continues.

## Driver source files

Each entry above corresponds to one file in
[`comparisons/Drivers/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/comparisons/Drivers):

| File | Driver type |
|---|---|
| `driver_interface.jl`     | `AbstractSolverDriver`, `BenchmarkResult`, `not_available_result`, `error_result` |
| `pulsar_driver.jl`        | `PulsarDriver(method::Symbol)` — covers all five Pulsar keys |
| `quantumcontrol_driver.jl`| `QuantumControlDriver` |
| `krotov_driver.jl`        | `KrotovDriver` |
| `qutip_driver.jl`         | `QuTiPDriver` |
| `qopt_driver.jl`          | `QoptDriver` |
| `spinach_driver.jl`       | `SpinachDriver` |
| `simpson_driver.jl`       | `SIMPSONDriver` |
| `quandary_driver.jl`      | `QuandaryDriver` |

## The translator layer

External solvers expect the problem in their own native form — a
serialized C struct for SIMPSON, a Python `Hamiltonian` object for QuTiP,
a MATLAB `control` struct for Spinach, and so on. Rather than each driver
re-implementing this conversion, the
[`comparisons/Translator/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/comparisons/Translator)
subsystem provides a shared, declarative emission pipeline:

| Module | Role |
|---|---|
| `Capabilities.jl`         | Per-package feature flags (open-system, ensembles, time-dependent drift, …). Used to decide which `(driver, problem)` pairs are valid. |
| `PhysicsAnnotation.jl`    | Translates a Pulsar `MRSpinSystem` / `TransmonSystem` / etc. into a solver-agnostic `PhysicsAnnotation` record. |
| `TransmonAnnotation.jl`   | The QC-specific specialisation of the above. |
| `Emitters/`               | One emitter per backend that turns the annotation + control template into the solver's native input file or in-memory object. |
| `EmitterHelpers.jl`       | Shared formatting (number printing, units, file-name conventions). |
| `Subprocess.jl`           | Common wrapper around `run(...)` for subprocess-based drivers (SIMPSON, Quandary, MATLAB CLI). |

The driver itself is then small: probe availability, build the annotation,
emit, run, parse the waveform back, re-evaluate fidelity through Pulsar,
return the `BenchmarkResult`.

## Adding a new driver

1. Create `comparisons/Drivers/mypackage_driver.jl`:

   ```julia
   struct MyPackageDriver <: AbstractSolverDriver end

   function run_driver(::MyPackageDriver, problem::BenchmarkProblem)
       driver_name = "MyPackage/method"

       # 1. Availability probe
       if !_my_package_available()
           return not_available_result(driver_name, problem.id,
               "Install: Pkg.add(\"MyPackage\")")
       end

       # 2. Build initial guess from the shared seed
       rng = Random.MersenneTwister(problem.guess_seed)
       w0  = randn(rng, n_ctrl, n_t)

       # 3. Run external optimizer in try/catch
       w_opt, hist, conv, n_iter, t_s = try
           _my_package_optimize(problem, w0)
       catch err
           return error_result(driver_name, problem.id, err)
       end

       # 4. Canonical fidelity re-evaluation
       F = grape_state_kernel(w_opt, problem.ctrl)

       return BenchmarkResult(
           driver_name, problem.id, F, t_s, n_iter, conv,
           w_opt, hist, true, "", Dict{String,Any}(),
       )
   end
   ```

2. Add an `include(...)` for the new file in `run_comparisons.jl`.
3. Register the driver in `ALL_DRIVERS` with a string key — that key
   becomes the value users pass to `--packages`.
