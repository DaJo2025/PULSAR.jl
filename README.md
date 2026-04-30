# Pulsar.jl

**Pulse Design Library for Spin Control Algorithms and Rollout**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Julia](https://img.shields.io/badge/Julia-≥1.9-9558B2.svg?logo=julia)](https://julialang.org/)

Pulsar.jl is a Julia toolbox for designing optimal control pulses for quantum
systems — qubits (transmon, trapped-ion, neutral-atom, spin-qubit, NV-center)
and spin systems (NMR, EPR, MAS solid-state, MRI, DNP). It provides a unified
interface to a wide library of optimization algorithms (GRAPE, L-BFGS-B,
Krotov, GOAT, CRAB, CMA-ES, Nelder–Mead, basin hopping, …), and supports
closed- and open-system (Lindblad) dynamics, ensemble robust control, automatic
differentiation, and CPU / CUDA / Metal backends.

## Installation

Pulsar.jl requires Julia ≥ 1.9. Until it is registered in the General registry,
install directly from the repository:

```julia
julia> ]
pkg> add https://github.com/DaJo2025/Pulsar.jl
```

Optional features (plotting, autodiff, L-BFGS-B) are loaded lazily via Julia
package extensions when their backing packages are present:

```julia
pkg> add Plots ForwardDiff Zygote LBFGSB
```

## Quickstart — single-qubit state transfer |0⟩ → |1⟩

```julia
using Pulsar
using LinearAlgebra, Random

# Pauli matrices
σx = ComplexF64[0 1; 1 0];  σy = ComplexF64[0 -im; im 0];  σz = ComplexF64[1 0; 0 -1]

# 1-qubit system: small off-resonance drift, x/y quadrature drives
ω_offset_hz = 10.0
H_drift = (2π * ω_offset_hz / 2) .* σz
sys     = qubit_system(1, H_drift, [0.5σx, 0.5σy])

# Target: |0⟩ → |1⟩
target = state_target(ComplexF64[0.0, 1.0])

# Random initial control sequence
ctrl = random_controls(sys, π, 100; amplitude=0.5,
                        rng=MersenneTwister(42))

# Run GRAPE
config = GRAPEConfig(max_iter=500, step_size=0.05,
                     adapt_step_size=true, verbose=false)
result = grape_optimize(sys, target, ctrl; config=config)

# Re-simulate independently and verify
ctrl_opt = ControlSequence(Matrix(result.controls'), ctrl.dt, ctrl.n_steps)
F_check  = compute_fidelity(sys, ctrl_opt, target)
@assert abs(F_check - result.fidelity) < 1e-6

println("Final fidelity: ", round(F_check; digits=8))
```

## Architecture

Pulsar is organized into five strict layers (no upward dependencies):

| Layer | Subdirectory | Contents |
|---|---|---|
| 1a | `src/Types/` | System types — `QuantumSystem`, `MRSpinSystem`, `EPRSpinSystem`, `MASSpinSystem`, `BlochSystem`, `DNPSpinSystem`, `TransmonSystem`, `TrappedIonSystem`, `NeutralAtomSystem`, `SpinQubitSystem`, `NVCenterSystem`, `ControlSequence`, `Targets` |
| 1b | `src/Computation/` | Propagators, ensemble maps, MAS propagators, Bloch propagator, Wigner rotations |
| 1c | `src/Backend/` | CPU / CUDA / Metal hardware, parallelism strategies, hybrid scheduling, device registry |
| 2  | `src/Physics/` | Objectives, penalties, gradients, Lindblad, AutoDiff, UQ, sensitivity, MR physics |
| 3  | `src/Optimization/` | GRAPE family, second-order, direct search, constrained, robust, multi-objective, metaheuristic, QOC-specific (Krotov, GOAT, CRAB, T-GRAPE), analytic (BB1, SCROFULOUS, SK1, CORPSE, DRAG, STA, SLR, VERSE) |
| 4  | `src/IO/`, `src/Runtime/`, `src/Utilities/` | Pulse export (Bruker, JEOL, EPR, Pulseq, Qiskit, QuilT, QUA, Pulser), checkpoints, performance monitoring, algorithm selection, validation, visualization |
| 5  | `src/Application/MR/`, `src/Application/QuantumComputing/` | Domain-specific thin wrappers |

## Documentation

Full documentation — installation, theory, algorithm reference, domain guides,
and API reference — is hosted at:

> https://DaJo2025.github.io/Pulsar.jl/

To build the docs locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
# open docs/build/index.html
```

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite is organized per layer (Physics, Algorithms, Computation,
Integration, Application, Architecture, AdvancedFeatures, Optimization,
Parallelization, Runtime, IO, Utilities).

## Comparisons framework

`comparisons/` contains a driver-based framework for benchmarking Pulsar
against external optimal-control solvers (QuantumControl.jl, Krotov.jl, QuTiP,
Qopt, Spinach, SIMPSON, Quandary). Driver code is shipped; benchmark problem
definitions are not. To run a comparison, supply your own `BenchmarkProblem`
instances — see [`comparisons/README.md`](comparisons/README.md).

## License

Pulsar.jl is licensed under the [Apache License, Version 2.0](LICENSE).

## Citation

A citation file will be provided once the accompanying paper is released.
