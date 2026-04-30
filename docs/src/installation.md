# Installation

Pulsar.jl requires **Julia ≥ 1.9** (it uses package extensions for optional
dependencies).

## From the repository

Until Pulsar is registered in the General registry, install directly from
the public Git URL:

```julia
julia> ]
pkg> add https://github.com/DaJo2025/Pulsar.jl
```

## From a local clone (for development)

```bash
git clone https://github.com/DaJo2025/Pulsar.jl
cd Pulsar.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Optional dependencies

Pulsar loads optional features lazily through Julia 1.9 package extensions.
To enable them, simply add the backing packages to your environment.

| Feature | Add | Activates |
|---|---|---|
| Plotting (`plot_convergence`, `plot_controls`, `plot_bloch_trajectory`, …) | `pkg> add Plots` | `PulsarPlotsExt` |
| Forward-mode AD | `pkg> add ForwardDiff` | `PulsarForwardDiffExt` |
| Reverse-mode AD | `pkg> add Zygote` | `PulsarZygoteExt` |
| Bound-constrained L-BFGS-B | `pkg> add LBFGSB` | `PulsarLBFGSBExt` |

GPU backends activate automatically when their packages are installed:

| Backend | Add |
|---|---|
| CUDA (NVIDIA GPUs) | `pkg> add CUDA` |
| Metal (Apple Silicon) | `pkg> add Metal` |

If neither is present, Pulsar falls back to the CPU backend with a warning.

## Verifying the install

```julia
julia> using Pulsar
julia> sys = qubit_system(1, ComplexF64[1 0; 0 -1], [ComplexF64[0 1; 1 0]])
julia> sys.dim, sys.n_controls
(2, 1)
```

## Running the test suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite is organized per layer (Physics, Algorithms, Computation, Integration,
Application, Architecture, AdvancedFeatures, Optimization, Parallelization,
Runtime, IO, Utilities) and runs to completion on a stock Julia 1.9 install.
