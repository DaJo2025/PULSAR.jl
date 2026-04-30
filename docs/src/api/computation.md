# API — Computation

Propagators, ensemble maps, MAS, and Bloch primitives. Source:
[`src/Computation/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/Computation).

## Propagators

```@docs
compute_propagator
build_total_hamiltonian
compute_forward_propagators
compute_backward_propagators
```

## Ensemble

```@docs
ensemble_fidelity
```

## MAS

```@docs
build_mas_hamiltonian
rotate_spin_system
compute_grape_gradient_powder
wigner_d2
wigner_D2
powder_grid
```

## Bloch propagator (MRI)

```@docs
bloch_forward_pass
bloch_adjoint_pass
bloch_fidelity
slice_profile_fidelity
```
