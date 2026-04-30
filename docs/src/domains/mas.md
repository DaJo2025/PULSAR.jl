# MAS — magic-angle-spinning solid-state NMR

Solid-state NMR with sample rotation requires time-dependent Hamiltonians
and powder averaging. Pulsar's MAS support is in
[`src/Types/MASSpinSystem.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Types/MASSpinSystem.jl),
[`src/Computation/MASPropagators.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Computation/MASPropagators.jl),
and
[`src/Application/MR/SolidStateNMR/MASOptControl.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/MR/SolidStateNMR/MASOptControl.jl).

## Building a MAS system

`mas_spin_system` wraps an existing `MRSpinSystem` (its isotopes,
dimensions, and operator algebra) and adds the rotor frequency plus the
relevant anisotropic interactions:

```julia
base = mr_system(["13C", "13C"])

csa = [
    CSATensor(1, 0.0, 80.0, 0.3, (0.0, 0.0, 0.0)),     # spin 1: δ_iso=0, δ_aniso=80 Hz, η=0.3
]

dipolar = [
    DipolarCoupling(1, 2, 2200.0, (0.0, π / 3, 0.0)),   # 2200 Hz at β=π/3
]

sys = mas_spin_system(base, 12_500.0;                   # 12.5 kHz rotor
                       csa     = csa,
                       dipolar = dipolar)
```

## Time-dependent Hamiltonian

The drift varies as the sample rotates. For each rotor angle `t_k`,
Pulsar builds the rotor-frame Hamiltonian:

```julia
H_k = build_mas_hamiltonian(sys, t_k)
```

`compute_propagators(H_total, dt)` then assembles the per-step
propagators. For an arbitrary single-crystal orientation:

```julia
sys_rot = rotate_spin_system(sys, Ω)        # Ω = (α, β, γ)
```

## Powder averaging

```julia
grid = powder_grid(64)                       # 64-point Zaremba/Lebedev grid

result = optimcon(sys, target, ctrl;
                  config       = LBFGSConfig(max_iter = 200),
                  orientations = grid)
```

The batched powder gradient kernel is `compute_grape_gradient_powder` —
called automatically by `optimcon(::MASSpinSystem, …)`, but available
directly for custom loops:

```julia
g = compute_grape_gradient_powder(sys, ctrl, target, grid)
```

## Common patterns

- **CP-MAS** — Hartmann–Hahn polarization transfer between
  dipolar-coupled spins, optimised over RF amplitudes and offsets.
- **DREAM / R-symmetry sequences** — built from elemental MAS pulses with
  rotor-synchronous timing.
- **Adiabatic CP under MAS** — robust to amplitude / offset spread by
  combining `MASSpinSystem` with `RobustConfig`.
