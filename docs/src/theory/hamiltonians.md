# Hamiltonians

Pulsar splits every Hamiltonian into a **drift** part `H_drift` and a sum of
**control** parts `{H_c}` weighted by time-dependent waveforms `u_c(t)`:

```math
H(t) = H_\text{drift} + \sum_c u_c(t)\, H_c
```

## Units and conventions

- All Hamiltonians are in **angular frequency units (rad/s)** with `ħ = 1`.
- To convert from Hz to rad/s, multiply by `2π`.
- Time evolution: `U = exp(-i H Δt)` for piecewise-constant slices.
- Spin operators (NMR convention, dimensionless `Iα = σα / 2`):

  ```julia
  Ix = ComplexF64[0  1;  1  0] ./ 2
  Iy = ComplexF64[0 -im; im  0] ./ 2
  Iz = ComplexF64[1  0;  0 -1] ./ 2
  ```

## Building system Hamiltonians

Pulsar provides domain-specific constructors that assemble drift Hamiltonians
from physical parameters:

| Constructor | Domain | Drift includes |
|---|---|---|
| `qubit_system` | Generic qubit | User-supplied |
| `mr_system` | Solution NMR | Chemical shifts, J-couplings |
| `heteronuclear_system` | Heteronuclear NMR | Multi-carrier offsets, J |
| `epr_system` | EPR | g-tensor, hyperfine, CSA, dipolar |
| `mas_spin_system` | MAS solid-state | Anisotropic + MAS rotation |
| `bloch_system` | MRI / Bloch | Larmor, gradients, B0 inhomogeneity |
| `dnp_system` | DNP | Electron + nuclear + microwave |
| `transmon_system` | Superconducting | Qubit + anharmonicity |
| `trapped_ion_system` | Trapped ion | Carrier + sidebands |
| `neutral_atom_system` | Rydberg array | Two-level + Rydberg blockade |
| `spin_qubit_system` | Quantum dot | Exchange + Zeeman |
| `nv_center_system` | NV diamond | ZFS + hyperfine |

After construction, `hamiltonian(sys)` returns the symbolic drift; control
Hamiltonians are accessed via `sys.H_controls`.

## Single-spin operators

```julia
spin_op(sys, spin_idx, :x)   # Iₓ on the chosen spin
spin_op(sys, spin_idx, :z)   # Iz
spin_op(sys, spin_idx, :+)   # raising
spin_op(sys, spin_idx, :-)   # lowering
spin_state(sys, ...)          # initial / target product states
```

These respect the system's tensor structure — `spin_op(sys, 2, :x)` returns
`I ⊗ Ix ⊗ I` for a 3-spin system.

## Inspection

For any system, you can verify controllability by computing
`‖[H_drift, H_c]‖`. If a control Hamiltonian commutes with the drift, that
control direction is uncontrollable for state-to-state transfer.

```julia
for (j, Hc) in enumerate(sys.H_controls)
    nc = norm(H_drift * Hc - Hc * H_drift)
    println("‖[H_drift, H_$j]‖ = $nc",  nc < 1e-10 ? " ⚠ commutes" : "")
end
```
