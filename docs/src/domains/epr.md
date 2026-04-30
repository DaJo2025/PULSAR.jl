# EPR — electron paramagnetic resonance

EPR systems differ from NMR by:

- much larger gyromagnetic ratios (~660× ¹H);
- anisotropic g-tensors and hyperfine couplings;
- broad resonance spectra requiring broadband or chirp pulses.

The EPR system type lives in
[`src/Types/EPRSpinSystem.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Types/EPRSpinSystem.jl)
and the optimisation wrapper in
[`src/Application/MR/EPR/EPROptControl.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/MR/EPR/EPROptControl.jl).

## Building a system

`epr_system` takes the electron spin number, a vector of nuclear spin
numbers, the static field, and the microwave carrier frequency, plus
keyword arguments for g/A/D/E tensors:

```julia
sys = epr_system(
    1//2,                    # electron spin S
    [1//2],                  # one ¹H ligand nucleus (or [] for pure electron)
    0.35,                    # B0 (T)
    9.4e9;                   # microwave carrier (Hz)
    g_vals  = (2.0023, 2.0084, 2.0260),         # g principal values
    g_euler = (0.0, 0.0, 0.0),                  # PAS Euler angles
    A_vals  = [(50e6, 50e6, 80e6)],             # hyperfine principal values (Hz)
    A_euler = [(0.0, 0.0, 0.0)],
    D_mhz   = 0.0,                              # zero-field-splitting D (MHz)
    E_mhz   = 0.0,                              # ZFS rhombicity E (MHz)
)
```

Auxiliary types (CSA, dipolar coupling) are still useful when extending
the system manually:

| Type | Purpose |
|---|---|
| `EPRSpinSystem` | Top-level system |
| `CSATensor` | Chemical-shift / g-shift anisotropy descriptor |
| `DipolarCoupling` | Dipole–dipole between two spins |

## Optimisation workflow

EPR uses the same `optimcon` dispatch as solution NMR:

```julia
target = unitary_target(my_target_unitary)
ctrl   = random_controls(sys, 100e-9, 200; amplitude = 0.5)

result = optimcon(sys, target, ctrl;
                  config       = LBFGSConfig(max_iter = 200),
                  bands        = [BandWeight(-50e6, 1.0),
                                  BandWeight(  0.0, 1.0),
                                  BandWeight(+50e6, 1.0)],
                  orientations = nothing)            # or a powder grid
```

Specialised objectives include broadband excitation profiles, AWG-shaped
chirps, and DEER-style double electron–electron resonance pulses.

## Powder averaging

EPR samples are usually polycrystalline. Use the MAS-style powder grid
even for static EPR:

```julia
grid = powder_grid(64)         # 64-point Zaremba/Lebedev grid of (α, β, γ)
result = optimcon(sys, target, ctrl;
                  config       = LBFGSConfig(max_iter = 200),
                  orientations = grid)
```

`compute_grape_gradient_powder` is the batched gradient kernel used
internally; you can call it directly for custom optimisation loops.
