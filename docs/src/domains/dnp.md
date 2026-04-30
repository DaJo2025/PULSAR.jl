# DNP — dynamic nuclear polarization

Dynamic nuclear polarization (DNP) transfers electron spin polarization to
coupled nuclei via microwave irradiation under MAS or static conditions.
Pulsar's DNP support lives in
[`src/Types/DNPSpinSystem.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Types/DNPSpinSystem.jl)
and
[`src/Application/MR/DNP/DNPOptControl.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Application/MR/DNP/DNPOptControl.jl).

## Building a DNP system

`dnp_system` takes positional arguments for the electron spin number, the
list of nuclear spin numbers, the static field, the microwave carrier
frequency, and the MAS rotor frequency:

```julia
sys = dnp_system(
    1//2,                    # electron spin S
    [1//2],                  # one ¹H nucleus
    9.4,                     # B0 (T)
    263e9,                   # microwave frequency (Hz)
    8e3;                     # MAS rotor frequency (Hz; 0 for static)
    g_vals          = (2.0023, 2.0023, 2.0023),
    A_vals          = [(2.5e6, 1.0e6, 1.0e6)],   # Hyperfine principal values (Hz)
    nuclear_isotope = "1H",
)
```

Hyperfine couplings (`A_vals`) and electron / nuclear CSA, dipolar
couplings, RF carrier offset, extra drift terms, and custom controls are
all keyword arguments. See the function docstring for the full surface.

The default control set is `[Sx, Sy, Ix, Iy]` — microwave (electron x/y)
and RF (nuclear x/y). Pass `custom_controls` to override.

## Optimization workflow

`optimcon_dnp` is the dedicated DNP entry point:

```julia
ctrl  = random_controls(sys, 10e-6, 2000; amplitude = 0.5)
config = LBFGSConfig(max_iter = 200, verbose = false)

result = optimcon_dnp(sys, ctrl;
    config           = config,
    orientations     = nothing,           # or a Vector{NTuple{3,Float64}} of Euler angles
    use_lindblad     = false,             # true to include T1/T2 relaxation
    T1_electron_s    = 1e-3,
    T2_electron_s    = 1e-6,
    T1_nuclear_s     = 10.0,
    cross_relax_rate = 0.0,
)
```

Internally the wrapper

- builds the joint electron–nuclear drift in the doubly-rotating frame
- adds Wigner-rotated MAS Hamiltonian samples when `omega_r_hz > 0`
- powder-averages over `orientations` (use `powder_grid(N)` from
  `WignerRotations.jl` to generate angular grids)
- maximises the nuclear-Zeeman expectation value
  `dnp_polarization_fidelity` as the objective

When `use_lindblad = true` the optimisation runs in Liouville space, with
T1/T2 jump operators built from the supplied relaxation times.

## Helper functions

| Function | Purpose |
|---|---|
| `electron_polarized_state(sys; T_K)` | Boltzmann-weighted electron + nuclear thermal state |
| `nuclear_polarization_operator(sys, k)` | `Iz` on the `k`-th nucleus in the joint Hilbert space |
| `dnp_polarization_fidelity(ρ_or_ψ, sys)` | Tr(ρ · I_z^total) — the DNP objective |
| `powder_grid(N)` | `[(α, β, γ), …]` powder average grid (Zaremba/Lebedev) |

## Common patterns

- **Solid effect** — single-electron, single-nucleus, narrow EPR line, low
  B₀; matching condition `ω_mw = ω_S ± ω_I`.
- **Cross effect** — two coupled electrons + nucleus; the DNP enhancement
  peaks when `ω_S1 − ω_S2 = ω_I`.
- **Pulsed DNP (NOVEL, BEAM, PulsePOL, etc.)** — explicit microwave pulse
  sequence as a `ControlSequence`, optimised for polarization-transfer
  efficiency under MAS modulation.
