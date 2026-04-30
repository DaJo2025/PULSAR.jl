# Pulse export

PULSAR's `IO` layer ([`src/IO/Output.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/IO/Output.jl), [`PulseExport.jl`](https://github.com/DaJo2025/PULSAR.jl/blob/main/src/IO/PulseExport.jl)) exports optimized pulses to the
file formats consumed by spectrometers and quantum-computing toolchains.

## API entry point

```julia
export_pulse(ctrl, "results/my_pulse.bruker"; format=:bruker)
export_pulse(ctrl, "results/my_pulse.csv";    format=:csv)
```

`export_pulse` dispatches on the `format` keyword and produces a side-effect
on disk. The full list of registered exporters is available via:

```julia
list_exporters()
```

## Built-in formats

| Format | Domain | Notes |
|---|---|---|
| `:bruker` | NMR / EPR | JCAMP-DX polar `(amp_%, phase_deg)` |
| `:jeol` | NMR | JEOL shape file |
| `:epr` | EPR | EPR-spectrometer-specific shape |
| `:pulseq` | MRI | `PulseqSequence` (Pulseq spec) |
| `:qiskit` | QC superconducting | `QiskitWaveformExport` |
| `:quil_t` | QC | QuilT timing-aware export |
| `:qua` | QC | QUA (Quantum Machines) |
| `:pulser` | Neutral-atom | Pulser pulse format |
| `:csv` | Generic | Plain-text fallback |

## Bruker format details

Bruker JCAMP-DX shape files use polar encoding. PULSAR's forward (save)
conversion is:

```julia
amp_pct   = clamp(√(wx² + wy²) / pwr_level × 100, 0, 100)
phase_deg = mod(atand(wy, wx), 360)
```

Note that PULSAR ships only the **save** path. There is no `load_bruker_shape`
in the package — for round-trip work, write a small helper:

```julia
function _load_bruker_as_w(filepath, n_ts, n_ctrl)
    # parse JCAMP-DX (amp_pct, phase_deg) lines, then:
    amp_norm  = amp_pct ./ 100
    phi       = phase_deg .* π/180
    wx        = amp_norm .* cos.(phi)
    wy        = amp_norm .* sin.(phi)
    return reshape(hcat(wx, wy)', 2, n_ts)
end
```

## Custom exporters

To add a new format, define a callable and register it:

```julia
register_exporter(:my_format, (ctrl, filepath; kwargs...) -> begin
    open(filepath, "w") do io
        # … write your format
    end
end)
```

`replace_exporter` overwrites an existing registration; `list_exporters`
returns the current symbol → callable map.
