"""
    Utilities/PulseExport.jl

Export optimised pulse waveforms to spectrometer file formats.

Currently supported:
  - Bruker JCAMP-DX shape format (`.bruker` / `.dsh` / `.dajo`)
"""

using Printf
using Dates

# ─── Bruker JCAMP-DX format ───────────────────────────────────────────────────

"""
    save_bruker_shape(filepath, w_mat, pwr_level; dt, kwargs...) → filepath

Save an optimised Cartesian waveform to a Bruker JCAMP-DX shape file.

# Positional arguments

| Argument     | Type               | Description                                        |
|--------------|--------------------|----------------------------------------------------|
| `filepath`   | `String`           | Output file path, e.g. `"pulse.bruker"`            |
| `w_mat`      | `Matrix{Float64}`  | `[2 × N_TS]` waveform in rad/s (row 1=x, row 2=y) |
| `pwr_level`  | `Float64`          | Peak RF amplitude in rad/s used for normalisation  |

# Keyword arguments

| Keyword              | Default                   | Description                                       |
|----------------------|---------------------------|---------------------------------------------------|
| `dt`                 | *(required)*              | Step duration in seconds; scalar or `Vector`      |
| `title`              | `""`                      | `##TITLE=` header field                           |
| `owner`              | `""`                      | `##OWNER=` header field                           |
| `origin`             | `"PULSAR.jl"`             | `##ORIGIN=` header field                          |
| `shape_parameters`   | `"no info"`               | Free-text in `##\$SHAPE_PARAMETERS= Type: …`      |
| `shape_exmode`       | `""`                      | `##\$SHAPE_EXMODE=` (e.g. `"Excitation"`)         |
| `shape_totrot_deg`   | `180.0`                   | Total rotation angle in degrees                   |
| `shape_type`         | `"Optimal_control_pulse"` | `##\$SHAPE_TYPE=` label                           |
| `bandwidth_hz`       | `0.0`                     | Pulse bandwidth in Hz (used for BWFAC)            |

# Amplitude / phase convention

Each time step is written as one line: `amplitude_percent, phase_deg`

- `amplitude_percent = 100 × sqrt(wx² + wy²) / pwr_level`  (clamped to [0, 100])
- `phase_deg = mod(atan2(wy, wx) × 180/π, 360)`             (∈ [0, 360))

# Computed header fields (not user-settable)

- `MINX / MAXX` — min / max amplitude percentage across all steps
- `MINY / MAXY` — min / max phase (degrees)
- `BWFAC = T_pulse × bandwidth_hz`
- `INTEGFAC = flip_angle_rad / (pwr_level × T_pulse)`

# Example

```julia
save_bruker_shape(
    "broadband_90.bruker", w_mat_opt, RF_MAX;
    dt               = DT,
    title            = "Broadband 90° ¹H pulse",
    shape_exmode     = "Excitation",
    shape_totrot_deg = 90.0,
    bandwidth_hz     = 12_000.0,
)
```
"""
function save_bruker_shape(
    filepath         :: String,
    w_mat            :: Matrix{Float64},
    pwr_level        :: Float64;
    dt               :: Union{Float64, Vector{Float64}},
    title            :: String  = "",
    owner            :: String  = "",
    origin           :: String  = "PULSAR.jl",
    shape_parameters :: String  = "no info",
    shape_exmode     :: String  = "",
    shape_totrot_deg :: Float64 = 180.0,
    shape_type       :: String  = "Optimal_control_pulse",
    bandwidth_hz     :: Float64 = 0.0,
)::String
    N_TS = size(w_mat, 2)
    size(w_mat, 1) == 2 ||
        throw(ArgumentError(
            "w_mat must be a [2 × N_TS] matrix; got size $(size(w_mat))"))
    pwr_level > 0 ||
        throw(ArgumentError("pwr_level must be positive; got $pwr_level"))

    # ── Time grid ──────────────────────────────────────────────────────────────
    dt_vec = dt isa Float64 ? fill(dt, N_TS) : collect(Float64, dt)
    length(dt_vec) == N_TS ||
        throw(ArgumentError(
            "length(dt) = $(length(dt_vec)) ≠ N_TS = $N_TS"))
    T_pulse = sum(dt_vec)

    # ── Convert Cartesian (wx, wy) → amplitude % and phase (deg) ──────────────
    amp_pct   = Vector{Float64}(undef, N_TS)
    phase_deg = Vector{Float64}(undef, N_TS)
    for n in 1:N_TS
        a = sqrt(w_mat[1, n]^2 + w_mat[2, n]^2)
        amp_pct[n]   = clamp(a / pwr_level * 100.0, 0.0, 100.0)
        phase_deg[n] = mod(atand(w_mat[2, n], w_mat[1, n]), 360.0)
    end

    # ── Computed header values ─────────────────────────────────────────────────
    minx     = minimum(amp_pct)
    maxx     = maximum(amp_pct)
    miny     = minimum(phase_deg)
    maxy     = maximum(phase_deg)
    bwfac    = T_pulse * bandwidth_hz
    flip_rad = shape_totrot_deg * π / 180.0
    integfac = flip_rad / (pwr_level * T_pulse)

    # ── Date / time strings ────────────────────────────────────────────────────
    now_dt   = now()
    date_str = Dates.format(now_dt, "yyyy/mm/dd")
    time_str = Dates.format(now_dt, "HH:MM:SS")

    # ── Bruker scientific-notation formatter: no '+' in positive exponents ─────
    # Bruker uses "1.000000E02" not "1.000000E+02"
    _bfmt(x::Float64) = replace(@sprintf("%.6E", x), "E+" => "E")

    # ── Write file ─────────────────────────────────────────────────────────────
    open(filepath, "w") do io
        write(io, "##TITLE= $(title)\n")
        write(io, "##JCAMP-DX= 5.00 Bruker JCAMP library\n")
        write(io, "##DATA TYPE= Shape Data\n")
        write(io, "##ORIGIN= $(origin)\n")
        write(io, "##OWNER= $(owner)\n")
        write(io, "##DATE= $(date_str)\n")
        write(io, "##TIME= $(time_str)\n")
        write(io, "##\$SHAPE_PARAMETERS= Type: $(shape_parameters)\n")
        write(io, "##MINX= $(_bfmt(minx))\n")
        write(io, "##MAXX= $(_bfmt(maxx))\n")
        write(io, "##MINY= $(_bfmt(miny))\n")
        write(io, "##MAXY= $(_bfmt(maxy))\n")
        write(io, "##\$SHAPE_EXMODE= $(shape_exmode)\n")
        write(io, "##\$SHAPE_TOTROT= $(_bfmt(shape_totrot_deg))\n")
        write(io, "##\$SHAPE_TYPE= $(shape_type)\n")
        write(io, "##\$SHAPE_USER_DEF= \n")
        write(io, "##\$SHAPE_REPHFAC= \n")
        write(io, "##\$SHAPE_BWFAC= $(_bfmt(bwfac))\n")
        write(io, "##\$SHAPE_BWFAC50= \n")
        write(io, "##\$SHAPE_INTEGFAC= $(_bfmt(integfac))\n")
        write(io, "##\$SHAPE_MODE= 0\n")
        write(io, "##NPOINTS= $(N_TS)\n")
        write(io, "##XYPOINTS= (XY..XY)\n")
        for n in 1:N_TS
            write(io, "$(_bfmt(amp_pct[n])), $(_bfmt(phase_deg[n]))\n")
        end
        write(io, "##END")   # no trailing newline — Bruker convention
    end

    return filepath
end

# ─── Bruker JCAMP-DX loader ──────────────────────────────────────────────────

"""
    load_bruker_shape(filepath; pwr_level=1.0, dt=nothing) → OptimizedPulse

Load a Bruker JCAMP-DX shape file and reconstruct the complex waveform.

Inverse of `save_bruker_shape`. Reads `(amplitude_%, phase_deg)` columns and
converts back to Cartesian: `sample = (amp/100 * pwr_level) * exp(im * φ)`.

# Arguments
- `filepath`   — path to `.bruker` / `.shape` file
- `pwr_level`  — RF power level in rad/s used during export (default `1.0`)
- `dt`         — time step in seconds. Bruker files do not store this; pass
                 `nothing` to leave `dt = 0.0` with a warning.

# Returns
`OptimizedPulse` with `samples::Vector{ComplexF64}`, `dt::Float64`, and
`meta` containing `:n_points`, `:pwr_level`, `:title`, `:shape_totrot_deg`.
"""
function load_bruker_shape(
    filepath   :: String;
    pwr_level  :: Float64              = 1.0,
    dt         :: Union{Float64,Nothing} = nothing,
)::OptimizedPulse
    isfile(filepath) || throw(ArgumentError("File not found: $filepath"))

    lines = readlines(filepath)

    amp_pct   = Float64[]
    phase_deg = Float64[]
    in_data   = false
    meta_title   = ""
    meta_totrot  = 180.0
    meta_npoints = 0

    for line in lines
        line = strip(line)
        isempty(line) && continue

        if startswith(line, "##TITLE=")
            meta_title = strip(line[length("##TITLE=") + 1 : end])
        elseif startswith(line, "##\$SHAPE_TOTROT=")
            v = strip(line[length("##\$SHAPE_TOTROT=") + 1 : end])
            try meta_totrot = parse(Float64, replace(v, "E" => "e")) catch end
        elseif startswith(line, "##NPOINTS=")
            v = strip(line[length("##NPOINTS=") + 1 : end])
            try meta_npoints = parse(Int, v) catch end
        elseif startswith(line, "##XYPOINTS=")
            in_data = true
        elseif startswith(line, "##END")
            in_data = false
        elseif in_data
            parts = split(line, ",")
            length(parts) == 2 || continue
            a_str = replace(strip(parts[1]), "E" => "e")
            p_str = replace(strip(parts[2]), "E" => "e")
            try
                push!(amp_pct,   parse(Float64, a_str))
                push!(phase_deg, parse(Float64, p_str))
            catch
            end
        end
    end

    isempty(amp_pct) && error("No data points found in $filepath")

    samples = Vector{ComplexF64}(undef, length(amp_pct))
    for n in eachindex(amp_pct)
        amp  = amp_pct[n] / 100.0 * pwr_level
        phi  = phase_deg[n] * π / 180.0
        samples[n] = amp * (cos(phi) + im * sin(phi))
    end

    if dt === nothing
        @warn "load_bruker_shape: dt not provided; setting dt = 0.0. " *
              "Pass dt=<value_in_seconds> to get correct time axis."
    end
    dt_val = dt === nothing ? 0.0 : dt

    return OptimizedPulse(samples, dt_val;
        title            = meta_title,
        shape_totrot_deg = meta_totrot,
        n_points         = length(samples),
        pwr_level        = pwr_level,
    )
end

# ─── New unified export dispatcher ───────────────────────────────────────────

"""
    export_pulse(pulse; application, vendor, save=true, output_dir="output",
                 format=nothing, kwargs...) → export result struct

Export an `OptimizedPulse` to a spectrometer or quantum-hardware file format.

This is a thin dispatcher that delegates all logic and file I/O to
`PULSAR.Output.export_pulse`. See `PULSAR.Output` for full documentation.

# Supported application / vendor combinations

| `application` | `vendor`      | Output                          |
|---------------|---------------|---------------------------------|
| `:NMR_EPR`    | `:bruker`     | Bruker JCAMP-DX `.shape` file   |
| `:NMR_EPR`    | `:jeol`       | JEOL Delta RF shape `.jrf` file |
| `:NMR_EPR`    | `:epr_bruker` | Bruker Xepr AWG IQ `.shp` file  |
| `:MRI`        | `:pulseq`     | Pulseq open MRI `.seq` file     |
| `:QC`         | `:qiskit`     | Qiskit Pulse waveform `.json`   |
| `:QC`         | `:quil_t`     | Quil-T DEFWAVEFORM `.quil`      |
| `:QC`         | `:qua`        | QUA I/Q play snippet `_qua.py`  |
| `:QC`         | `:pulser`     | Pulser waveform script `_pulser.py` |

# Arguments

- `pulse`       — `OptimizedPulse` canonical pulse object
- `application` — domain symbol (`:NMR_EPR`, `:MRI`, or `:QC`)
- `vendor`      — target symbol (`:bruker`, `:jeol`, etc.)
- `save`        — if `true`, write file(s) to `output_dir`; if `false`, return in-memory struct only
- `output_dir`  — directory for output files (created if absent); ignored when `save=false`
- `format`      — optional format hint (vendor-specific; pass `nothing` for default)
- `kwargs...`   — forwarded to the vendor exporter (e.g. `name`, `flip_angle`, `carrier_hz`)

# Example

```julia
pulse  = OptimizedPulse(w_mat, pwr_level, dt; flip_angle=90.0)
result = export_pulse(pulse; application=:NMR_EPR, vendor=:bruker,
                      save=true, output_dir="pulses", name="BB90")
println(result.filepath)
```
"""
function export_pulse(
    pulse        :: OptimizedPulse;
    application  :: Union{Symbol, AbstractString},
    vendor       :: Union{Symbol, AbstractString},
    save         :: Bool               = true,
    output_dir   :: AbstractString     = "output",
    format                             = nothing,
    kwargs...
)
    return Output.export_pulse(pulse;
        application = application,
        vendor      = vendor,
        save        = save,
        output_dir  = output_dir,
        format      = format,
        kwargs...
    )
end
