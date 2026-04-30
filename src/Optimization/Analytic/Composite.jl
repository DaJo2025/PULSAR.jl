# ============================================================================
# Analytic/Composite.jl — Analytic composite pulse design
# ============================================================================

# ── Shared return types ──────────────────────────────────────────────────────

"""
    CompositePulseSegment(flip_deg, phase_deg)

One hard-pulse segment. `duration` is always 1.0; real time = flip_deg / (360 × rf_hz).
"""
struct CompositePulseSegment
    flip_deg  :: Float64
    phase_deg :: Float64
    duration  :: Float64
    CompositePulseSegment(f::Real, p::Real) = new(Float64(f), Float64(p), 1.0)
end

"""
    AnalyticPulse(t, B1, phase, dt)

Time-domain shaped pulse. `B1` is normalised to peak = 1; `phase` in radians.
"""
struct AnalyticPulse
    t     :: Vector{Float64}
    B1    :: Vector{Float64}
    phase :: Vector{Float64}
    dt    :: Float64
end

# ── Internal helper ──────────────────────────────────────────────────────────
_w360(x::Real) = mod(Float64(x), 360.0)

# ── BB1 ─────────────────────────────────────────────────────────────────────

"""
    bb1(θ_deg) → Vector{CompositePulseSegment}

Wimperis BB1 broadband composite: `[θ_0, 180_φ, 360_3φ, 180_φ]`, φ = acos(−θ_rad/4π).
Compensates B1 inhomogeneity over ±100% for arbitrary flip θ.
"""
function bb1(θ_deg::Real)::Vector{CompositePulseSegment}
    θ_rad = deg2rad(Float64(θ_deg))
    arg   = clamp(-θ_rad / (4π), -1.0, 1.0)
    φ_deg = rad2deg(acos(arg))
    return [
        CompositePulseSegment(θ_deg,   0.0),
        CompositePulseSegment(180.0,   _w360(φ_deg)),
        CompositePulseSegment(360.0,   _w360(3.0 * φ_deg)),
        CompositePulseSegment(180.0,   _w360(φ_deg)),
    ]
end

# ── SCROFULOUS ───────────────────────────────────────────────────────────────

"""
    scrofulous(θ_deg=90.0) → Vector{CompositePulseSegment}

Cummins SCROFULOUS: `[180_φ, 270_0, 180_−φ]`, φ = atan(2)×180/π ≈ 63.43°.
Calibrated for 90° target; compensates B1 and resonance offset simultaneously.
"""
function scrofulous(θ_deg::Real=90.0)::Vector{CompositePulseSegment}
    abs(Float64(θ_deg) - 90.0) < 0.5 ||
        @warn "scrofulous is designed for 90° target; θ = $(θ_deg)°"
    φ_deg = rad2deg(atan(2.0))   # ≈ 63.43°
    return [
        CompositePulseSegment(180.0,   _w360( φ_deg)),
        CompositePulseSegment(270.0,   0.0),
        CompositePulseSegment(180.0,   _w360(-φ_deg)),
    ]
end

# ── SK1 ──────────────────────────────────────────────────────────────────────

"""
    sk1(θ_deg=90.0) → Vector{CompositePulseSegment}

Brown SK1: `[θ_0, 360_φ, θ_0]`, φ = acos(−1/(2 sin(θ/2))). Valid for θ ≥ 60°.
Short 3-pulse composite; compensates B1 errors with reduced total flip.
"""
function sk1(θ_deg::Real=90.0)::Vector{CompositePulseSegment}
    θ_rad = deg2rad(Float64(θ_deg))
    s = sin(θ_rad / 2)
    abs(s) >= 0.5 ||
        throw(ArgumentError("sk1 requires θ ≥ 60°; got θ = $(θ_deg)°"))
    φ_deg = rad2deg(acos(clamp(-1.0 / (2.0 * s), -1.0, 1.0)))
    return [
        CompositePulseSegment(θ_deg, 0.0),
        CompositePulseSegment(360.0, φ_deg),
        CompositePulseSegment(θ_deg, 0.0),
    ]
end

# ── CORPSE ───────────────────────────────────────────────────────────────────

"""
    corpse(θ_deg) → Vector{CompositePulseSegment}

Cummins CORPSE: `[n₁_0, n₂_180, n₃_0]`; compensates resonance offset for arbitrary θ.
n₁ = 2π+θ/2−α, n₂ = π−2α, n₃ = θ/2−α, α = asin(sin(θ/2)/2).
"""
function corpse(θ_deg::Real)::Vector{CompositePulseSegment}
    θ = deg2rad(Float64(θ_deg))
    α = asin(clamp(sin(θ / 2) / 2.0, -1.0, 1.0))
    n1 = rad2deg(2π + θ / 2 - α)
    n2 = rad2deg(π  - 2α)
    n3 = rad2deg(θ  / 2 - α)
    return [
        CompositePulseSegment(n1, 0.0),
        CompositePulseSegment(n2, 180.0),
        CompositePulseSegment(n3, 0.0),
    ]
end

# ── Short CORPSE ─────────────────────────────────────────────────────────────

"""
    short_corpse(θ_deg) → Vector{CompositePulseSegment}

Short CORPSE: removes 2π wrap from CORPSE segment 1; shorter total flip time.
Sequence `[θ/2−α, 0°], [π−2α, 180°], [θ/2−α, 0°]`; α = asin(sin(θ/2)/2).
"""
function short_corpse(θ_deg::Real)::Vector{CompositePulseSegment}
    θ = deg2rad(Float64(θ_deg))
    α = asin(clamp(sin(θ / 2) / 2.0, -1.0, 1.0))
    n1 = rad2deg(θ / 2 - α)
    n2 = rad2deg(π - 2α)
    return [
        CompositePulseSegment(n1, 0.0),
        CompositePulseSegment(n2, 180.0),
        CompositePulseSegment(n1, 0.0),
    ]
end

# ── F1 ───────────────────────────────────────────────────────────────────────

"""
    f1(θ_bb_deg=90.0) → Vector{CompositePulseSegment}

Freeman composite: `[θ_0, 2θ_90, θ_0]`; effective rotation ≈ 2θ about y-axis.
Building block θ_bb = 90° gives the standard composite 180° (90x·180y·90x).
"""
function f1(θ_bb_deg::Real=90.0)::Vector{CompositePulseSegment}
    θ = Float64(θ_bb_deg)
    return [
        CompositePulseSegment(θ,       0.0),
        CompositePulseSegment(2.0 * θ, 90.0),
        CompositePulseSegment(θ,       0.0),
    ]
end

# ── G1 ───────────────────────────────────────────────────────────────────────

"""
    g1(θ_bb_deg=90.0) → Vector{CompositePulseSegment}

Quadrature companion to f1: `[θ_90, 2θ_0, θ_90]`; effective rotation ≈ 2θ about x-axis.
Building block θ_bb = 90° gives composite 180° (90y·180x·90y).
"""
function g1(θ_bb_deg::Real=90.0)::Vector{CompositePulseSegment}
    θ = Float64(θ_bb_deg)
    return [
        CompositePulseSegment(θ,       90.0),
        CompositePulseSegment(2.0 * θ, 0.0),
        CompositePulseSegment(θ,       90.0),
    ]
end

# ── Concatenated: CORPSE-in-BB1 (Full CinBB) ────────────────────────────────

"""
    corpse_in_bb1(θ_deg) → Vector{CompositePulseSegment}

Full CinBB: every BB1 segment replaced by its CORPSE sub-sequence (12 pulses total).
Combines BB1 B1-robustness with CORPSE offset-compensation.
"""
function corpse_in_bb1(θ_deg::Real)::Vector{CompositePulseSegment}
    θ_rad     = deg2rad(Float64(θ_deg))
    φ_bb1     = rad2deg(acos(clamp(-θ_rad / (4π), -1.0, 1.0)))
    bb1_segs  = [(Float64(θ_deg), 0.0),
                 (180.0,          φ_bb1),
                 (360.0,          3.0 * φ_bb1),
                 (180.0,          φ_bb1)]
    result = CompositePulseSegment[]
    for (flip, gphase) in bb1_segs
        for seg in corpse(flip)
            push!(result, CompositePulseSegment(
                seg.flip_deg,
                _w360(seg.phase_deg + gphase)))
        end
    end
    return result
end

# ── DRAG pulse (Derivative Removal via Adiabatic Gate) ───────────────────────

"""
    drag_pulse(Ω::Vector{Float64}, dt::Float64; β=nothing, anharm_hz=nothing)
    -> AnalyticPulse

Construct the DRAG (Derivative Removal via Adiabatic Gate) correction for a
superconducting transmon qubit.

Given a real envelope Ω(t) (in rad/s), the DRAG correction adds a quadrature
component proportional to the time-derivative of the envelope:

    Ω_x(t) = Ω(t)
    Ω_y(t) = −β × dΩ/dt

where β = 1 / (2 × α) and α = 2π × anharmonicity (rad/s).  This first-order
correction suppresses leakage to the |2⟩ level to O(Ω/α)².

The returned `AnalyticPulse` encodes:
  - `B1`    — Ω_x(t) / max(|Ω|) (normalised amplitude)
  - `phase` — atan(Ω_y, Ω_x)  (instantaneous phase in radians)
  - `t`     — midpoint time vector (seconds)
  - `dt`    — time step

# Arguments
- `Ω`          — Vector{Float64} of drive envelope amplitudes (rad/s), length N_t
- `dt`         — time step (seconds)
- `β`          — DRAG coefficient (s); if `nothing`, `anharm_hz` must be given
- `anharm_hz`  — anharmonicity in Hz (negative for transmon); used to compute
  `β = 1 / (2 × 2π × |anharm_hz|)` when `β` is not supplied

# Returns
[`AnalyticPulse`](@ref) with normalised amplitude and DRAG-corrected phase.

# Example
```julia
# Gaussian envelope, 20 MHz anharmonicity
N = 100; dt = 5e-9
t = (0.5:N) .* dt
Ω = exp.(−((t .- 0.5*N*dt).^2) ./ (2*(0.1*N*dt)^2)) .* 2π .* 20e6
pulse = drag_pulse(Ω, dt; anharm_hz = -200e6)
```
"""
function drag_pulse(Ω         :: Vector{Float64},
                    dt        :: Float64;
                    β         :: Union{Float64, Nothing} = nothing,
                    anharm_hz :: Union{Float64, Nothing} = nothing)::AnalyticPulse
    if isnothing(β)
        isnothing(anharm_hz) && throw(ArgumentError(
            "drag_pulse: supply either β or anharm_hz"))
        β = 1.0 / (2.0 * 2π * abs(anharm_hz))
    end

    N = length(Ω)
    # Numerical derivative via central differences
    dΩ = zeros(N)
    for k in 2:(N-1)
        dΩ[k] = (Ω[k+1] - Ω[k-1]) / (2dt)
    end
    dΩ[1] = (Ω[2]   - Ω[1]) / dt        # forward difference at edges
    dΩ[N] = (Ω[N]   - Ω[N-1]) / dt

    Ωy = -β .* dΩ

    # Amplitude and phase
    amp_raw = sqrt.(Ω.^2 .+ Ωy.^2)
    peak    = maximum(amp_raw)
    peak < eps(Float64) && (peak = 1.0)
    B1    = amp_raw ./ peak
    phase = atan.(Ωy, Ω)
    t_vec = (0.5:N) .* dt

    return AnalyticPulse(collect(t_vec), B1, phase, dt)
end
