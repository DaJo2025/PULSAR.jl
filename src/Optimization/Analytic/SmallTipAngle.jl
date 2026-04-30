# ============================================================================
# Analytic/SmallTipAngle.jl — Small-tip-angle Fourier pulse design
#
# Reference: Pauly, Nishimura & Macovski (1989), "A k-space analysis of
# small-tip-angle excitation," J. Magn. Reson. 81, 43–56.
#
# NOTE ON NAMING:
#   In the MRI / pulse-design community, "STA" unambiguously means
#   *Small-Tip-Angle*.  In the quantum-optimal-control community, "STA"
#   usually means *Shortcut-to-Adiabaticity* (Berry 2009; Chen 2010).
#   This file implements the MRI meaning.  The legacy names
#   `sta_fourier_1d` / `STA.jl` are retained as deprecated aliases.
# ============================================================================

# ── Internal DFT (no external deps, O(N²), suitable for N ≤ 512) ────────────

function _sta_dft(x::AbstractVector{ComplexF64})::Vector{ComplexF64}
    N = length(x)
    X = Vector{ComplexF64}(undef, N)
    @inbounds for k in 0:N-1
        s = ComplexF64(0.0)
        w = exp(-2π * im * k / N)
        wn = ComplexF64(1.0)
        for n in 1:N
            s  += x[n] * wn
            wn *= w
        end
        X[k+1] = s
    end
    return X
end

function _sta_idft(X::AbstractVector{ComplexF64})::Vector{ComplexF64}
    N = length(X)
    x = Vector{ComplexF64}(undef, N)
    @inbounds for n in 0:N-1
        s = ComplexF64(0.0)
        w = exp(2π * im * n / N)
        wk = ComplexF64(1.0)
        for k in 1:N
            s  += X[k] * wk
            wk *= w
        end
        x[n+1] = s / N
    end
    return x
end

# Interpolate complex vector `v` defined on grid `src` onto grid `dst`
# using nearest-neighbour (simple, avoids Interpolations.jl dependency).
function _sta_interp(v::AbstractVector{<:Complex},
                     src::AbstractVector{<:Real},
                     dst::AbstractVector{<:Real})::Vector{ComplexF64}
    out = zeros(ComplexF64, length(dst))
    for (i, d) in enumerate(dst)
        _, j = findmin(x -> abs(x - d), src)
        out[i] = v[j]
    end
    return out
end

# ── Small-Tip-Angle Fourier 1D ───────────────────────────────────────────────

"""
    small_tip_angle_fourier_1d(profile, f_axis_hz, duration_s; N_ts=256)
        → AnalyticPulse

Small-tip-angle (STA) 1D slice-selective pulse design
(Pauly–Nishimura–Macovski 1989).  In the small-flip-angle regime
(|flip| ≲ 30°), the transverse magnetisation M_xy(ω) is approximately
the Fourier transform of the RF envelope B1(t).  This routine inverts
that relationship: it IDFT's the desired frequency profile, applies a
Hamming window to suppress Gibbs ringing, and normalises to unit peak.

!!! note "`sta_` refers to Small-Tip-Angle, NOT Shortcut-to-Adiabaticity"
    In the MRI community, STA unambiguously means Small-Tip-Angle.  In
    quantum control, STA usually means Shortcut-to-Adiabaticity (Berry
    2009; Chen 2010) — a distinct concept not implemented here.

`profile` and `f_axis_hz` must have equal length.
"""
function small_tip_angle_fourier_1d(
    profile    :: AbstractVector{<:Number},
    f_axis_hz  :: AbstractVector{<:Real},
    duration_s :: Real;
    N_ts       :: Int = 256,
)::AnalyticPulse
    length(profile) == length(f_axis_hz) ||
        throw(ArgumentError("profile and f_axis_hz must have the same length"))
    N_ts > 0 || throw(ArgumentError("N_ts must be positive"))

    dt = Float64(duration_s) / N_ts

    # ── Interpolate profile onto N_ts evenly-spaced frequency bins ───────────
    f_min  = minimum(f_axis_hz)
    f_max  = maximum(f_axis_hz)
    f_grid = range(f_min, f_max; length=N_ts)
    P_grid = _sta_interp(ComplexF64.(profile), collect(f_axis_hz), collect(f_grid))

    # Centre the spectrum (shift zero-frequency to index 1 for IDFT convention)
    # IDFT of a spectrum centred at DC gives a time-centred pulse.
    P_shift = circshift(P_grid, N_ts ÷ 2)

    # ── IDFT → time-domain envelope ──────────────────────────────────────────
    b1_complex = _sta_idft(P_shift)

    # ── Hamming window to reduce Gibbs ringing ───────────────────────────────
    hamming = [0.54 - 0.46 * cos(2π * n / (N_ts - 1)) for n in 0:N_ts-1]
    b1_win  = b1_complex .* hamming

    # ── Normalise to peak amplitude = 1 ─────────────────────────────────────
    pk = maximum(abs.(b1_win))
    pk < 1e-30 && throw(ArgumentError("Profile is effectively zero; cannot normalise"))
    b1_norm = b1_win ./ pk

    t = collect(range(0.0, Float64(duration_s) - dt; length=N_ts))
    return AnalyticPulse(t, abs.(b1_norm), angle.(b1_norm), dt)
end

# ---------------------------------------------------------------------------
# Deprecation alias: sta_fourier_1d → small_tip_angle_fourier_1d
# ---------------------------------------------------------------------------
const _sta_fourier_1d_warned = Ref(false)

"""
    sta_fourier_1d(args...; kwargs...)

Deprecated alias for [`small_tip_angle_fourier_1d`](@ref).  The `sta_`
abbreviation collides with the quantum-control meaning
*Shortcut-to-Adiabaticity*, which is a distinct method.  Use the
unambiguous name in new code.
"""
function sta_fourier_1d(args...; kwargs...)
    if !_sta_fourier_1d_warned[]
        @warn "`sta_fourier_1d` has been renamed to " *
              "`small_tip_angle_fourier_1d`. `sta_` here means *Small-Tip-" *
              "Angle* (Pauly et al. 1989), not Shortcut-to-Adiabaticity." maxlog=1
        _sta_fourier_1d_warned[] = true
    end
    return small_tip_angle_fourier_1d(args...; kwargs...)
end
