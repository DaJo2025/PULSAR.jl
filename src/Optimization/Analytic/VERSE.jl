# ============================================================================
# Analytic/VERSE.jl — Variable-Rate Selective Excitation
# ============================================================================
# All three functions return (t_v, B1_v, grad_v) where:
#   t_v    :: Vector{Float64} — cumulative time axis (s)
#   B1_v   :: Vector{ComplexF64} — VERSE RF waveform (same units as input)
#   grad_v :: Vector{Float64}    — VERSE gradient waveform (same units as input)
# The excitation profile is preserved under the time reparameterisation.
# ============================================================================

# ── Internal: build cumulative time axis from per-sample dτ ─────────────────
function _verse_t_axis(dt_vec::Vector{Float64})::Vector{Float64}
    t = Vector{Float64}(undef, length(dt_vec))
    t[1] = 0.0
    @inbounds for i in 2:length(dt_vec)
        t[i] = t[i-1] + dt_vec[i-1]
    end
    return t
end

# ── VERSE ─────────────────────────────────────────────────────────────────────

"""
    verse(B1, grad, dt; slew_rate=Inf, grad_max=Inf, B1_max=maximum(abs,B1))

Amplitude-limited VERSE: stretch time where |B1| exceeds B1_max; preserve profile.
Slew-rate limit applied by smoothing the per-sample scale factor; returns (t_v, B1_v, grad_v).
"""
function verse(
    B1        :: AbstractVector{<:Number},
    grad      :: AbstractVector{<:Real},
    dt        :: Real;
    slew_rate :: Real   = Inf,
    grad_max  :: Real   = Inf,
    B1_max    :: Real   = maximum(abs, B1),
)::Tuple{Vector{Float64}, Vector{ComplexF64}, Vector{Float64}}
    N      = length(B1)
    length(grad) == N || throw(ArgumentError("B1 and grad must have equal length"))
    B1_max > 0 || throw(ArgumentError("B1_max must be positive"))
    dt_f   = Float64(dt)

    B1_v   = Vector{ComplexF64}(undef, N)
    G_v    = Vector{Float64}(undef, N)
    dt_vec = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
        amp  = abs(B1[i])
        # α = local time compression factor (0 < α ≤ 1)
        α    = amp > Float64(B1_max) ? Float64(B1_max) / amp : 1.0
        # Stretched timestep, rescaled B1 and gradient
        dt_vec[i] = dt_f / α
        B1_v[i]   = ComplexF64(B1[i]) * α
        G_v[i]    = Float64(grad[i]) * α
    end

    # Enforce gradient amplitude constraint by re-clamping G and adjusting α
    if isfinite(grad_max)
        gmax = Float64(grad_max)
        @inbounds for i in 1:N
            if abs(G_v[i]) > gmax
                scale    = gmax / abs(G_v[i])
                G_v[i]  *= scale
                B1_v[i] *= scale
                dt_vec[i] /= scale
            end
        end
    end

    # Enforce slew-rate constraint (smooth gradient transitions)
    if isfinite(slew_rate)
        sr = Float64(slew_rate)
        @inbounds for i in 2:N
            avg_dt = (dt_vec[i-1] + dt_vec[i]) / 2.0
            dG     = G_v[i] - G_v[i-1]
            if abs(dG) / avg_dt > sr
                G_v[i] = G_v[i-1] + sign(dG) * sr * avg_dt
                # Recompute corresponding B1 and dt to keep profile consistent
                if abs(G_v[i-1]) > 1e-30
                    α_new    = abs(G_v[i]) / abs(Float64(grad[i]))
                    B1_v[i]  = ComplexF64(B1[i]) * α_new
                    dt_vec[i] = dt_f / max(α_new, 1e-30)
                end
            end
        end
    end

    return _verse_t_axis(dt_vec), B1_v, G_v
end

# ── VERSE min-time ─────────────────────────────────────────────────────────

"""
    verse_min_time(B1, grad, dt; slew_rate, grad_max, B1_max) → (t_v, B1_v, grad_v)

VERSE with minimum-time gradient: compresses plateau (constant-gradient) regions after amplitude VERSE.
Slew-rate and gradient-amplitude constraints are enforced; returns (t_v, B1_v, grad_v).
"""
function verse_min_time(
    B1        :: AbstractVector{<:Number},
    grad      :: AbstractVector{<:Real},
    dt        :: Real;
    slew_rate :: Real   = Inf,
    grad_max  :: Real   = Inf,
    B1_max    :: Real   = maximum(abs, B1),
)::Tuple{Vector{Float64}, Vector{ComplexF64}, Vector{Float64}}
    # Run standard VERSE first
    t_v, B1_v, G_v = verse(B1, grad, dt;
                            slew_rate=slew_rate, grad_max=grad_max, B1_max=B1_max)

    N      = length(G_v)
    dt_vec = diff([0.0; t_v])   # per-sample durations from cumulative axis

    # Compress plateau regions: where consecutive |G| values are equal,
    # the step can be shortened to the slew-rate-limited minimum.
    if isfinite(slew_rate) && isfinite(grad_max)
        sr   = Float64(slew_rate)
        gmax = Float64(grad_max)
        @inbounds for i in 2:N-1
            dG_prev = abs(G_v[i] - G_v[i-1])
            dG_next = abs(G_v[i+1] - G_v[i])
            # Minimum time imposed by slew rate to reach G[i] from both neighbours
            dt_min = max(dG_prev, dG_next) / sr
            if dt_vec[i] > dt_min && dt_min > 1e-12
                # Scale B1 so that flip-angle per segment is preserved
                scale      = dt_min / dt_vec[i]
                B1_v[i]   /= scale          # larger B1 for shorter dt → same flip
                G_v[i]    /= scale
                dt_vec[i]  = dt_min
            end
        end
    end

    t_out = _verse_t_axis(dt_vec)
    return t_out, B1_v, G_v
end

# ── VERSE acoustic noise ───────────────────────────────────────────────────

"""
    verse_acoustic_noise(B1, grad, dt; freq_notch_hz, bw_notch_hz, slew_rate, grad_max, B1_max)

VERSE with acoustic notch: attenuates gradient spectral power near `freq_notch_hz` ± `bw_notch_hz/2`.
Applies a frequency-domain notch to the gradient waveform, then runs standard VERSE to restore profile.
"""
function verse_acoustic_noise(
    B1           :: AbstractVector{<:Number},
    grad         :: AbstractVector{<:Real},
    dt           :: Real;
    freq_notch_hz :: Real  = 1000.0,
    bw_notch_hz   :: Real  = 200.0,
    slew_rate     :: Real  = Inf,
    grad_max      :: Real  = Inf,
    B1_max        :: Real  = maximum(abs, B1),
)::Tuple{Vector{Float64}, Vector{ComplexF64}, Vector{Float64}}
    N    = length(grad)
    dt_f = Float64(dt)

    # ── Apply frequency-domain notch to gradient using DFT ──────────────────
    # Build notch-weight vector in frequency domain
    f_nyq  = 0.5 / dt_f                           # Nyquist frequency (Hz)
    f_res  = 1.0 / (N * dt_f)                     # frequency resolution (Hz)
    f0     = Float64(freq_notch_hz)
    bw     = Float64(bw_notch_hz)

    # DFT of gradient
    G_c = ComplexF64.(grad)
    Gw  = zeros(ComplexF64, N)
    @inbounds for k in 0:N-1
        s  = ComplexF64(0.0)
        w  = exp(-2π * im * k / N)
        wn = ComplexF64(1.0)
        for n in 1:N; s += G_c[n] * wn; wn *= w; end
        Gw[k+1] = s
    end

    # Notch weight: 1 − raised-cosine window centred at ±f0
    # DFT bin k has frequency k*f_res. For real signals bins k > N/2 represent
    # negative frequencies; fold symmetrically into [−f_nyq, +f_nyq].
    weight = ones(Float64, N)
    @inbounds for k in 0:N-1
        fk_raw = k * f_res
        # Symmetric fold: map to the nearest alias in [−f_nyq, +f_nyq]
        fk = fk_raw - round(fk_raw / (2 * f_nyq)) * (2 * f_nyq)
        for f_centre in (f0, -f0)
            Δf = abs(fk - f_centre)
            if Δf < bw / 2
                weight[k+1] = min(weight[k+1],
                    0.5 * (1.0 - cos(π * (Δf - bw/2) / (bw/2))))
            end
        end
    end
    Gw .*= weight

    # IDFT → notch-filtered gradient
    grad_notch = Vector{Float64}(undef, N)
    @inbounds for n in 0:N-1
        s  = ComplexF64(0.0)
        w  = exp(2π * im * n / N)
        wk = ComplexF64(1.0)
        for k in 1:N; s += Gw[k] * wk; wk *= w; end
        grad_notch[n+1] = real(s) / N
    end

    # ── Run VERSE on the notch-filtered gradient ─────────────────────────────
    return verse(B1, grad_notch, dt_f;
                 slew_rate=slew_rate, grad_max=grad_max, B1_max=B1_max)
end
