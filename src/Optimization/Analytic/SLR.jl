# ============================================================================
# Analytic/SLR.jl — Shinnar-Le Roux (SLR) pulse design
# ============================================================================

# ── Internal helpers ─────────────────────────────────────────────────────────

# Modified Bessel function I₀ via series (accurate to < 1e-12 for |x| ≤ 3.75,
# then asymptotic form; no special-functions package needed).
function _slr_bessel_i0(x::Float64)::Float64
    ax = abs(x)
    if ax < 3.75
        t  = (x / 3.75)^2
        return 1.0 + t*(3.5156229 + t*(3.0899424 + t*(1.2067492 +
               t*(0.2659732 + t*(0.0360768 + t*0.0045813)))))
    else
        t  = 3.75 / ax
        return (exp(ax) / sqrt(ax)) *
               (0.39894228 + t*(0.01328592 + t*(0.00225319 +
               t*(-0.00157565 + t*(0.00916281 + t*(-0.02057706 +
               t*(0.02635537 + t*(-0.01647633 + t*0.00392377))))))))
    end
end

# Kaiser window for index n ∈ 0:N-1
function _slr_kaiser(n::Int, N::Int, β::Float64)::Float64
    halfN  = (N - 1) / 2.0
    ratio  = 1.0 - ((n - halfN) / halfN)^2
    return _slr_bessel_i0(β * sqrt(max(ratio, 0.0))) / _slr_bessel_i0(β)
end

# Conjugate-reverse a polynomial array p = [p₀, p₁, ..., pₙ]
_slr_crev(p::Vector{ComplexF64}) = conj.(reverse(p))

# DFT of a complex vector (O(N²), reused from STA helpers via module scope)
function _slr_dft(x::Vector{ComplexF64})::Vector{ComplexF64}
    N = length(x)
    X = Vector{ComplexF64}(undef, N)
    @inbounds for k in 0:N-1
        s  = ComplexF64(0.0)
        w  = exp(-2π * im * k / N)
        wn = ComplexF64(1.0)
        for n in 1:N; s += x[n] * wn; wn *= w; end
        X[k+1] = s
    end
    return X
end

function _slr_idft(X::Vector{ComplexF64})::Vector{ComplexF64}
    N = length(X)
    x = Vector{ComplexF64}(undef, N)
    @inbounds for n in 0:N-1
        s  = ComplexF64(0.0)
        w  = exp(2π * im * n / N)
        wk = ComplexF64(1.0)
        for k in 1:N; s += X[k] * wk; wk *= w; end
        x[n+1] = s / N
    end
    return x
end

# Minimum-phase spectral factorisation via complex cepstrum:
# returns a[0:N-1] such that |A(ω)|² ≈ 1 − |B(ω)|²
function _slr_min_phase_a(b::Vector{Float64})::Vector{Float64}
    N = length(b)
    M = max(512, 1 << (ceil(Int, log2(4N))))   # next power of 2 ≥ 4N

    B_pad = zeros(ComplexF64, M)
    B_pad[1:N] .= b

    Bw  = _slr_dft(B_pad)
    Pw  = max.(1.0 .- abs2.(Bw), 1e-15)        # |A|² = 1 − |B|²

    # Log cepstrum
    cep = real.(_slr_idft(ComplexF64.(log.(Pw))))

    # Causal (minimum-phase) half
    c_mp = zeros(M)
    c_mp[1]           = cep[1]
    c_mp[2:M÷2]       .= 2.0 .* cep[2:M÷2]
    c_mp[M÷2+1]       = cep[M÷2+1]             # Nyquist — keep as-is

    Aw   = exp.(_slr_dft(ComplexF64.(c_mp)))
    a_td = real.(_slr_idft(Aw))
    return a_td[1:N]
end

# Inverse SLR hard-pulse recursion.
# Given CK polynomials A (length N) and B (length N) in ascending-power order,
# returns the RF pulse as complex flip-angle array (radians × exp(iφ)).
function _slr_inverse(A_in::Vector{Float64}, B_in::Vector{Float64})::Vector{ComplexF64}
    N  = length(A_in)
    RF = zeros(ComplexF64, N)

    A_cur = ComplexF64.(A_in)
    B_cur = ComplexF64.(B_in)

    for k = N:-1:1
        a0  = A_cur[1];  b0 = B_cur[1]
        nrm = sqrt(abs2(a0) + abs2(b0))
        if nrm < 1e-14
            break
        end
        αk  = a0 / nrm;  βk = b0 / nrm
        θk  = 2.0 * atan(abs(βk), abs(αk))
        φk  = angle(βk) - angle(αk)
        RF[k] = θk * exp(im * φk)

        k == 1 && break

        # Recover A_{k-1} (with trailing-0 padding) and B̃_{k-1}
        A_pad = conj(αk) .* A_cur .+ conj(βk) .* B_cur     # length k; last entry ≈ 0
        zBt   = -βk .* A_cur .+ αk .* B_cur                 # length k; first entry ≈ 0

        A_cur = A_pad[1:end-1]                               # drop trailing 0
        B_cur = conj.(reverse(zBt[2:end]))                   # drop leading 0, then crev
    end
    return RF
end

# ── SLR 1D ───────────────────────────────────────────────────────────────────

"""
    slr_1d(flip_deg, tbw, duration_s, N_ts; filter_type, pass_ripple, stop_ripple) → AnalyticPulse

Shinnar-Le Roux 1D selective pulse: Kaiser-windowed FIR → spectral factorisation → inverse SLR.
`filter_type` ∈ {:excitation, :inversion, :saturation, :refocusing}; ripples in (0, 1).
"""
function slr_1d(
    flip_deg     :: Real,
    tbw          :: Real,
    duration_s   :: Real,
    N_ts         :: Int;
    filter_type  :: Symbol  = :excitation,
    pass_ripple  :: Float64 = 0.01,
    stop_ripple  :: Float64 = 0.01,
)::AnalyticPulse
    flip_rad  = deg2rad(Float64(flip_deg))
    N         = N_ts % 2 == 0 ? N_ts + 1 : N_ts   # odd length for symmetric FIR
    dt        = Float64(duration_s) / N

    # ── Kaiser window parameter from worst-case ripple ──────────────────────
    A_att = -20.0 * log10(min(pass_ripple, stop_ripple))
    β_k   = A_att > 50.0  ? 0.1102 * (A_att - 8.7) :
            A_att >= 21.0 ? 0.5842 * (A_att - 21)^0.4 + 0.07886 * (A_att - 21) : 0.0

    # ── Windowed-sinc FIR (linear phase, symmetric) ─────────────────────────
    half_N = (N - 1) / 2.0
    fc     = Float64(tbw) / N              # normalised cutoff (0 < fc < 0.5)
    b_arr  = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        n       = i - 1 - half_N           # centre at 0
        sinc_v  = abs(n) < 1e-9 ? 2.0 * fc : sin(2π * fc * n) / (π * n)
        b_arr[i] = sinc_v * _slr_kaiser(i - 1, N, β_k)
    end

    # ── Scale B polynomial for target flip and filter type ───────────────────
    b_scale = filter_type == :inversion  ? sin(flip_rad / 2)^2 :
              filter_type == :refocusing ? sin(flip_rad / 2)    :
              sin(flip_rad / 2)          # excitation / saturation
    b_arr  .*= b_scale / (maximum(abs.(b_arr)) + 1e-30)

    # ── Minimum-phase A polynomial ───────────────────────────────────────────
    a_arr = _slr_min_phase_a(b_arr)
    # Normalise so that |A(0)| + |B(0)| satisfies CK constraint at DC
    nrm = sqrt(abs2(a_arr[1]) + abs2(b_arr[1]))
    if nrm > 1e-14
        a_arr ./= nrm
        b_arr ./= nrm
    end

    # ── Inverse SLR transform → complex RF array (rad) ──────────────────────
    RF = _slr_inverse(a_arr, b_arr)

    # Convert hard-pulse flip angles (rad) to amplitude (rad/s) and phase
    B1_amp   = abs.(RF) ./ dt
    B1_phase = angle.(RF)
    pk       = maximum(B1_amp)
    pk < 1e-30 && throw(ErrorException("SLR produced a zero RF pulse; check inputs"))
    B1_norm  = B1_amp ./ pk

    t = collect(range(0.0, Float64(duration_s) - dt; length=N))
    return AnalyticPulse(t, B1_norm, B1_phase, dt)
end
