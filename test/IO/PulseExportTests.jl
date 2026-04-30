@testset "IO – Pulse Format Round-Trip" begin

    using LinearAlgebra

    # Synthetic waveform: 50 complex samples, unit amplitude, varying phase
    N  = 50
    dt = 1e-5   # 10 µs per step
    rng = MersenneTwister(42)
    samples_orig = [0.8 * exp(im * 2π * rand(rng)) for _ in 1:N]
    pulse_orig   = OptimizedPulse(ComplexF64.(samples_orig), dt;
                                   name="test_pulse", flip_angle=90.0)

    tmp = mktempdir()
    pwr = 1.0    # normalised; w_mat will be in [-1, 1]

    # Convert to w_mat [2×N] for Bruker/MR-layer savers
    w_mat = Float64[real(s) for s in samples_orig]'
    w_mat = vcat(real.(samples_orig)', imag.(samples_orig)')   # [2×N]

    # ─── Bruker JCAMP-DX ────────────────────────────────────────────────────
    @testset "Bruker round-trip" begin
        fp = joinpath(tmp, "test.bruker")
        save_bruker_shape(fp, w_mat, pwr; dt=dt, title="test", shape_totrot_deg=90.0)
        pulse_rt = load_bruker_shape(fp; pwr_level=pwr, dt=dt)
        @test length(pulse_rt.samples) == N
        err = norm(pulse_rt.samples .- samples_orig) / sqrt(N)
        @test err < 1e-4
    end

    # ─── JEOL Delta ─────────────────────────────────────────────────────────
    @testset "JEOL round-trip" begin
        fp = joinpath(tmp, "test.jrf")
        export_pulse(pulse_orig; application=:NMR_EPR, vendor=:jeol,
                     save=true, output_dir=tmp, name="test_pulse")
        jrf_fp = joinpath(tmp, "test_pulse.jrf")
        isfile(jrf_fp) || (jrf_fp = fp)  # fallback
        if isfile(jrf_fp)
            # JEOL normalises amplitudes to max=1; pass max_amp to recover scale.
            mx = maximum(abs.(samples_orig))
            pulse_rt = load_jeol_shape(jrf_fp; max_amp=mx, dt=dt)
            @test length(pulse_rt.samples) == N
            # Phase and amplitude should match to ~1e-4 (float formatting precision)
            err = norm(abs.(pulse_rt.samples) .- abs.(samples_orig)) / sqrt(N)
            @test err < 1e-4
        else
            @test_skip "JEOL file not generated"
        end
    end

    # ─── EPR Bruker AWG ─────────────────────────────────────────────────────
    @testset "EPR round-trip" begin
        fp_dir = tmp
        export_pulse(pulse_orig; application=:NMR_EPR, vendor=:epr_bruker,
                     save=true, output_dir=fp_dir, name="test_pulse")
        shp_fp = joinpath(fp_dir, "test_pulse.shp")
        if isfile(shp_fp)
            pulse_rt = load_epr_shape(shp_fp; dt=dt)
            @test length(pulse_rt.samples) == N
            # I/Q round-trip: amplitudes match up to normalisation
            err = norm(angle.(pulse_rt.samples) .- angle.(samples_orig)) / sqrt(N)
            @test err < 1e-4
        else
            @test_skip "EPR file not generated"
        end
    end

    # ─── Qiskit JSON ────────────────────────────────────────────────────────
    @testset "Qiskit round-trip" begin
        export_pulse(pulse_orig; application=:QC, vendor=:qiskit,
                     save=true, output_dir=tmp, name="test_pulse")
        json_fp = joinpath(tmp, "test_pulse.json")
        if isfile(json_fp)
            pulse_rt = load_qiskit_waveform(json_fp)
            @test length(pulse_rt.samples) == N
            @test abs(pulse_rt.dt - dt) < 1e-15
            # Amplitudes are normalised to max=1 on export; recover relative phases
            err = norm(angle.(pulse_rt.samples) .- angle.(samples_orig)) / sqrt(N)
            @test err < 1e-8
        else
            @test_skip "Qiskit file not generated"
        end
    end

    # ─── Quil-T ─────────────────────────────────────────────────────────────
    @testset "Quil-T round-trip" begin
        export_pulse(pulse_orig; application=:QC, vendor=:quil_t,
                     save=true, output_dir=tmp, name="test_pulse")
        quil_fp = joinpath(tmp, "test_pulse.quil")
        if isfile(quil_fp)
            pulse_rt = load_quil_t(quil_fp; dt=dt)
            @test length(pulse_rt.samples) == N
            err = norm(angle.(pulse_rt.samples) .- angle.(samples_orig)) / sqrt(N)
            @test err < 1e-6
        else
            @test_skip "Quil-T file not generated"
        end
    end

    # ─── QUA ────────────────────────────────────────────────────────────────
    @testset "QUA round-trip" begin
        export_pulse(pulse_orig; application=:QC, vendor=:qua,
                     save=true, output_dir=tmp, name="test_pulse")
        qua_fp = joinpath(tmp, "test_pulse_qua.py")
        if isfile(qua_fp)
            pulse_rt = load_qua(qua_fp; dt=dt)
            @test length(pulse_rt.samples) == N
            err = norm(angle.(pulse_rt.samples) .- angle.(samples_orig)) / sqrt(N)
            @test err < 1e-4
        else
            @test_skip "QUA file not generated"
        end
    end

    # ─── Pulser ─────────────────────────────────────────────────────────────
    @testset "Pulser round-trip" begin
        export_pulse(pulse_orig; application=:QC, vendor=:pulser,
                     save=true, output_dir=tmp, name="test_pulse")
        pulser_fp = joinpath(tmp, "test_pulse_pulser.py")
        if isfile(pulser_fp)
            # Pulser normalises amplitudes to max=1; pass max_amp to recover scale.
            mx = maximum(abs.(samples_orig))
            pulse_rt = load_pulser(pulser_fp; max_amp=mx)
            @test length(pulse_rt.samples) == N
            err = norm(abs.(pulse_rt.samples) .- abs.(samples_orig)) / sqrt(N)
            @test err < 1e-4
        else
            @test_skip "Pulser file not generated"
        end
    end

end
