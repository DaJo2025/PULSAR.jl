"""
    comparisons/Translator/Emitters/SpinachEmitter.jl

Emit a native Spinach (MATLAB) optimal-control script from a
[`PhysicsAnnotation`](@ref).

The emitted `.m` file is what a competent Spinach user would write for the
same problem — `sys` / `inter` / `bas`, `create`, `basis`,
`operator(..,'Lx',..)`, `state(..,'Lz',..)`, `optimcon`, `fminnewton` —
not a generic matrix-dump shim.

Broadband offset sweeps are encoded as N non-interacting spins with the
offsets written as `inter.zeeman.scalar` (ppm).  This is the Spinach idiom
for offset ensembles.

Capabilities declared via [`SPINACH_CAPABILITIES`](@ref).
"""

using Printf

const SPINACH_CAPABILITIES = SolverCapabilities(
    multi_spin       = true,
    ensemble         = true,
    multi_state_pair = true,
    lindblad         = false,
    multichannel     = true,
    nonuniform_dt    = false,
    heteronuclear    = true,
    csa              = false,
    dipolar          = false,
    j_coupling       = true,
    amplitude_bounds = true,
)

"""
    emit_spinach(ann, workdir; problem_id="Pulsar",
                    guess_seed=nothing) -> (m_script_path, waveform_output_path)

Write a full Spinach .m script to `workdir`.  The optimised waveform is
written to `waveform_output_path` at the end as `n_ctrl` whitespace-separated
columns × `n_t` rows in the **normalised** convention (already divided by
`pwr_levels`).
"""
function emit_spinach(ann::PhysicsAnnotation, workdir::String;
                       problem_id::String="Pulsar",
                       guess_seed::Union{Nothing,Int}=nothing)::Tuple{String,String}
    script_path   = joinpath(workdir, "spinach_run.m")
    waveform_path = joinpath(workdir, "spinach_result.txt")
    seed          = guess_seed === nothing ? ann.guess_seed : guess_seed

    open(script_path, "w") do io
        _emit_spinach_header(io, ann, problem_id, seed)
        is_broadband_single_spin(ann) ?
            _emit_spinach_broadband_ensemble(io, ann) :
            _emit_spinach_generic(io, ann)
        _emit_spinach_controls(io, ann)
        _emit_spinach_state_pairs(io, ann)
        _emit_spinach_optimcon_run(io, ann, waveform_path)
    end

    return script_path, waveform_path
end

# True when the problem is: single-spin system described by an OffsetSweep.
# Such problems are encoded in Spinach as N non-interacting spins, one per
# offset — the textbook broadband-ensemble idiom.
is_broadband_single_spin(ann::PhysicsAnnotation) =
    length(ann.spins) == 1 && ann.sweep !== nothing &&
    isempty(ann.couplings)

# ─── Header ───────────────────────────────────────────────────────────────────

function _emit_spinach_header(io::IO, ann::PhysicsAnnotation,
                               problem_id::String, seed::Int)
    println(io, "% Pulsar benchmark $problem_id — emitted by SpinachEmitter.jl")
    println(io, "% target kind: :$(ann.target.kind)")
    println(io)
    # Cap each MATLAB process's BLAS/compute threads via Pulsar_NCORES. We do NOT
    # cap parpool NumWorkers here — Spinach's parfor may request a worker per
    # ensemble member (e.g. 15+ for broadband), and a tight worker cap throws.
    println(io, "pulsar_ncores = getenv('Pulsar_NCORES');")
    println(io, "if ~isempty(pulsar_ncores)")
    println(io, "    nc = str2double(pulsar_ncores);")
    println(io, "    if ~isnan(nc) && nc > 0")
    println(io, "        maxNumCompThreads(nc);")
    println(io, "    end")
    println(io, "end")
    println(io)
    println(io, "if exist('create', 'file')")
    println(io, "    addpath(genpath(fileparts(fileparts(which('create')))));")
    println(io, "else")
    println(io, "    error('Spinach not found. Add Spinach root to MATLAB path.');")
    println(io, "end")
    println(io, "rng($seed);")
    println(io)
    @printf(io, "sys.magnet = %.6f;\n", ann.b0_tesla)
end

# ─── Spin system: broadband single-spin ensemble ──────────────────────────────

function _emit_spinach_broadband_ensemble(io::IO, ann::PhysicsAnnotation)
    isotope     = ann.spins[1].isotope
    base_off_hz = ann.spins[1].offset_hz
    offsets_hz  = collect(ann.sweep.offsets_hz) .+ base_off_hz

    larmor_hz_isotope = larmor_hz(isotope, ann.b0_tesla)
    offsets_ppm = offsets_hz ./ (larmor_hz_isotope / 1e6)

    println(io)
    println(io, "% Broadband ensemble: $(length(offsets_hz)) non-interacting $isotope spins")
    println(io, "n_spins      = $(length(offsets_hz));")
    print(io,   "offsets_ppm  = [")
    for (k, v) in pairs(offsets_ppm)
        @printf(io, "%.10f", v)
        k < length(offsets_ppm) && print(io, ", ")
    end
    println(io, "];")
    println(io, "sys.isotopes = cell(n_spins, 1);")
    println(io, "for n = 1:n_spins, sys.isotopes{n} = '$isotope'; end")
    println(io, "inter.zeeman.scalar = num2cell(offsets_ppm);")
    println(io)
    println(io, "bas.formalism     = 'sphten-liouv';")
    println(io, "bas.approximation = 'IK-2';")
    println(io, "bas.space_level   = 1;")
    println(io, "bas.connectivity  = 'scalar_couplings';")
    println(io)
    println(io, "spin_system = create(sys, inter);")
    println(io, "spin_system = basis(spin_system, bas);")
end

# ─── Spin system: generic multi-spin / coupled / single on-resonance ──────────

function _emit_spinach_generic(io::IO, ann::PhysicsAnnotation)
    isotopes = join(("'$(s.isotope)'" for s in ann.spins), ", ")
    println(io)
    println(io, "sys.isotopes = {$isotopes};")

    # zeeman offsets (ppm)
    print(io, "inter.zeeman.scalar = {")
    for (k, s) in pairs(ann.spins)
        larmor_mhz = larmor_hz(s.isotope, ann.b0_tesla) / 1e6
        off_ppm    = s.offset_hz / larmor_mhz
        @printf(io, "%.10f", off_ppm)
        k < length(ann.spins) && print(io, ", ")
    end
    println(io, "};")

    # J-couplings
    for c in ann.couplings
        c.kind == :j_isotropic || continue
        @printf(io, "inter.coupling.scalar{%d,%d} = %.6f;\n", c.i, c.j, c.value_hz)
        @printf(io, "inter.coupling.scalar{%d,%d} = %.6f;\n", c.j, c.i, c.value_hz)
    end

    println(io)
    println(io, "bas.formalism     = 'sphten-liouv';")
    println(io, "bas.approximation = 'none';")
    println(io)
    println(io, "spin_system = create(sys, inter);")
    println(io, "spin_system = basis(spin_system, bas);")
end

# ─── Controls block ───────────────────────────────────────────────────────────

function _emit_spinach_controls(io::IO, ann::PhysicsAnnotation)
    println(io)
    op_tokens = String[]
    for c in ann.controls
        letter  = String(c.axis)                # "x" / "y" / "z"
        isotope = ann.spins[c.spin_idx].isotope
        var     = "L$(letter)_c$(length(op_tokens)+1)"
        println(io, "$var = operator(spin_system, 'L$letter', '$isotope');")
        push!(op_tokens, var)
    end
    println(io, "H = hamiltonian(assume(spin_system, 'nmr'));")
    println(io)
    println(io, "control.drifts    = {{H}};")
    println(io, "control.operators = {$(join(op_tokens, ", "))};")
end

# ─── State pairs ──────────────────────────────────────────────────────────────

function _emit_spinach_state_pairs(io::IO, ann::PhysicsAnnotation)
    # For broadband ensembles, Spinach uses collective states (sum over spins).
    broadband = is_broadband_single_spin(ann)

    inits = String[]
    targs = String[]
    for (pair_i, (init_syms, targ_syms)) in enumerate(zip(ann.target.initial_states,
                                                           ann.target.final_states))
        push!(inits, _spinach_state_token(io, ann, init_syms, "init_$pair_i", broadband))
        push!(targs, _spinach_state_token(io, ann, targ_syms, "targ_$pair_i", broadband))
    end
    println(io)
    println(io, "control.rho_init = {$(join(inits, ", "))};")
    println(io, "control.rho_targ = {$(join(targs, ", "))};")
end

# Emit a variable assignment for one composite state (Kronecker/tensor of
# single-spin labels in `syms`) and return its MATLAB variable name.  A
# leading minus, if any, is peeled off and re-applied in the variable form.
function _spinach_state_token(io::IO, ann::PhysicsAnnotation,
                               syms::Vector{Symbol}, varname::String,
                               broadband::Bool)
    if broadband
        # Single-spin problem expressed as N non-interacting spins:
        # use the collective state (sum over spins) in Spinach.
        sym      = syms[1]
        iso      = ann.spins[1].isotope
        sign, op = _split_sign_letter(sym)
        println(io, "$varname = state(spin_system, 'L$op', '$iso');")
        println(io, "$varname = $varname / norm(full($varname), 2);")
        return sign == "-" ? "-$varname" : varname
    end

    # Multi-spin system: Spinach's `state(spin_system, {'Lz',...}, {i,...})`
    # builds the product state directly (one cell entry per participating spin).
    ops   = String[]
    idxs  = String[]
    global_sign = "+"
    for (k, sym) in pairs(syms)
        sign, letter = _split_sign_letter(sym)
        push!(ops, "'L$letter'")
        push!(idxs, string(k))
        sign == "-" && (global_sign = global_sign == "+" ? "-" : "+")
    end
    println(io, "$varname = state(spin_system, {$(join(ops, ", "))}, {$(join(idxs, ", "))});")
    println(io, "$varname = $varname / norm(full($varname), 2);")
    return global_sign == "-" ? "-$varname" : varname
end

function _split_sign_letter(sym::Symbol)
    s = String(sym)
    startswith(s, "m") && return ("-", lowercase(s[3:end]))   # :mIy → ("-", "y")
    return ("+", lowercase(s[end:end]))                       # :Iy  → ("+", "y")
end

# ─── optimcon config + run ────────────────────────────────────────────────────

function _emit_spinach_optimcon_run(io::IO, ann::PhysicsAnnotation,
                                     waveform_path::String)
    dt     = ann.dt_s
    n_t    = ann.n_time_steps
    n_ctrl = length(ann.controls)
    pwr    = 2π * ann.controls[1].pwr_max_hz

    println(io)
    @printf(io, "control.pulse_dt   = repmat(%.12e, 1, %d);\n", dt, n_t)
    @printf(io, "control.pwr_levels = %.10e;\n", pwr)
    println(io, "control.method     = 'lbfgs';")
    println(io, "control.max_iter   = $(ann.max_iter);")
    println(io, "control.plotting   = {};")
    println(io)
    println(io, "spin_system = optimcon(spin_system, control);")
    println(io)
    @printf(io, "guess = 0.05 * randn(%d, %d);\n", n_ctrl, n_t)
    println(io, "guess = max(-1, min(1, guess));")
    println(io)
    println(io, "xy_profile = fminnewton(spin_system, @grape_xy, guess);")
    println(io)

    fmt  = join(fill("%24.16e", n_ctrl), " ")
    args = join(("xy_profile($k, t)" for k in 1:n_ctrl), ", ")
    println(io, "out_file = '$(escape_string(waveform_path))';")
    println(io, "fid_out  = fopen(out_file, 'w');")
    println(io, "for t = 1:$n_t")
    println(io, "    fprintf(fid_out, '$fmt\\n', $args);")
    println(io, "end")
    println(io, "fclose(fid_out);")
    println(io, "exit;")
end
