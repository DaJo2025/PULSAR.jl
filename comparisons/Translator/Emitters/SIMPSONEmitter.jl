"""
    comparisons/Translator/Emitters/SIMPSONEmitter.jl

Emit a native SIMPSON optimal-control input file (Tcl) from a `PhysicsAnnotation`.

The emitted script is what a competent SIMPSON user would write for the same
problem — `spinsys {}`, `par {}`, `pulseq`, `target_function`, `gradient`,
`main` — not a generic matrix-dump shim.

Per-offset weighting: uniform (SIMPSON's native accumulator averages).

Capabilities (declared in [`SIMPSON_CAPABILITIES`](@ref)):
- `multi_state_pair = true`  — looped via `par(start_operator)` / `par(detect_operator)`
- `lindblad = false`         — SIMPSON OC is closed-system only
- `multichannel = false`     — emitter is single-channel (1H or 13C)
- `heteronuclear = false`    — multi-channel emission not yet implemented
- `csa = false`              — CSA + powder averaging not emitted here
- `dipolar = false`          — same reason as CSA
"""

using Printf

const SIMPSON_CAPABILITIES = SolverCapabilities(
    multi_spin       = false,   # single-spin only for this emitter (no J/dipolar yet)
    ensemble         = true,
    multi_state_pair = true,
    lindblad         = false,
    multichannel     = false,
    nonuniform_dt    = false,
    heteronuclear    = false,
    csa              = false,
    dipolar          = false,
    j_coupling       = false,
    amplitude_bounds = true,
)

"""
    emit_simpson(ann, workdir; problem_id="BM??") -> (script_path, shape_output_path)

Write a complete SIMPSON `.in` script to `workdir`.  The shape is written to
`shape_output_path` at the end of `main` as two columns `Bx By` in Hz, one row
per time step.  Waveform units: `:hz` (driver normalises by `pwr_max_hz`).
"""
function emit_simpson(ann::PhysicsAnnotation, workdir::String;
                       problem_id::String="Pulsar")::Tuple{String,String}
    script_path       = joinpath(workdir, "grape.in")
    shape_output_path = joinpath(workdir, "shape.txt")

    open(script_path, "w") do io
        _emit_simpson_header(io, ann, problem_id, shape_output_path)
        _emit_simpson_spinsys(io, ann)
        _emit_simpson_par(io, ann)
        _emit_simpson_offset_list(io, ann)
        _emit_simpson_pulseq(io)
        _emit_simpson_target_and_gradient(io, ann)
        _emit_simpson_main(io, ann)
    end

    return script_path, shape_output_path
end

# ─── Header ───────────────────────────────────────────────────────────────────

function _emit_simpson_header(io::IO, ann::PhysicsAnnotation,
                               problem_id::String, shape_out::String)
    println(io, "# Pulsar benchmark $problem_id — emitted by SIMPSONEmitter.jl")
    println(io)
    println(io, "set Pulsar_OUT \"$(escape_string(shape_out))\"")
    println(io)
end

# ─── spinsys ──────────────────────────────────────────────────────────────────

function _emit_simpson_spinsys(io::IO, ann::PhysicsAnnotation)
    n = length(ann.spins)
    isotopes = join((s.isotope for s in ann.spins), " ")
    println(io, "spinsys {")
    println(io, "    channels $(ann.spins[1].isotope)")
    println(io, "    nuclei   $isotopes")
    for (k, s) in pairs(ann.spins)
        # shift <spin> iso aniso eta alpha beta gamma
        println(io, "    shift $k $(s.offset_hz) 0 0 0 0 0")
    end
    println(io, "}")
    println(io)
end

# ─── par ──────────────────────────────────────────────────────────────────────

function _emit_simpson_par(io::IO, ann::PhysicsAnnotation)
    n_t          = ann.n_time_steps
    T_us         = ann.total_time_s * 1e6
    # proton_frequency sets B0 (Hz on 1H); if first isotope isn't 1H, scale.
    proton_freq  = larmor_hz("1H", ann.b0_tesla)
    # First state pair used for the initial par(start_operator)/par(detect_operator)
    # (target_function may override these for multi-pair objectives).
    init_sym     = ann.target.initial_states[1][1]
    targ_sym     = ann.target.final_states[1][1]
    start_tok    = state_symbol_to(init_sym, :simpson, 1)
    detect_tok   = state_symbol_to(targ_sym, :simpson, 1)

    println(io, "par {")
    println(io, "    name              pulsar_run")
    @printf(io, "    proton_frequency  %.6e\n", proton_freq)
    println(io, "    crystal_file      alpha0beta0")
    println(io, "    gamma_angles      1")
    println(io, "    spin_rate         0")
    println(io, "    sw                1000")
    println(io, "    conjugate_fid     false")
    println(io)
    println(io, "    start_operator    $start_tok")
    println(io, "    detect_operator   $detect_tok")
    println(io)
    println(io, "    variable NOC      $n_t")
    @printf(io, "    variable duration %.4f\n", T_us)
    println(io)
    println(io, "    oc_method         L-BFGS")
    println(io, "    oc_max_iter       $(ann.max_iter)")
    println(io, "}")
    println(io)
end

# ─── Offset list ──────────────────────────────────────────────────────────────

function _emit_simpson_offset_list(io::IO, ann::PhysicsAnnotation)
    println(io, "set lims {}")
    if ann.sweep === nothing
        println(io, "lappend lims 0")
    else
        for Δf in ann.sweep.offsets_hz
            @printf(io, "lappend lims %.6f\n", Δf)
        end
    end
    println(io)
end

# ─── pulseq ───────────────────────────────────────────────────────────────────

function _emit_simpson_pulseq(io::IO)
    println(io, "proc pulseq {} {")
    println(io, "    global par rfsh")
    println(io, "    reset")
    println(io, "    pulse_shaped \$par(duration) \$rfsh")
    println(io, "    oc_acq_hermit")
    println(io, "}")
    println(io)
end

# ─── target_function + gradient ───────────────────────────────────────────────

function _emit_simpson_target_and_gradient(io::IO, ann::PhysicsAnnotation)
    pairs_ = [(state_symbol_to(i[1], :simpson, 1),
                "{$(state_symbol_to(t[1], :simpson, 1))}")
               for (i, t) in zip(ann.target.initial_states,
                                  ann.target.final_states)]
    npairs = length(pairs_)

    # target_function — returns average fidelity over offsets × pairs
    println(io, "proc target_function {} {")
    println(io, "    global par lims")
    println(io, "    set par(np) 1")
    println(io, "    set Res 0.0")
    println(io, "    set noff [llength \$lims]")
    println(io, "    foreach shft \$lims {")
    for (s, d) in pairs_
        println(io, "        set par(start_operator)  $s")
        println(io, "        set par(detect_operator) $d")
        println(io, "        set f [fsimpson [list [list shift_1_iso \$shft]]]")
        println(io, "        set Res [expr {\$Res + [findex \$f 1 -re]}]")
        println(io, "        funload \$f")
    end
    println(io, "    }")
    @printf(io, "    return [format \"%%.20f\" [expr {2.0 * \$Res / (%d.0 * double(\$noff))}]]\n",
            npairs)
    println(io, "}")
    println(io)

    # gradient — accumulates fsimpson gradient FIDs across offsets × pairs
    println(io, "proc gradient {} {")
    println(io, "    global par lims")
    println(io, "    set par(np) \$par(NOC)")
    println(io, "    set first 1")
    println(io, "    set fsum  {}")
    for (s, d) in pairs_
        println(io, "    set par(start_operator)  $s")
        println(io, "    set par(detect_operator) $d")
        println(io, "    foreach shft \$lims {")
        println(io, "        set g [fsimpson [list [list shift_1_iso \$shft]]]")
        println(io, "        if {\$first} { set fsum \$g; set first 0")
        println(io, "        } else      { fadd \$fsum \$g; funload \$g }")
        println(io, "    }")
    end
    println(io, "    return \$fsum")
    println(io, "}")
    println(io)
end

# ─── main ─────────────────────────────────────────────────────────────────────

function _emit_simpson_main(io::IO, ann::PhysicsAnnotation)
    pwr_hz = ann.controls[1].pwr_max_hz
    println(io, "proc main {} {")
    println(io, "    global par rfsh Pulsar_OUT")
    println(io, "    set noc    [expr {int(\$par(NOC))}]")
    @printf(io, "    set pwr_hz %.4f\n", pwr_hz)
    println(io)
    println(io, "    set rfsh [rand_shape \$pwr_hz \$noc \$noc]")
    println(io, "    oc_optimize \$rfsh")
    println(io)
    println(io, "    # shape2list returns {amplitude_Hz phase_deg} pairs; convert to")
    println(io, "    # Cartesian {Bx_Hz By_Hz} so the :hz parser in the driver is correct.")
    println(io, "    set deg2rad [expr {3.14159265358979323846 / 180.0}]")
    println(io, "    set wfm [shape2list \$rfsh]")
    println(io, "    set fp [open \$Pulsar_OUT w]")
    println(io, "    for {set k 0} {\$k < \$noc} {incr k} {")
    println(io, "        set pair [lindex \$wfm \$k]")
    println(io, "        set amp  [lindex \$pair 0]")
    println(io, "        set phi  [expr {[lindex \$pair 1] * \$deg2rad}]")
    println(io, "        set Bx   [expr {\$amp * cos(\$phi)}]")
    println(io, "        set By   [expr {\$amp * sin(\$phi)}]")
    println(io, "        puts \$fp [format \"%24.16e %24.16e\" \$Bx \$By]")
    println(io, "    }")
    println(io, "    close \$fp")
    println(io, "    free_all_shapes")
    println(io, "}")
end
