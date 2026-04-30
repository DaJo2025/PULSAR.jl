"""
    comparisons/Translator/Emitters/QuandaryEmitter.jl

Emit a native Python script that drives Quandary (LLNL) on a problem described
by a [`PhysicsAnnotation`](@ref).

Quandary is optimised for superconducting-qubit control and does not natively
support NMR-style offset-sweep ensembles or Lindblad propagation out of the
box.  The emitter therefore declares narrow capabilities: single drift,
no ensemble, closed-system state transfer, at most two controls.  Unsupported
problems short-circuit in the driver via `check_supported`.

The emitted script uses Quandary's custom-Hamiltonian mode (`standardmodel =
False`) so that the drift + control operators built from the annotation are
passed through verbatim — no transmon parameterisation is assumed.

Capabilities declared in [`QUANDARY_CAPABILITIES`](@ref).
"""

using Printf

const QUANDARY_CAPABILITIES = SolverCapabilities(
    multi_spin       = true,
    ensemble         = false,
    multi_state_pair = false,   # NMR path: single state pair only. Gate
                                # objectives on transmons are handled by
                                # Quandary's internal basis scan (see
                                # TransmonAnnotation check_supported).
    lindblad         = false,
    multichannel     = true,
    nonuniform_dt    = false,
    heteronuclear    = true,
    csa              = false,
    dipolar          = false,
    j_coupling       = true,
    amplitude_bounds = true,
    qc_transmon      = true,    # native target: superconducting qudits
)

"""
    emit_quandary(ann, workdir; problem_id="PULSAR",
                    guess_seed=nothing) -> (script_path, waveform_path)
"""
function emit_quandary(ann::PhysicsAnnotation, workdir::String;
                         problem_id::String="PULSAR",
                         guess_seed::Union{Nothing,Int}=nothing)::Tuple{String,String}
    script_path   = joinpath(workdir, "quandary_run.py")
    waveform_path = joinpath(workdir, "quandary_shape.txt")
    seed          = guess_seed === nothing ? ann.guess_seed : guess_seed

    open(script_path, "w") do io
        _emit_quandary_script(io, ann, problem_id, seed, waveform_path, workdir)
    end

    return script_path, waveform_path
end

function _emit_quandary_script(io::IO, ann::PhysicsAnnotation,
                                 problem_id::String, seed::Int,
                                 waveform_path::String, workdir::String)
    n_spins = length(ann.spins)
    n_ctrl  = length(ann.controls)
    n_t     = ann.n_time_steps
    dim     = 2 ^ n_spins

    println(io, "# PULSAR benchmark $problem_id — emitted by QuandaryEmitter.jl")
    println(io)
    println(io, "import numpy as np")
    println(io, "import quandary")
    println(io)
    println(io, "np.random.seed($seed)")
    println(io)

    println(io, "sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)")
    println(io, "sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)")
    println(io, "sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)")
    println(io, "I2 = np.eye(2, dtype=complex)")
    println(io)

    if n_spins == 1
        println(io, "def lift(op, k): return op")
    else
        println(io, "def kron_list(ops):")
        println(io, "    out = ops[0]")
        println(io, "    for o in ops[1:]: out = np.kron(out, o)")
        println(io, "    return out")
        println(io)
        println(io, "def lift(op, k):")
        println(io, "    ops = [I2] * $n_spins")
        println(io, "    ops[k - 1] = op")
        println(io, "    return kron_list(ops)")
    end
    println(io)

    # Drift Hamiltonian (single — ensemble disabled via capabilities)
    base_off = "[" * join((string(s.offset_hz) for s in ann.spins), ", ") * "]"
    println(io, "offsets_hz = $base_off")
    println(io, "H_drift = np.zeros(($dim, $dim), dtype=complex)")
    for k in 1:n_spins
        println(io, "H_drift = H_drift + 2*np.pi * offsets_hz[$(k-1)] * lift(sz, $k)")
    end
    for c in ann.couplings
        c.kind == :j_isotropic || continue
        @printf(io, "H_drift = H_drift + 2*np.pi * %.6f * (lift(sx,%d)@lift(sx,%d) + lift(sy,%d)@lift(sy,%d) + lift(sz,%d)@lift(sz,%d))\n",
                c.value_hz, c.i, c.j, c.i, c.j, c.i, c.j)
    end
    println(io)

    # Control operators
    for (k, c) in pairs(ann.controls)
        axis_op = c.axis == :x ? "sx" : c.axis == :y ? "sy" : "sz"
        @printf(io, "H_ctrl_%d = 2*np.pi * %.6f * lift(%s, %d)\n",
                k, c.pwr_max_hz, axis_op, c.spin_idx)
    end
    ctrl_list = "[" * join(("H_ctrl_$k" for k in 1:n_ctrl), ", ") * "]"
    println(io, "Hc_re = $ctrl_list")
    println(io, "Hc_im = [np.zeros_like(H_drift)] * $n_ctrl")
    println(io)

    # State pair (single — multi_state_pair disabled via capabilities)
    println(io, "_single = {")
    println(io, "  'Iz':  np.array([1.0, 0.0], dtype=complex),")
    println(io, "  'mIz': np.array([0.0, 1.0], dtype=complex),")
    println(io, "  'Ix':  np.array([1.0, 1.0], dtype=complex) / np.sqrt(2),")
    println(io, "  'mIx': np.array([1.0,-1.0], dtype=complex) / np.sqrt(2),")
    println(io, "  'Iy':  np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2),")
    println(io, "  'mIy': np.array([1.0,-1.0j], dtype=complex) / np.sqrt(2),")
    println(io, "}")
    if n_spins == 1
        println(io, "def composite(symbols): return _single[symbols[0]]")
    else
        println(io, "def composite(symbols):")
        println(io, "    v = _single[symbols[0]]")
        println(io, "    for s in symbols[1:]: v = np.kron(v, _single[s])")
        println(io, "    return v")
    end
    init_syms = ann.target.initial_states[1]
    targ_syms = ann.target.final_states[1]
    init_py = "['" * join((String(s) for s in init_syms), "','") * "']"
    targ_py = "['" * join((String(s) for s in targ_syms), "','") * "']"
    println(io, "psi0 = composite($init_py)")
    println(io, "psit = composite($targ_py)")
    println(io)

    # Timing + bounds
    @printf(io, "T     = %.12e\n", ann.total_time_s)
    println(io, "n_t   = $n_t")
    pwrs_mhz = [c.pwr_max_hz * 1e-6 for c in ann.controls]
    bounds_py = "[" * join((@sprintf("%.6e", p) for p in pwrs_mhz), ", ") * "]"
    println(io, "maxctrl_MHz = $bounds_py")
    println(io)

    # Build Quandary config (custom Hamiltonian mode)
    levels_list = "[" * join(fill("2", n_spins), ", ") * "]"
    println(io, "cfg = quandary.Quandary(")
    println(io, "    Ne               = $levels_list,")
    println(io, "    Ng               = " * "[" * join(fill("0", n_spins), ", ") * "],")
    println(io, "    standardmodel    = False,")
    println(io, "    Hsys             = H_drift,")
    println(io, "    Hc_re            = Hc_re,")
    println(io, "    Hc_im            = Hc_im,")
    println(io, "    T                = T,")
    println(io, "    nsteps           = n_t,")
    println(io, "    initialcondition = psi0,")
    println(io, "    targetstate      = psit,")
    println(io, "    optim_target     = \"pure\",")
    println(io, "    costfunction     = \"Jtrace\",")
    println(io, "    maxctrl_MHz      = maxctrl_MHz,")
    @printf(io, "    maxiter          = %d,\n", ann.max_iter)
    println(io, "    rand_seed        = $seed,")
    println(io, ")")
    println(io)

    println(io, "datadir = \"$(escape_string(joinpath(workdir, "quandary_out")))\"")
    println(io, "t, pt, qt, infid, _, _ = cfg.optimize(datadir=datadir)")
    println(io)

    # Extract optimised real/imag envelopes, normalise by 2π·pwr_max_hz.
    # quandary returns pt[k], qt[k] as arrays of length >= n_t+1; we sample the
    # first n_t points as the dimensionless [-1,1] control.
    println(io, "pwr_hz = np.array($bounds_py) * 1e6     # MHz → Hz")
    println(io, "pwr    = 2*np.pi * pwr_hz")
    println(io, "opt    = np.zeros((n_t, $n_ctrl))")
    println(io, "for k in range($n_ctrl):")
    println(io, "    xk = np.asarray(pt[k])[:n_t] if k < len(pt) else np.zeros(n_t)")
    println(io, "    opt[:, k] = xk / pwr[k]")
    println(io, "opt = np.clip(opt, -1.0, 1.0)")
    println(io)

    println(io, "with open(\"$(escape_string(waveform_path))\", \"w\") as f:")
    fmt  = join(fill("% 24.16e", n_ctrl), " ")
    args = join(("opt[i, $(k-1)]" for k in 1:n_ctrl), ", ")
    println(io, "    for i in range(n_t):")
    println(io, "        f.write(\"$fmt\\n\" % ($args,))")
end


# ─── TransmonAnnotation path (Quandary's native standardmodel) ────────────────
#
# Quandary's native strength is transmon / qudit control — `freq01`, `selfkerr`,
# `crosskerr`, guard levels, rotating-frame complex envelope.  A
# `TransmonAnnotation` maps onto this path directly; the emitter does NOT build
# a custom Hamiltonian here — Quandary does it internally.

function emit_quandary(ann::TransmonAnnotation, workdir::String;
                         problem_id::String="PULSAR",
                         guess_seed::Union{Nothing,Int}=nothing)::Tuple{String,String}
    script_path   = joinpath(workdir, "quandary_run.py")
    waveform_path = joinpath(workdir, "quandary_shape.txt")
    seed          = guess_seed === nothing ? ann.guess_seed : guess_seed

    open(script_path, "w") do io
        _emit_quandary_transmon_script(io, ann, problem_id, seed,
                                         waveform_path, workdir)
    end

    return script_path, waveform_path
end

function _emit_quandary_transmon_script(io::IO, ann::TransmonAnnotation,
                                          problem_id::String, seed::Int,
                                          waveform_path::String, workdir::String)
    nq      = n_qubits(ann)
    n_t     = ann.n_time_steps
    n_ctrl  = 2 * nq                 # Quandary returns (pt, qt) per qudit

    println(io, "# PULSAR benchmark $problem_id — emitted by QuandaryEmitter.jl")
    println(io, "# TransmonAnnotation path (standardmodel=True)")
    println(io)
    println(io, "import numpy as np")
    println(io, "import quandary")
    println(io)
    println(io, "np.random.seed($seed)")
    println(io)

    # freq01 / selfkerr (GHz)
    freq_ghz     = [f / 1e9 for f in ann.freq01_hz]
    selfkerr_ghz = [abs(a) / 1e9 for a in ann.anharm_hz]  # Quandary positive-magnitude convention
    println(io, "freq01       = $(_py_float_list(freq_ghz))")
    println(io, "selfkerr     = $(_py_float_list(selfkerr_ghz))")

    # Cross-Kerr matrix (upper triangle, ordered as quandary expects)
    xk_vals = Float64[]
    for i in 1:nq, j in i+1:nq
        push!(xk_vals, ann.coupling_hz[i, j] / 1e9)    # GHz
    end
    if !isempty(xk_vals) && any(!iszero, xk_vals)
        println(io, "crosskerr    = $(_py_float_list(xk_vals))")
    end

    # maxctrl_MHz — one per qudit. Quandary's drive uses (a+a†) while PULSAR's
    # Ix = (a+a†)/2, so Quandary's pt saturates at half the "omega_max" to give
    # the same peak Rabi frequency PULSAR achieves at w = 1.  Emit maxctrl_MHz
    # = omega_max_hz / 2e6 so the returned pt/qt are directly comparable and
    # the normalisation below stays inside [-1, 1] without lossy clipping.
    maxctrl_mhz = fill(ann.omega_max_hz / 2e6, nq)
    println(io, "maxctrl_MHz  = $(_py_float_list(maxctrl_mhz))")

    @printf(io, "T            = %.9f\n", ann.total_time_s * 1e9)   # ns
    @printf(io, "nsteps       = %d\n",  n_t)
    @printf(io, "maxiter      = %d\n",  ann.max_iter)
    @printf(io, "rand_seed    = %d\n",  seed)
    println(io, "Ne           = $(_py_int_list(fill(ann.n_essential, nq)))")
    println(io, "Ng           = $(_py_int_list(fill(ann.n_guard,     nq)))")
    println(io)

    # Target: either state transfer on the essential subspace or a unitary gate
    if ann.target_kind === :state_transfer
        println(io, "initialcondition = $(_py_complex_list(ann.initial_state))")
        println(io, "targetstate      = $(_py_complex_list(ann.target_state))")
        println(io, "targetgate       = None")
        println(io, "optim_target     = \"pure\"")
    elseif ann.target_kind === :gate
        ann.target_unitary === nothing &&
            throw(ArgumentError("TransmonAnnotation.target_kind = :gate requires target_unitary"))
        U = ann.target_unitary
        d = size(U, 1)
        rows = String[]
        for r in 1:d
            entries = String[]
            for c in 1:d
                z = U[r, c]
                push!(entries, @sprintf("(%.10f%+.10fj)", real(z), imag(z)))
            end
            push!(rows, "[" * join(entries, ", ") * "]")
        end
        println(io, "targetgate = np.array([" * join(rows, ", ") * "], dtype=complex)")
        println(io, "initialcondition = \"basis\"")
        println(io, "targetstate      = None")
        println(io, "optim_target     = \"gate\"")
    else
        throw(ArgumentError("Unknown target_kind :$(ann.target_kind)"))
    end
    println(io)

    # Build the config dataclass — pass optional fields only if defined in
    # this script, to keep the block clean for both target types.
    println(io, "kwargs = dict(")
    println(io, "    Ne=Ne, Ng=Ng,")
    println(io, "    freq01=freq01, selfkerr=selfkerr,")
    isempty(xk_vals) || println(io, "    crosskerr=crosskerr,")
    println(io, "    maxctrl_MHz=maxctrl_MHz,")
    println(io, "    T=T, nsteps=nsteps,")
    println(io, "    rand_seed=rand_seed, maxiter=maxiter,")
    println(io, "    optim_target=optim_target,")
    println(io, "    initialcondition=initialcondition,")
    println(io, "    tol_infidelity=1e-5,")
    println(io, ")")
    println(io, "if targetstate is not None: kwargs['targetstate'] = targetstate")
    println(io, "if targetgate is not None:  kwargs['targetgate']  = targetgate")
    println(io, "cfg = quandary.Quandary(**kwargs)")
    println(io)

    println(io, "datadir = \"$(escape_string(joinpath(workdir, "quandary_out")))\"")
    println(io, "t, pt, qt, infid, _, _ = cfg.optimize(datadir=datadir)")
    println(io)

    # Write the optimised controls normalised into [-1, 1] per column.
    # Column order: [pt_q1, qt_q1, pt_q2, qt_q2, ...] to match PULSAR's
    # [Ix_q1, Iy_q1, Ix_q2, Iy_q2, ...] operator ordering.
    # Quandary's pt, qt are in MHz; because maxctrl_MHz was set to
    # omega_max_hz/2e6, the map to PULSAR's w ∈ [-1,1] is w = pt/maxctrl_MHz.
    println(io, "maxctrl_MHz_arr = np.asarray(maxctrl_MHz, dtype=float)")
    println(io, "opt = np.zeros((nsteps, $n_ctrl))")
    for q in 1:nq
        println(io, "xk = np.asarray(pt[$(q-1)])[:nsteps] if $(q-1) < len(pt) else np.zeros(nsteps)")
        println(io, "yk = np.asarray(qt[$(q-1)])[:nsteps] if $(q-1) < len(qt) else np.zeros(nsteps)")
        println(io, "opt[:, $(2*(q-1))]   = xk / maxctrl_MHz_arr[$(q-1)]")
        println(io, "opt[:, $(2*(q-1)+1)] = yk / maxctrl_MHz_arr[$(q-1)]")
    end
    println(io, "opt = np.clip(opt, -1.0, 1.0)")
    println(io)

    println(io, "with open(\"$(escape_string(waveform_path))\", \"w\") as f:")
    fmt  = join(fill("% 24.16e", n_ctrl), " ")
    args = join(("opt[i, $(k-1)]" for k in 1:n_ctrl), ", ")
    println(io, "    for i in range(nsteps):")
    println(io, "        f.write(\"$fmt\\n\" % ($args,))")
end

_py_float_list(xs)   = "[" * join((@sprintf("%.10f", x) for x in xs), ", ") * "]"
_py_int_list(xs)     = "[" * join((string(Int(x)) for x in xs), ", ") * "]"
_py_complex_list(zs) = "[" *
    join((@sprintf("(%.10f%+.10fj)", real(z), imag(z)) for z in zs), ", ") * "]"
