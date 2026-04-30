"""
    comparisons/Translator/Emitters/QuTiPEmitter.jl

Emit a native Python script using QuTiP + the `krotov` package from a
[`PhysicsAnnotation`](@ref).

QuTiP is the Python package for simulating open quantum systems; the
companion `krotov` package (by Michael Goerz) implements Krotov's method
on top of QuTiP with native support for ensemble and multi-state-pair
objectives.  The emitted script is what a competent QuTiP user would write
by hand: `qutip.sigmax()`, `qutip.tensor`, `krotov.Objective`,
`krotov.optimize_pulses`.

Capabilities declared in [`QUTIP_CAPABILITIES`](@ref).
"""

using Printf

const QUTIP_CAPABILITIES = SolverCapabilities(
    multi_spin       = true,
    ensemble         = true,
    multi_state_pair = true,
    lindblad         = false,
    multichannel     = true,
    nonuniform_dt    = true,
    heteronuclear    = true,
    csa              = false,
    dipolar          = false,
    j_coupling       = true,
    amplitude_bounds = true,
)

"""
    emit_qutip(ann, workdir; problem_id="Pulsar",
                 guess_seed=nothing) -> (script_path, waveform_path)
"""
function emit_qutip(ann::PhysicsAnnotation, workdir::String;
                      problem_id::String="Pulsar",
                      guess_seed::Union{Nothing,Int}=nothing)::Tuple{String,String}
    script_path   = joinpath(workdir, "qutip_run.py")
    waveform_path = joinpath(workdir, "qutip_shape.txt")
    seed          = guess_seed === nothing ? ann.guess_seed : guess_seed

    open(script_path, "w") do io
        _emit_qutip_script(io, ann, problem_id, seed, waveform_path)
    end

    return script_path, waveform_path
end

function _emit_qutip_script(io::IO, ann::PhysicsAnnotation,
                              problem_id::String, seed::Int,
                              waveform_path::String)
    n_spins = length(ann.spins)
    n_ctrl  = length(ann.controls)
    n_t     = ann.n_time_steps

    println(io, "# Pulsar benchmark $problem_id — emitted by QuTiPEmitter.jl")
    println(io)
    println(io, "import numpy as np")
    println(io, "import qutip")
    println(io, "import krotov")
    println(io)
    println(io, "np.random.seed($seed)")
    println(io)

    # Single-spin operators
    println(io, "# Single-spin Pauli/2 operators")
    println(io, "sx = qutip.sigmax() / 2")
    println(io, "sy = qutip.sigmay() / 2")
    println(io, "sz = qutip.sigmaz() / 2")
    println(io, "I2 = qutip.qeye(2)")
    println(io)

    # Helper to wrap a single-spin op into the full Hilbert space
    if n_spins == 1
        println(io, "def lift(op, k):")
        println(io, "    return op")
    else
        println(io, "def lift(op, k):")
        println(io, "    ops = [I2] * $n_spins")
        println(io, "    ops[k - 1] = op")
        println(io, "    return qutip.tensor(*ops)")
    end
    println(io)

    # Drift Hamiltonians
    println(io, "def build_drift(offsets_hz):")
    println(io, "    H = 0")
    for k in 1:n_spins
        println(io, "    H = H + 2*np.pi * offsets_hz[$(k-1)] * lift(sz, $k)")
    end
    for c in ann.couplings
        c.kind == :j_isotropic || continue
        @printf(io, "    H = H + 2*np.pi * %.6f * (lift(sx, %d) * lift(sx, %d) + lift(sy, %d) * lift(sy, %d) + lift(sz, %d) * lift(sz, %d))\n",
                c.value_hz, c.i, c.j, c.i, c.j, c.i, c.j)
    end
    println(io, "    return H")
    println(io)

    # Base offsets
    base_off = "[" * join((string(s.offset_hz) for s in ann.spins), ", ") * "]"
    println(io, "base_offsets = $base_off")

    # Drift list
    if ann.sweep === nothing
        println(io, "drift_offsets_list = [base_offsets]")
    else
        off_list = "[" * join((string(Δf) for Δf in ann.sweep.offsets_hz), ", ") * "]"
        println(io, "sweep_offsets = $off_list")
        idx = ann.sweep.target_spin_idx
        if idx == 0
            println(io, "drift_offsets_list = [[b + d for b in base_offsets] for d in sweep_offsets]")
        else
            println(io, "drift_offsets_list = []")
            println(io, "for d in sweep_offsets:")
            println(io, "    offs = list(base_offsets)")
            println(io, "    offs[$(idx-1)] += d")
            println(io, "    drift_offsets_list.append(offs)")
        end
    end
    println(io, "drifts = [build_drift(o) for o in drift_offsets_list]")
    println(io)

    # Control operators
    println(io, "# Control operators (amplitude = 2π * pwr_max_hz * u(t))")
    for (k, c) in pairs(ann.controls)
        axis_op = c.axis == :x ? "sx" : c.axis == :y ? "sy" : "sz"
        @printf(io, "H_ctrl_%d = 2*np.pi * %.6f * lift(%s, %d)\n",
                k, c.pwr_max_hz, axis_op, c.spin_idx)
    end
    println(io)

    # Time grid
    @printf(io, "T_total = %.12e\n", ann.total_time_s)
    println(io, "n_t     = $n_t")
    println(io, "tlist   = np.linspace(0, T_total, n_t + 1)")
    println(io)

    # Initial guess amplitudes (one per control) as step functions
    println(io, "guess_amps = 0.05 * np.random.randn($n_ctrl, n_t)")
    println(io, "guess_amps = np.clip(guess_amps, -1.0, 1.0)")
    println(io)

    # State builders
    println(io, "# State builders — map Pulsar single-spin symbols to QuTiP states")
    println(io, "def single_state(sym):")
    println(io, "    table = {")
    println(io, "      'Iz':  qutip.basis(2, 0),")
    println(io, "      'mIz': qutip.basis(2, 1),")
    println(io, "      'Ix':  (qutip.basis(2, 0) + qutip.basis(2, 1)).unit(),")
    println(io, "      'mIx': (qutip.basis(2, 0) - qutip.basis(2, 1)).unit(),")
    println(io, "      'Iy':  (qutip.basis(2, 0) + 1j*qutip.basis(2, 1)).unit(),")
    println(io, "      'mIy': (qutip.basis(2, 0) - 1j*qutip.basis(2, 1)).unit(),")
    println(io, "    }")
    println(io, "    return table[sym]")
    println(io)
    if n_spins == 1
        println(io, "def composite(symbols):")
        println(io, "    return single_state(symbols[0])")
    else
        println(io, "def composite(symbols):")
        println(io, "    return qutip.tensor(*(single_state(s) for s in symbols))")
    end
    println(io)

    # Time-dependent control wrappers
    for k in 1:n_ctrl
        println(io, "def _u$k(t, args):")
        println(io, "    k = min(int(t / T_total * n_t), n_t - 1)")
        println(io, "    return args['u$k'][k]")
    end
    println(io)

    # Build objectives — cartesian product drifts × state pairs
    println(io, "objectives = []")
    println(io, "for H_d in drifts:")
    ctrl_terms = join(("[H_ctrl_$k, _u$k]" for k in 1:n_ctrl), ", ")
    println(io, "    H = [H_d, $ctrl_terms]")
    init_list = "[" * join(("['" * join((String(s) for s in syms), "','") * "']"
                              for syms in ann.target.initial_states), ", ") * "]"
    targ_list = "[" * join(("['" * join((String(s) for s in syms), "','") * "']"
                              for syms in ann.target.final_states), ", ") * "]"
    println(io, "    inits = $init_list")
    println(io, "    targs = $targ_list")
    println(io, "    for (isyms, tsyms) in zip(inits, targs):")
    println(io, "        psi0 = composite(isyms)")
    println(io, "        psit = composite(tsyms)")
    println(io, "        objectives.append(krotov.Objective(")
    println(io, "            initial_state=psi0, target=psit, H=H))")
    println(io)

    # Pulse options
    println(io, "pulse_options = {")
    for k in 1:n_ctrl
        println(io, "    _u$k: dict(lambda_a=5, update_shape=1),")
    end
    println(io, "}")
    println(io)
    println(io, "args = {'u$(1)': guess_amps[0]," *
               join((" 'u$k': guess_amps[$(k-1)]," for k in 2:n_ctrl), "") * "}")
    println(io)

    # Call krotov.optimize_pulses
    println(io, "result = krotov.optimize_pulses(")
    println(io, "    objectives,")
    println(io, "    pulse_options = pulse_options,")
    println(io, "    tlist         = tlist,")
    println(io, "    iter_stop     = $(ann.max_iter),")
    println(io, "    propagator    = krotov.propagators.expm,")
    println(io, "    chi_constructor = krotov.functionals.chis_re,")
    println(io, "    info_hook     = None,")
    println(io, ")")
    println(io)

    # Extract normalised controls (divide out the 2π * pwr_max_hz factor we put
    # into the control Hamiltonian).  Controls are sampled at tlist midpoints.
    for k in 1:n_ctrl
        println(io, "u$(k)_opt = result.optimized_controls[$(k-1)][:n_t]")
    end
    println(io)

    println(io, "with open(\"$(escape_string(waveform_path))\", \"w\") as f:")
    args  = join(("u$(k)_opt[i]" for k in 1:n_ctrl), ", ")
    fmt   = join(fill("% 24.16e", n_ctrl), " ")
    println(io, "    for i in range(n_t):")
    println(io, "        f.write(\"$fmt\\n\" % ($args,))")
end
