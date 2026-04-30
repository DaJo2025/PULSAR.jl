"""
    comparisons/Translator/Emitters/QOptEmitter.jl

Emit a native Python script that uses `qopt` (Forschungszentrum Jülich) on
a problem described by a [`PhysicsAnnotation`](@ref).

qopt's canonical pipeline:
  `MatrixSolver` → `CostFunction` → `ScalarMinimizingOptimizer`.

The emitter supports closed-system state-transfer objectives.  Ensemble
problems are translated by summing one `StateInfidelity` cost function per
drift; multi-state-pair objectives by summing one per pair.  (Lindblad/open
systems would require `LindbladSolver` and are declined by capability.)

Capabilities declared in [`QOPT_CAPABILITIES`](@ref).
"""

using Printf

const QOPT_CAPABILITIES = SolverCapabilities(
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
    emit_qopt(ann, workdir; problem_id="PULSAR",
                guess_seed=nothing) -> (script_path, waveform_path)
"""
function emit_qopt(ann::PhysicsAnnotation, workdir::String;
                     problem_id::String="PULSAR",
                     guess_seed::Union{Nothing,Int}=nothing)::Tuple{String,String}
    script_path   = joinpath(workdir, "qopt_run.py")
    waveform_path = joinpath(workdir, "qopt_shape.txt")
    seed          = guess_seed === nothing ? ann.guess_seed : guess_seed

    open(script_path, "w") do io
        _emit_qopt_script(io, ann, problem_id, seed, waveform_path)
    end

    return script_path, waveform_path
end

function _emit_qopt_script(io::IO, ann::PhysicsAnnotation,
                             problem_id::String, seed::Int,
                             waveform_path::String)
    n_spins = length(ann.spins)
    n_ctrl  = length(ann.controls)
    n_t     = ann.n_time_steps

    println(io, "# PULSAR benchmark $problem_id — emitted by QOptEmitter.jl")
    println(io)
    println(io, "import numpy as np")
    println(io, "import qopt")
    println(io, "from qopt import MatrixSolver, StateInfidelity, SumOfCostFunctions")
    println(io, "from qopt.optimizer import ScalarMinimizingOptimizer")
    println(io)
    println(io, "np.random.seed($seed)")
    println(io)

    # Single-spin operators
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

    # Drift builder
    println(io, "def build_drift(offsets_hz):")
    println(io, "    H = np.zeros_like(lift(sz, 1))")
    for k in 1:n_spins
        println(io, "    H = H + 2*np.pi * offsets_hz[$(k-1)] * lift(sz, $k)")
    end
    for c in ann.couplings
        c.kind == :j_isotropic || continue
        @printf(io, "    H = H + 2*np.pi * %.6f * (lift(sx,%d)@lift(sx,%d) + lift(sy,%d)@lift(sy,%d) + lift(sz,%d)@lift(sz,%d))\n",
                c.value_hz, c.i, c.j, c.i, c.j, c.i, c.j)
    end
    println(io, "    return H")
    println(io)

    base_off = "[" * join((string(s.offset_hz) for s in ann.spins), ", ") * "]"
    println(io, "base_offsets = $base_off")
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

    # Control operators, scaled by 2π·pwr_max_hz so qopt's control amps are
    # dimensionless in [-1, 1].
    for (k, c) in pairs(ann.controls)
        axis_op = c.axis == :x ? "sx" : c.axis == :y ? "sy" : "sz"
        @printf(io, "H_ctrl_%d = 2*np.pi * %.6f * lift(%s, %d)\n",
                k, c.pwr_max_hz, axis_op, c.spin_idx)
    end
    println(io, "h_ctrl = [" * join(("H_ctrl_$k" for k in 1:n_ctrl), ", ") * "]")
    println(io)

    # State builders
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
    println(io)

    # Timing
    @printf(io, "dt     = %.12e\n", ann.dt_s)
    println(io, "n_t    = $n_t")
    println(io, "tlist  = dt * np.ones(n_t)")
    println(io)

    # Build one Simulator+CostFunction per (drift, state_pair)
    init_py = "[" * join(("['" * join((String(s) for s in syms), "','") * "']"
                           for syms in ann.target.initial_states), ", ") * "]"
    targ_py = "[" * join(("['" * join((String(s) for s in syms), "','") * "']"
                           for syms in ann.target.final_states), ", ") * "]"
    println(io, "inits = $init_py")
    println(io, "targs = $targ_py")
    println(io)
    println(io, "cost_fns = []")
    println(io, "for H_d in drifts:")
    println(io, "    sim = MatrixSolver(h_drift=H_d, h_ctrl=h_ctrl, tau=tlist)")
    println(io, "    for (isyms, tsyms) in zip(inits, targs):")
    println(io, "        psi0 = composite(isyms)")
    println(io, "        psit = composite(tsyms)")
    println(io, "        cost_fns.append(StateInfidelity(solver=sim,")
    println(io, "                                         target=psit,")
    println(io, "                                         initial_state=psi0))")
    println(io)
    println(io, "cost = SumOfCostFunctions(cost_functions=cost_fns)")
    println(io)

    # Initial guess
    println(io, "guess = 0.05 * np.random.randn(n_t, $n_ctrl)")
    println(io, "guess = np.clip(guess, -1.0, 1.0)")
    println(io)

    # Optimize
    println(io, "optimizer = ScalarMinimizingOptimizer(cost_function=cost,")
    println(io, "                                       bounds=[[-1.0, 1.0]] * ($n_ctrl * n_t))")
    println(io, "result = optimizer.run_optimization(guess.flatten())")
    println(io)

    # Extract + write
    println(io, "opt = result.final_parameters.reshape(n_t, $n_ctrl)")
    println(io, "with open(\"$(escape_string(waveform_path))\", \"w\") as f:")
    fmt = join(fill("% 24.16e", n_ctrl), " ")
    args = join(("opt[i, $(k-1)]" for k in 1:n_ctrl), ", ")
    println(io, "    for i in range(n_t):")
    println(io, "        f.write(\"$fmt\\n\" % ($args,))")
end
