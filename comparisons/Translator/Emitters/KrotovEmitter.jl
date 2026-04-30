"""
    comparisons/Translator/Emitters/KrotovEmitter.jl

Emit a native Julia script that runs Krotov.jl on a problem described by a
[`PhysicsAnnotation`](@ref).  The emitted script:

  - Uses Pulsar's public API (`mr_system`, `hamiltonian`, `spin_op`,
    `spin_state`) to build drifts / operators / states from the annotation
    (no serialised matrix literals).
  - Constructs QuantumControl.jl `Trajectory` objects — one per `(drift,
    state pair)` combination — and calls `QuantumControl.optimize(...;
    method=Krotov)`.
  - Writes the optimised controls, normalised into `[l_bound, u_bound]`, to
    a text file for the driver to parse.

Capabilities declared in [`KROTOV_CAPABILITIES`](@ref).
"""

using Printf

const KROTOV_CAPABILITIES = SolverCapabilities(
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
    emit_krotov(ann, workdir; problem_id="Pulsar",
                  guess_seed=nothing, project_root=...) -> (script_path, waveform_path)
"""
function emit_krotov(ann::PhysicsAnnotation, workdir::String;
                      problem_id::String="Pulsar",
                      guess_seed::Union{Nothing,Int}=nothing)::Tuple{String,String}
    script_path   = joinpath(workdir, "krotov_run.jl")
    waveform_path = joinpath(workdir, "krotov_shape.txt")
    seed          = guess_seed === nothing ? ann.guess_seed : guess_seed

    open(script_path, "w") do io
        _emit_krotov_script(io, ann, problem_id, seed, waveform_path)
    end

    return script_path, waveform_path
end

function _emit_krotov_script(io::IO, ann::PhysicsAnnotation,
                              problem_id::String, seed::Int,
                              waveform_path::String)
    n_ctrl = length(ann.controls)
    n_t    = ann.n_time_steps
    pwr    = 2π * ann.controls[1].pwr_max_hz
    l, u   = -1.0, 1.0                 # annotation amplitude bounds

    println(io, "# Pulsar benchmark $problem_id — emitted by KrotovEmitter.jl")
    println(io)
    println(io, "using LinearAlgebra")
    println(io, "using Printf")
    println(io, "using Random")
    println(io, "using Pulsar")
    println(io, "using QuantumControl")
    println(io, "using QuantumControl.Functionals: J_T_sm")
    println(io, "using QuantumPropagators: ExpProp")
    println(io, "import Krotov")
    println(io)

    # Spin system
    isos = join(("\"$(s.isotope)\"" for s in ann.spins), ", ")
    if length(ann.spins) == 1
        println(io, "sys = mr_system(\"$(ann.spins[1].isotope)\")")
    else
        println(io, "sys = mr_system([$isos])")
    end

    # Drifts (Pulsar.hamiltonian — fully-qualified to avoid clash with
    # QuantumControl.hamiltonian which is also loaded below).
    if ann.sweep === nothing
        base_offsets = [s.offset_hz for s in ann.spins]
        _write_couplings_matrix(io, ann)
        println(io, "drifts = [Pulsar.hamiltonian(sys; B0_tesla=$(ann.b0_tesla), " *
                     "offsets_hz=$(base_offsets), couplings_hz=J)]")
    else
        base_offsets = [s.offset_hz for s in ann.spins]
        _write_couplings_matrix(io, ann)
        print(io, "sweep_offsets = [")
        for (k, v) in pairs(ann.sweep.offsets_hz)
            @printf(io, "%.10g", v)
            k < length(ann.sweep.offsets_hz) && print(io, ", ")
        end
        println(io, "]")
        if ann.sweep.target_spin_idx == 0
            println(io, "drifts = [Pulsar.hamiltonian(sys; B0_tesla=$(ann.b0_tesla), " *
                         "offsets_hz=$(base_offsets) .+ Δf, couplings_hz=J) " *
                         "for Δf in sweep_offsets]")
        else
            idx = ann.sweep.target_spin_idx
            println(io, "base_offsets = Float64[$(join(base_offsets, ", "))]")
            println(io, "drifts = map(sweep_offsets) do Δf")
            println(io, "    offs = copy(base_offsets)")
            println(io, "    offs[$idx] += Δf")
            println(io, "    Pulsar.hamiltonian(sys; B0_tesla=$(ann.b0_tesla), " *
                         "offsets_hz=offs, couplings_hz=J)")
            println(io, "end")
        end
    end
    println(io)

    # Control operators
    for (k, c) in pairs(ann.controls)
        sym = c.axis == :x ? ":Ix" : c.axis == :y ? ":Iy" : ":Iz"
        if length(ann.spins) == 1
            println(io, "Op$k = ComplexF64.(spin_op(sys, $sym))")
        else
            println(io, "Op$k = ComplexF64.(spin_op(sys, $sym, $(c.spin_idx)))")
        end
    end
    println(io)

    # State pairs
    for (s, (init_syms, targ_syms)) in enumerate(zip(ann.target.initial_states,
                                                      ann.target.final_states))
        _emit_krotov_state(io, "psi0_$s", init_syms, ann)
        _emit_krotov_state(io, "psit_$s", targ_syms, ann)
    end
    println(io)

    # Time grid and initial guess
    @printf(io, "dt   = %.12e\n", ann.dt_s)
    println(io, "n_t  = $n_t")
    @printf(io, "pwr  = %.12e\n", pwr)
    println(io, "tlist = collect(range(0.0, n_t * dt, length = n_t + 1))")
    println(io, "rng = Random.MersenneTwister($seed)")
    @printf(io, "u_init = 0.05 .* randn(rng, %d, n_t)\n", n_ctrl)
    @printf(io, "u_init = clamp.(u_init, %g, %g)\n", l, u)
    for k in 1:n_ctrl
        println(io, "u$k = vcat(pwr .* u_init[$k, :], pwr * u_init[$k, end])")
    end
    println(io)

    # Trajectories
    n_pairs = length(ann.target.initial_states)
    ctrl_tup = join(("(Op$k, u$k)" for k in 1:n_ctrl), ", ")
    println(io, "trajectories = Trajectory[]")
    println(io, "for H_d in drifts")
    println(io, "    H = QuantumControl.hamiltonian(H_d, $ctrl_tup)")
    for s in 1:n_pairs
        println(io, "    push!(trajectories, Trajectory(psi0_$s, H; target_state=psit_$s))")
    end
    println(io, "end")
    println(io)

    # Run Krotov
    println(io, "problem = ControlProblem(trajectories, tlist;")
    @printf(io, "    iter_stop   = %d,\n", ann.max_iter)
    println(io, "    J_T         = J_T_sm,")
    println(io, "    prop_method = ExpProp,")
    println(io, ")")
    println(io, "result = optimize(problem; method=Krotov)")
    println(io)

    # Extract, normalise, write
    for k in 1:n_ctrl
        println(io, "opt_u$k = result.optimized_controls[$k][1:n_t] ./ pwr")
    end
    fmt  = join(fill("%24.16e", n_ctrl), " ")
    args = join(("opt_u$k[k]" for k in 1:n_ctrl), ", ")
    println(io, "open(\"$(escape_string(waveform_path))\", \"w\") do f")
    println(io, "    for k in 1:n_t")
    println(io, "        @printf(f, \"$fmt\\n\", $args)")
    println(io, "    end")
    println(io, "end")
    println(io)
    println(io, "krotov_fidelity = 1.0 - result.J_T")
    println(io, "println(\"Pulsar_KROTOV_FIDELITY: \$(krotov_fidelity)\")")
end

function _write_couplings_matrix(io::IO, ann::PhysicsAnnotation)
    N = length(ann.spins)
    println(io, "J = zeros(Float64, $N, $N)")
    for c in ann.couplings
        c.kind == :j_isotropic || continue
        @printf(io, "J[%d, %d] = %.6f; J[%d, %d] = %.6f\n",
                c.i, c.j, c.value_hz, c.j, c.i, c.value_hz)
    end
end

function _emit_krotov_state(io::IO, varname::String, syms::Vector{Symbol},
                              ann::PhysicsAnnotation)
    if length(ann.spins) == 1
        println(io, "$varname = ComplexF64.(spin_state(sys, :$(syms[1])))")
        return
    end
    parts = ["ComplexF64.(spin_state(mr_system(\"$(ann.spins[k].isotope)\"), :$(syms[k])))"
              for k in 1:length(syms)]
    println(io, "$varname = kron($(join(parts, ", ")))")
end
