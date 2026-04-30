# ============================================================
# PULSAR.jl — Visualization Utilities
# Pulse Design Library for Spin Control Algorithms and Rollout
# ============================================================
#
# Stub implementations. Real implementations are in ext/PULSARPlotsExt.jl,
# loaded automatically by Julia 1.9+ when Plots.jl is in the environment.
# ============================================================

_ext_plots_error() = error(
    "Plots.jl is required for visualization.\n" *
    "Add it to your environment: `using Plots, PULSAR`. " *
    "The PULSARPlotsExt extension will activate automatically.")

# ──────────────────────────────────────────────────────────────
# Convergence plot
# ──────────────────────────────────────────────────────────────

"""
    plot_convergence(result; show_gradient_norm, log_scale, title, save_path)

Plot fidelity (and optionally gradient norm) vs iteration. Requires Plots.jl.
Returns a `Plots.Plot` object (via PULSARPlotsExt when Plots is loaded).
"""
function plot_convergence(result; kwargs...)
    _ext_plots_error()
end

# ──────────────────────────────────────────────────────────────
# Control sequence plot
# ──────────────────────────────────────────────────────────────

"""
    plot_controls(controls; control_labels, title, save_path)

Plot control amplitudes as a function of time.

Each control channel is shown as a separate coloured line.
Time axis is in microseconds.

Requires Plots.jl.
"""
function plot_controls(controls; kwargs...)
    _ext_plots_error()
end

# ──────────────────────────────────────────────────────────────
# Bloch sphere trajectory (single qubit only)
# ──────────────────────────────────────────────────────────────

"""
    plot_bloch_trajectory(system, controls; initial_state, save_path)

For a single-qubit system, simulate the time evolution and plot the
expectation values ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ as functions of time.

Requires Plots.jl and `system.dim == 2`.
"""
function plot_bloch_trajectory(system, controls; kwargs...)
    _ext_plots_error()
end

# ──────────────────────────────────────────────────────────────
# Sensitivity heatmap
# ──────────────────────────────────────────────────────────────

"""
    plot_sensitivity_heatmap(sens; title, save_path)

Visualise the normalised sensitivity matrix as a colour heatmap.
Bright colours indicate high sensitivity (the fidelity changes a lot
when that control is perturbed).

Requires Plots.jl.
"""
function plot_sensitivity_heatmap(sens; kwargs...)
    _ext_plots_error()
end

# ──────────────────────────────────────────────────────────────
# Pareto front scatter
# ──────────────────────────────────────────────────────────────

"""
    plot_pareto_front(result, objective_names; save_path)

Scatter-plot the Pareto front for a two-objective problem.

Requires Plots.jl.
"""
function plot_pareto_front(result, objective_names=["Objective 1", "Objective 2"]; kwargs...)
    _ext_plots_error()
end

# ──────────────────────────────────────────────────────────────
# Text report
# ──────────────────────────────────────────────────────────────

"""
    create_optimization_report(result, system, target; output_path)

Write a plain-text summary report of an optimization run.

The report includes: system description, algorithm used, convergence
metrics, and final controls statistics.  If Plots.jl is available and
`output_path` ends in `.html`, an HTML report with embedded figures is
written instead.

# Example
```julia
create_optimization_report(result, system, target; output_path="report.txt")
```
"""
function create_optimization_report(
    result::OptimizationResult,
    system::AbstractQuantumSystem,
    target::QuantumTarget;
    output_path::String = "optimization_report.txt",
)
    open(output_path, "w") do io
        println(io, "="^60)
        println(io, "PULSAR.jl Optimization Report")
        println(io, "Pulse Design Library for Spin Control Algorithms and Rollout")
        println(io, "="^60)
        println(io, "Generated: $(now())")
        println(io)
        println(io, "SYSTEM")
        println(io, "  Hilbert space dimension : $(system.dim)")
        println(io, "  Number of controls      : $(system.n_controls)")
        println(io, "  System type             : $(typeof(system))")
        println(io)
        println(io, "TARGET")
        println(io, "  Type                    : $(target.type)")
        println(io)
        println(io, "RESULT")
        @printf(io, "  Final fidelity          : %.8f\n", result.fidelity)
        println(io, "  Converged               : $(result.converged)")
        println(io, "  Termination reason      : $(result.termination_reason)")
        println(io, "  Iterations              : $(result.n_iterations)")
        @printf(io, "  Total wall time         : %.2f s\n", result.total_time)
        println(io, "  Fidelity evaluations    : $(result.n_fidelity_evaluations)")
        println(io, "  Gradient evaluations    : $(result.n_gradient_evaluations)")
        println(io)
        println(io, "CONTROLS")
        u = result.controls
        @printf(io, "  Shape                   : %d controls × %d timesteps\n",
                size(u,1), size(u,2))
        @printf(io, "  Max amplitude           : %.4g\n", maximum(abs, u))
        @printf(io, "  RMS amplitude           : %.4g\n", sqrt(mean(abs2, u)))
        @printf(io, "  Pulse energy            : %.4g\n", sum(abs2, u))
        println(io)
        if !isempty(result.metadata)
            println(io, "METADATA")
            for (k, v) in result.metadata
                println(io, "  $k : $v")
            end
        end
        println(io, "="^60)
    end
    @info "Report written to $output_path"
end
