module PulsarPlotsExt

using Pulsar
using Plots
import Pulsar: plot_convergence, plot_controls, plot_bloch_trajectory,
               plot_sensitivity_heatmap, plot_pareto_front

function plot_convergence(
    result :: Pulsar.OptimizationResult;
    show_gradient_norm :: Bool   = true,
    log_scale          :: Bool   = true,
    title              :: String = "Pulsar Optimization Convergence",
    save_path          :: String = "",
)
    iters = 1:length(result.fidelity_history)

    if show_gradient_norm && !isempty(result.gradient_norm_history)
        p1 = Plots.plot(iters, result.fidelity_history;
                        xlabel="Iteration", ylabel="Fidelity",
                        label="Fidelity", lw=2, color=:royalblue,
                        title=title, legend=:bottomright)
        Plots.hline!(p1, [1.0]; ls=:dash, color=:gray, label="Target (F=1)")
        gn = result.gradient_norm_history
        g_iters = 1:length(gn)
        p2 = Plots.plot(g_iters, gn;
                        xlabel="Iteration", ylabel="‖∇F‖",
                        label="Gradient norm", lw=2, color=:firebrick,
                        yscale=log_scale ? :log10 : :identity,
                        title="Gradient Norm")
        plt = Plots.plot(p1, p2; layout=(2,1), size=(800,600))
    else
        plt = Plots.plot(iters, result.fidelity_history;
                         xlabel="Iteration", ylabel="Fidelity",
                         label="Fidelity", lw=2, color=:royalblue,
                         title=title, legend=:bottomright, size=(800,400))
        Plots.hline!(plt, [1.0]; ls=:dash, color=:gray, label="Target (F=1)")
    end

    isempty(save_path) || Plots.savefig(plt, save_path)
    return plt
end

function plot_controls(
    controls :: Pulsar.ControlSequence;
    control_labels :: Vector{String} = String[],
    title          :: String         = "Optimal Control Sequence",
    save_path      :: String         = "",
)
    nc, nt = size(controls.controls)
    t_us   = (1:nt) .* (controls.dt * 1e6)
    labels = isempty(control_labels) ? ["u_$(j)" for j in 1:nc] : control_labels

    plt = Plots.plot(; xlabel="Time (μs)", ylabel="Amplitude",
                       title=title, size=(900, 400))
    for j in 1:nc
        Plots.plot!(plt, t_us, controls.controls[j,:]; label=labels[j], lw=1.5)
    end
    isempty(save_path) || Plots.savefig(plt, save_path)
    return plt
end

function plot_bloch_trajectory(
    system   :: Pulsar.AbstractQuantumSystem,
    controls :: Pulsar.ControlSequence;
    initial_state :: Vector{ComplexF64} = ComplexF64[1.0, 0.0],
    save_path     :: String             = "",
)
    system.dim == 2 || error("plot_bloch_trajectory requires a 2-dimensional system")
    σ_x = ComplexF64[0 1; 1 0]
    σ_y = ComplexF64[0 -1im; 1im 0]
    σ_z = ComplexF64[1 0; 0 -1]
    nt  = controls.n_timesteps
    dt  = controls.dt
    t_us = (1:nt) .* (dt * 1e6)
    bx, by, bz = Float64[], Float64[], Float64[]
    ψ = copy(initial_state)
    for k in 1:nt
        H = system.H_drift + sum(controls.controls[j,k] .* system.H_controls[j]
                                  for j in 1:system.n_controls)
        U = Pulsar.compute_propagator(H, dt)
        ψ = U * ψ
        push!(bx, real(ψ' * σ_x * ψ))
        push!(by, real(ψ' * σ_y * ψ))
        push!(bz, real(ψ' * σ_z * ψ))
    end
    plt = Plots.plot(t_us, [bx by bz];
                      label=["⟨σ_x⟩" "⟨σ_y⟩" "⟨σ_z⟩"],
                      xlabel="Time (μs)", ylabel="Bloch coordinate",
                      title="Bloch Vector Trajectory", lw=2,
                      ylims=(-1.05, 1.05), size=(900, 400))
    isempty(save_path) || Plots.savefig(plt, save_path)
    return plt
end

function plot_sensitivity_heatmap(
    sens      :: Pulsar.SensitivityResult;
    title     :: String = "Control Sensitivity Heatmap",
    save_path :: String = "",
)
    S = sens.normalized_sensitivities
    nc, nt = size(S)
    plt = Plots.heatmap(1:nt, 1:nc, S;
                         xlabel="Timestep", ylabel="Control channel",
                         title=title, color=:viridis, clims=(0,1),
                         size=(900, max(300, 80*nc)))
    isempty(save_path) || Plots.savefig(plt, save_path)
    return plt
end

function plot_pareto_front(
    result           :: Pulsar.MultiObjectiveResult,
    objective_names  :: Vector{String} = ["Objective 1", "Objective 2"];
    save_path        :: String         = "",
)
    isempty(result.pareto_objectives) && return nothing
    xs  = [v[1] for v in result.pareto_objectives]
    ys  = [v[2] for v in result.pareto_objectives]
    plt = Plots.scatter(xs, ys;
                         xlabel=get(objective_names, 1, "Obj 1"),
                         ylabel=get(objective_names, 2, "Obj 2"),
                         title="Pareto Front",
                         markersize=6, legend=false, size=(700,500))
    isempty(save_path) || Plots.savefig(plt, save_path)
    return plt
end

end  # module PulsarPlotsExt
