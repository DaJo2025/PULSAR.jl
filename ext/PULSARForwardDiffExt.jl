module PULSARForwardDiffExt

using PULSAR
using ForwardDiff
import PULSAR: forward_diff_gradient

function forward_diff_gradient(
    system   :: PULSAR.AbstractQuantumSystem,
    controls :: PULSAR.ControlSequence,
    target   :: PULSAR.QuantumTarget,
    config   :: PULSAR.AutoDiffConfig,
)::Matrix{Float64}
    nc = system.n_controls
    nt = controls.n_timesteps
    dt = controls.dt
    u0 = vec(controls.controls)

    f = u_flat -> PULSAR._fidelity_flat(u_flat, system, target, nc, nt, dt)

    g_flat = if config.chunk_size > 0
        cfg_fd = ForwardDiff.GradientConfig(f, u0, ForwardDiff.Chunk{config.chunk_size}())
        ForwardDiff.gradient(f, u0, cfg_fd)
    else
        ForwardDiff.gradient(f, u0)
    end

    return reshape(g_flat, nc, nt)
end

end  # module PULSARForwardDiffExt
