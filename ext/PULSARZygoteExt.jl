module PULSARZygoteExt

using PULSAR
using Zygote
import PULSAR: reverse_diff_gradient

function reverse_diff_gradient(
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

    g_flat = try
        gs = Zygote.gradient(f, u0)
        gs[1]
    catch e
        @warn "[AutoDiff] Zygote failed ($e); falling back to finite differences"
        return PULSAR.finite_diff_gradient_ad(system, controls, target;
                                               eps=config.numerical_eps)
    end

    if g_flat === nothing
        @warn "[AutoDiff] Zygote returned nothing gradient; falling back to FD"
        return PULSAR.finite_diff_gradient_ad(system, controls, target;
                                               eps=config.numerical_eps)
    end

    return reshape(g_flat, nc, nt)
end

end  # module PULSARZygoteExt
