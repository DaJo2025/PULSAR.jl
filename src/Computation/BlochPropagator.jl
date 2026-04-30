# Computation/BlochPropagator.jl
# Bloch equation propagator for MRI optimal control.
# Uses Rodrigues rotation formula with exponential relaxation.
using LinearAlgebra

"""
    bloch_step!(M::Vector{Float64}, B1x, B1y, Gx, Gy, Gz,
                r, gamma, T1, T2, delta_B0, M0z, dt)

Propagate magnetization M = [Mx,My,Mz] by one time step dt using
Rodrigues rotation + exponential relaxation.
"""
function bloch_step!(M::Vector{Float64},
                     B1x::Float64, B1y::Float64,
                     Gx::Float64, Gy::Float64, Gz::Float64,
                     r::NTuple{3,Float64},
                     gamma::Float64,
                     T1::Float64, T2::Float64,
                     delta_B0::Float64,
                     M0z::Float64,
                     dt::Float64)
    Bx = B1x
    By = B1y
    Bz = Gx*r[1] + Gy*r[2] + Gz*r[3] + delta_B0

    B_norm = sqrt(Bx^2 + By^2 + Bz^2)
    if B_norm < 1e-15
        e1 = exp(-dt/T1); e2 = exp(-dt/T2)
        M[1] *= e2; M[2] *= e2; M[3] = M[3]*e1 + M0z*(1-e1)
        return
    end

    θ = gamma * B_norm * dt
    sinθ, cosθ = sin(θ), cos(θ)
    nx, ny, nz = Bx/B_norm, By/B_norm, Bz/B_norm

    Mx, My, Mz = M[1], M[2], M[3]
    ndotM = nx*Mx + ny*My + nz*Mz

    M[1] = Mx*cosθ + (ny*Mz - nz*My)*sinθ + nx*ndotM*(1-cosθ)
    M[2] = My*cosθ + (nz*Mx - nx*Mz)*sinθ + ny*ndotM*(1-cosθ)
    M[3] = Mz*cosθ + (nx*My - ny*Mx)*sinθ + nz*ndotM*(1-cosθ)

    e1 = exp(-dt/T1); e2 = exp(-dt/T2)
    M[1] *= e2; M[2] *= e2; M[3] = M[3]*e1 + M0z*(1-e1)
end

"""
    bloch_forward_pass(sys::BlochSystem, ctrl::MRIControlSequence)
    -> M_traj::Array{Float64,3}

Forward propagate all isochromats. Returns M_traj of size [3, n_iso, n_steps+1].
"""
function bloch_forward_pass(sys::BlochSystem,
                             ctrl::MRIControlSequence)::Array{Float64,3}
    n_iso = sys.n_isochromats
    n_t   = ctrl.n_steps
    M_traj = zeros(Float64, 3, n_iso, n_t+1)

    for i in 1:n_iso
        iso = sys.isochromats[i]
        M_traj[1,i,1] = iso.M0[1]
        M_traj[2,i,1] = iso.M0[2]
        M_traj[3,i,1] = iso.M0[3]
    end

    # Lesson 2: per-isochromat Bloch propagation; `@threadsif` adds BLAS
    # thread guard around the parallel iso loop.
    @threadsif true for i in 1:n_iso
        iso = sys.isochromats[i]
        M = [M_traj[1,i,1], M_traj[2,i,1], M_traj[3,i,1]]
        for k in 1:n_t
            bloch_step!(M,
                        ctrl.B1[1,k], ctrl.B1[2,k],
                        ctrl.G[1,k], ctrl.G[2,k], ctrl.G[3,k],
                        iso.position, sys.gamma,
                        iso.T1, iso.T2, iso.delta_B0, iso.M0[3],
                        ctrl.dt)
            M_traj[1,i,k+1] = M[1]
            M_traj[2,i,k+1] = M[2]
            M_traj[3,i,k+1] = M[3]
        end
    end
    return M_traj
end
