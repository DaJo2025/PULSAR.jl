# ============================================================================
# Utilities/FiniteDifference.jl
# Shared central-difference gradient helper for 2-D control matrices.
#
# The 1-D case in finite_difference_gradient (Physics/Gradients.jl) is a
# verification utility that constructs ControlSequence objects inside the
# loop — its pipeline is too problem-specific to share.  Likewise the
# HVP-FD in HighOrderOC.jl is a forward-difference of the gradient, not a
# central-difference of f, and is therefore a distinct operation.
# ============================================================================

"""
    central_diff_gradient_2d!(grad_out, f, u; eps=1e-5) -> grad_out

Fill `grad_out` with the central-difference gradient of scalar function `f(u)`
with respect to each entry of the 2-D matrix `u`.

The matrix `u` is used as the in-place workspace (restore-in-place pattern):
each entry is temporarily perturbed by ±eps, `f` is evaluated, and the entry
is restored before moving on.  The matrix returns bit-for-bit identical to
its input state on exit.
"""
function central_diff_gradient_2d!(grad_out::AbstractMatrix{Float64},
                                    f::Function,
                                    u::AbstractMatrix{Float64};
                                    eps::Float64 = 1e-5)
    @assert size(grad_out) == size(u) "grad_out and u must have identical shape"
    for j in axes(u, 1), k in axes(u, 2)
        orig    = u[j,k]
        u[j,k]  = orig + eps; fp = f(u)
        u[j,k]  = orig - eps; fm = f(u)
        u[j,k]  = orig
        grad_out[j,k] = (fp - fm) / (2 * eps)
    end
    return grad_out
end
