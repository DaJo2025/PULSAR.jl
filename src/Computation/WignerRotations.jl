# Computation/WignerRotations.jl
# Wigner rotation matrices and powder averaging grids for solid-state NMR and EPR.
using LinearAlgebra

"""
    wigner_d2(β::Float64) -> Matrix{Float64}

Reduced Wigner d-matrix d^(2)_{mm'}(β) for rank l=2.
Returns a 5×5 matrix with rows/cols indexed m = -2,-1,0,+1,+2 → index 1..5.

# Note
This implementation uses the closed-form factorial expression and is safe only
for `l ≤ 6`; `Int64` factorials overflow at `l ≥ 7` (21! > typemax(Int64)). The
`l = 2` specialization here is fully inside the safe regime. If generalizing to
arbitrary `l`, use the binomial-coefficient recursion or `big(...)` promotion.
"""
function wigner_d2(β::Float64)::Matrix{Float64}
    ch = cos(β/2)
    sh = sin(β/2)

    # d^2_{m,m'}  (m row, m' col), both from -2 to +2, index offset +3
    # Using the general formula:
    # d^l_{m m'} = Σ_k (-1)^k * sqrt((l+m)!(l-m)!(l+m')!(l-m')!) /
    #              ((l+m-k)! k! (m'-m+k)! (l-m'-k)!) * cos(β/2)^(2l+m-m'-2k) * sin(β/2)^(m'-m+2k)
    function d2elem(m::Int, mp::Int)::Float64
        l = 2
        @assert l ≤ 6 "wigner_d2 factorial form requires l ≤ 6 (overflow guard)"
        (abs(m) > l || abs(mp) > l) && return 0.0
        fac_top = sqrt(Float64(factorial(l+m)*factorial(l-m)*factorial(l+mp)*factorial(l-mp)))
        val = 0.0
        for k in 0:(2l)
            f1 = l + m - k
            f2 = k
            f3 = mp - m + k
            f4 = l - mp - k
            (f1 < 0 || f2 < 0 || f3 < 0 || f4 < 0) && continue
            expc = 2*l + m - mp - 2*k
            exps = mp - m + 2*k
            (expc < 0 || exps < 0) && continue
            fac_bot = Float64(factorial(f1)*factorial(f2)*factorial(f3)*factorial(f4))
            val += (-1)^k * (fac_top/fac_bot) * ch^expc * sh^exps
        end
        return val
    end

    d = zeros(Float64, 5, 5)
    for mi in -2:2, mpi in -2:2
        d[mi+3, mpi+3] = d2elem(mi, mpi)
    end
    return d
end

"""
    wigner_D2(α::Float64, β::Float64, γ::Float64) -> Matrix{ComplexF64}

Full Wigner D-matrix D^(2)_{mm'}(α,β,γ) = exp(-im α) d^2_{mm'}(β) exp(-im' γ).
"""
function wigner_D2(α::Float64, β::Float64, γ::Float64)::Matrix{ComplexF64}
    d = wigner_d2(β)
    D = zeros(ComplexF64, 5, 5)
    for mi in -2:2, mpi in -2:2
        D[mi+3, mpi+3] = exp(-im*mi*α) * d[mi+3, mpi+3] * exp(-im*mpi*γ)
    end
    return D
end

"""
    powder_grid(scheme::Symbol, n::Int) -> Vector{NTuple{4,Float64}}

Return a powder averaging grid as a vector of (α, β, γ, weight) tuples.

# Schemes
- `:zcw`        — Zaremba-Conroy-Wolfsberg (equal-area in cos(β))
- `:repulsion`  — REPULSION (uses Fibonacci as approximation)
- `:fibonacci`  — Spherical Fibonacci (uniform)
- `:isotropic`  — Uniformly random (Monte Carlo)

Returns n orientations with weights summing to 1.
"""
function powder_grid(scheme::Symbol, n::Int)::Vector{NTuple{4,Float64}}
    if scheme == :fibonacci
        return _fibonacci_grid(n)
    elseif scheme == :zcw
        return _zcw_grid(n)
    elseif scheme == :repulsion
        return _repulsion_grid(n)
    else  # :isotropic
        return _random_grid(n)
    end
end

function _fibonacci_grid(n::Int)::Vector{NTuple{4,Float64}}
    w = 1.0 / n
    golden = (1 + sqrt(5)) / 2
    orientations = NTuple{4,Float64}[]
    for k in 0:n-1
        β = acos(clamp(1 - 2*(k+0.5)/n, -1.0, 1.0))
        α = 2π * k / golden
        push!(orientations, (mod(α, 2π), β, 0.0, w))
    end
    return orientations
end

function _zcw_grid(n::Int)::Vector{NTuple{4,Float64}}
    w = 1.0 / n
    orientations = NTuple{4,Float64}[]
    for k in 1:n
        cosβ = -1.0 + (2k-1)/n
        β = acos(clamp(cosβ, -1.0, 1.0))
        α = mod(2π * k * 0.6180339887, 2π)  # golden angle
        push!(orientations, (α, β, 0.0, w))
    end
    return orientations
end

function _repulsion_grid(n::Int)::Vector{NTuple{4,Float64}}
    _fibonacci_grid(n)  # Fibonacci as substitute
end

function _random_grid(n::Int)::Vector{NTuple{4,Float64}}
    w = 1.0/n
    orientations = NTuple{4,Float64}[]
    for _ in 1:n
        cosβ = 2*rand() - 1
        push!(orientations, (2π*rand(), acos(clamp(cosβ,-1.0,1.0)), 2π*rand(), w))
    end
    return orientations
end
