"""
    MR/LindbladMR.jl

Physical helpers for Lindblad (open-system) MR applications.

Provides:
  mr_relaxation   — build T1/T2 Lindblad operators from physical times
  density_matrix  — build ρ = |ψ⟩⟨ψ| from a pure-state vector

## Relaxation model

NMR relaxation is described by the Bloch equations in the secular approximation.
For a spin-1/2 nucleus the relevant channels are:

  T1 (longitudinal, spin-lattice):
    Drives Mz toward thermal equilibrium. Modelled by two jump operators per spin:
      L₊ = I₊  (stimulated emission, spin flips up)   rate γ₊ = 1/(2T1)
      L₋ = I₋  (stimulated absorption, spin flips down) rate γ₋ = 1/(2T1)
    Together: d⟨Iz⟩/dt = −⟨Iz⟩/T1  (assuming high-temperature ∴ Iz_eq ≈ 0).

  T2* (effective transverse, includes inhomogeneities):
    Decay of transverse magnetisation Mx, My.  Decomposed as:
      1/T2* = 1/(2T1) + 1/T2_pure_dephasing
    The pure-dephasing component is modelled by:
      L_z = 2Iz   rate γ_z = 1/T2* − 1/(2T1)
    Only added when γ_z > 0 (i.e. T2* < 2T1).

  T2 (intrinsic transverse, homogeneous):
    Same formula but using T2 instead of T2star; pass whichever is relevant.
"""

using LinearAlgebra

# ─── density_matrix ───────────────────────────────────────────────────────────

"""
    density_matrix(ψ) → Matrix{ComplexF64}

Build the N×N density matrix ρ = |ψ⟩⟨ψ| for a pure-state vector.

All NMR initial/target states in Pulsar are specified as pure-state vectors
(from `spin_state`). `LindbladMRControl` accepts them directly and calls this
function internally. You can also call it explicitly to inspect the density matrix.

# Example
```julia
sys  = mr_system("1H")
ψ    = spin_state(sys, :Iz)     # [1+0i, 0+0i] (|+z⟩)
ρ    = density_matrix(ψ)        # [1 0; 0 0]
```
"""
density_matrix(ψ::AbstractVector)::Matrix{ComplexF64} =
    ComplexF64.(ψ) * ComplexF64.(ψ)'

# ─── mr_relaxation ────────────────────────────────────────────────────────────

"""
    mr_relaxation(sys::MRSpinSystem;
                  T1     = Inf,
                  T2star = Inf,
                  T2     = Inf) → (jump_ops, decay_rates)

Build Lindblad jump operators and decay rates for NMR relaxation.

Returns `(jump_ops, decay_rates)` ready to pass to `LindbladMRControl`.
Each element of `jump_ops` is an N×N matrix; the matching element of
`decay_rates` is the corresponding rate in rad/s.

Operators with zero or non-finite rates are excluded from the output.

# Arguments
- `sys`    — `MRSpinSystem` from `mr_system`
- `T1`     — longitudinal relaxation time in seconds (default: `Inf`, no T1)
- `T2star` — effective transverse relaxation time in seconds (default: `Inf`)
- `T2`     — intrinsic transverse relaxation time in seconds (default: `Inf`)

Pass `T2` and `T2star` independently: if both are finite the most restrictive
(shorter) is used for the dephasing rate.

# Relaxation model per spin

| Channel | Jump operator | Rate               | Condition            |
|---------|---------------|--------------------|----------------------|
| T1      | `I₊` = `Ip`   | `1/(2T1)`          | `T1 < Inf`           |
| T1      | `I₋` = `Im`   | `1/(2T1)`          | `T1 < Inf`           |
| T2/T2*  | `2Iz`         | `1/T2eff − 1/(2T1)`| result > 0           |

where `T2eff = min(T2star, T2)`.

# Example
```julia
sys = mr_system("13C")

# T1 and T2* relaxation typical for ¹³C at 800 MHz
jump_ops, rates = mr_relaxation(sys; T1=2.0, T2star=0.05)

ctrl = LindbladMRControl(
    drifts      = drifts,
    operators   = [spin_op(sys,:Ix), spin_op(sys,:Iy)],
    rho_init    = [spin_state(sys, :Iz)],
    rho_targ    = [spin_state(sys, :mIz)],
    jump_ops    = jump_ops,
    decay_rates = rates,
    pwr_levels  = [2π * 600.0],
    pulse_dt    = fill(DT, N_TS),
)
```
"""
function mr_relaxation(
    sys    :: MRSpinSystem;
    T1     :: Float64 = Inf,
    T2star :: Float64 = Inf,
    T2     :: Float64 = Inf,
)
    jump_ops    = Matrix{ComplexF64}[]
    decay_rates = Float64[]

    for k in 1:sys.n_spins
        Ip_k = sys.Ip[k]
        Im_k = sys.Im[k]
        Iz_k = sys.Iz[k]

        # ── T1 longitudinal relaxation ────────────────────────────────────────
        # L₊ = I₊ and L₋ = I₋, each with rate γ = 1/(2T1).
        # Combined effect: d⟨Iz⟩/dt = −⟨Iz⟩/T1 (high-temperature limit).
        if isfinite(T1) && T1 > 0.0
            γ1 = 1.0 / (2.0 * T1)
            push!(jump_ops, Ip_k);  push!(decay_rates, γ1)
            push!(jump_ops, Im_k);  push!(decay_rates, γ1)
        elseif T1 <= 0.0
            throw(ArgumentError("T1 must be positive (got T1=$T1)"))
        end

        # ── T2 / T2* pure dephasing ───────────────────────────────────────────
        # Take the most restrictive transverse time.
        # Pure-dephasing rate = 1/T2eff − 1/(2T1);  only add if positive.
        T2eff = min(T2star, T2)
        if isfinite(T2eff)
            T2eff > 0.0 || throw(ArgumentError(
                "T2star / T2 must be positive (got T2star=$T2star, T2=$T2)"))
            γ_T1 = isfinite(T1) && T1 > 0.0 ? 1.0 / (2.0 * T1) : 0.0
            γ_z  = 1.0 / T2eff - γ_T1
            if γ_z > 0.0
                # L_z = 2Iz; prefactor 2 chosen so that the resulting
                # dephasing rate matches 1/T2eff exactly.
                push!(jump_ops, 2.0 .* Iz_k)
                push!(decay_rates, γ_z)
            elseif γ_z < -1e-12
                @warn "mr_relaxation: T2eff=$T2eff s > 2T1=$(2T1) s for spin $k; " *
                      "pure-dephasing rate γ_z = $γ_z < 0 — skipping T2 channel " *
                      "(T2* longer than 2T1 is unphysical in the Bloch model)."
            end
        end
    end

    return jump_ops, decay_rates
end
