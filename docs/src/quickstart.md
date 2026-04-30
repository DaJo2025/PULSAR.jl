# Quickstart — single-qubit state transfer

This walkthrough drives the canonical Pulsar optimization loop: build a system,
specify a target, initialize a control sequence, configure GRAPE, run, and
re-simulate the optimum to verify.

## The problem

Drive a single qubit from |0⟩ to |1⟩ in time `T = π` under
`H(t) = (ω₀/2) σz + uₓ(t) σx/2 + u_y(t) σy/2` with a small off-resonance shift
`ω₀ = 2π · 10 Hz` and an RF amplitude budget `|u| ≤ 1` rad/s.

## The full code

```julia
using Pulsar
using LinearAlgebra, Random

# §1 Pauli matrices
σx = ComplexF64[0 1; 1 0]
σy = ComplexF64[0 -im; im 0]
σz = ComplexF64[1 0; 0 -1]

# §2 1-qubit system: Larmor drift + x/y quadrature drives
ω_offset_hz = 10.0
H_drift = (2π * ω_offset_hz / 2) .* σz
sys     = qubit_system(1, H_drift, [0.5σx, 0.5σy])

# §3 State-transfer target |0⟩ → |1⟩
ψ_init   = ComplexF64[1.0, 0.0]
ψ_target = ComplexF64[0.0, 1.0]
target   = state_target(ψ_target; psi_init = ψ_init)

# §4 Random initial control sequence (100 piecewise-constant slices, T = π)
ctrl = random_controls(sys, π, 100; amplitude=0.5,
                        rng=MersenneTwister(42))

# §5 GRAPE
config = GRAPEConfig(
    max_iter        = 500,
    step_size       = 0.05,
    adapt_step_size = true,
    convergence_tol = 1e-9,
    verbose         = false,
)
result = grape_optimize(sys, target, ctrl; config=config)

# §6 Independent re-simulation of the optimum
ctrl_opt = ControlSequence(Matrix(result.controls'), ctrl.dt, ctrl.n_steps)
F_check  = compute_fidelity(sys, ctrl_opt, target)
@assert abs(F_check - result.fidelity) < 1e-6 "Fidelity mismatch!"

println("Final fidelity: ", round(F_check; digits=8))
```

## Walking through it

### Hamiltonian conventions

Drift and control Hamiltonians are in **angular frequency units (rad/s)**. To
convert from Hz, multiply by `2π`. Time evolution is `U = exp(-i H dt)` with
`ħ = 1`. See [Hamiltonians](theory/hamiltonians.md) for the full convention.

### `qubit_system`

Constructs a `QubitSystem` (a `QuantumSystem` subtype) carrying a drift
Hamiltonian and a vector of control Hamiltonians.

### `state_target` / `unitary_target`

Wraps a target state vector or unitary in a `QuantumTarget`. The optimizer
infers the right fidelity metric — `|⟨ψ_target|ψ⟩|²` for state targets,
`|Tr(U_target† U)|² / dim²` for unitaries.

### `random_controls`

Returns a `ControlSequence` with `[n_steps × n_controls]` waveform array,
each entry uniformly sampled in `[-amplitude, +amplitude]`.

### `grape_optimize` and `GRAPEConfig`

Runs first-order GRAPE with optional step-size adaptation. The result has:

- `result.fidelity` — final fidelity
- `result.controls` — `[n_controls × n_steps]` waveform (note the **transposed
  layout** vs `ControlSequence`)
- `result.fidelity_history` / `result.gradient_norm_history` — convergence trace
- `result.converged`, `result.termination_reason`, `result.n_iterations`

### Re-simulation

Every example in Pulsar follows a strict rule: independently re-simulate the
optimized control sequence with `compute_fidelity` and `@assert` agreement to
≤ 1e-6. This catches subtle bugs in optimizer-internal kernels and is required
in every benchmark/comparison driver.

```julia
ctrl_opt = ControlSequence(Matrix(result.controls'), ctrl.dt, ctrl.n_steps)
F_check  = compute_fidelity(sys, ctrl_opt, target)
@assert abs(F_check - result.fidelity) < 1e-6
```

## Where to go next

- Use a more capable optimizer: [Second-order methods](algorithms/second_order.md)
- Add an open-system noise model: [Lindblad](theory/propagators.md#lindblad-evolution)
- Make it robust to parameter drift: [Robust optimization](algorithms/robust.md)
- Constrain pulse amplitude / bandwidth: [Constrained optimization](algorithms/constrained.md)
- See full domain examples: [NMR](domains/nmr.md), [QC platforms](domains/qc_platforms.md)
