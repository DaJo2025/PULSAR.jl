# Automatic differentiation

Pulsar provides automatic-differentiation gradients as an alternative to the
analytic GRAPE kernel. AD is useful for:

- Verifying analytic gradients during development
- Custom objectives that don't yet have an analytic gradient
- Rapid prototyping of new penalty terms

Source: [`src/Physics/AutoDiff.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/src/Physics/AutoDiff.jl).

## Backends

| Backend | Mode | Activate |
|---|---|---|
| `ForwardDiff.jl` | Forward-mode, dual numbers | `pkg> add ForwardDiff` |
| `Zygote.jl` | Reverse-mode, source-to-source | `pkg> add Zygote` |

Each is wired through a Julia 1.9 package extension; the corresponding
extension is loaded as soon as the backing package is in your environment.

| Extension | Source |
|---|---|
| `PulsarForwardDiffExt` | [`ext/PulsarForwardDiffExt.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/ext/PulsarForwardDiffExt.jl) |
| `PulsarZygoteExt`      | [`ext/PulsarZygoteExt.jl`](https://github.com/DaJo2025/Pulsar.jl/blob/main/ext/PulsarZygoteExt.jl) |

## API

`compute_gradient_autodiff` and `verify_gradient_autodiff` follow the
canonical `(system, controls, target)` argument order:

```julia
g_ad = compute_gradient_autodiff(sys, ctrl, target;
                                  config = AutoDiffConfig(backend = :forwarddiff))
```

For a paranoia / regression check against the analytic gradient:

```julia
ok = verify_gradient_autodiff(sys, ctrl, target; tol = 1e-7, verbose = true)
```

Both routines fall back to a finite-difference gradient
(`finite_difference_gradient`) if no AD backend is available.

## When to use AD vs analytic

- **Analytic** (`compute_grape_gradient`): fastest, available for the standard
  fidelity / penalty combinations. Always prefer this in production.
- **Forward AD**: small parameter counts; dimension-light per-parameter cost
- **Reverse AD**: many parameters, single objective — but Pulsar's reverse
  kernels are still the tighter choice for `n_steps × n_c > ~1000`
- **Finite differences**: only for verification, never optimization

## Caveats

- `Zygote.jl` does not differentiate every Julia construct (in particular,
  mutating `Array` operations). If reverse AD fails, fall back to ForwardDiff
  for that objective.
- AD-built gradients are typically 5–20× slower than the analytic GRAPE
  kernel; use them as ground-truth, not as the optimizer's gradient source.
