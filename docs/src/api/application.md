# API — Application

Domain wrappers for MR (NMR / EPR / MAS / MRI / DNP) and quantum-computing
platforms. Source:
[`src/Application/`](https://github.com/DaJo2025/PULSAR.jl/tree/main/src/Application).

## MR layer — generic

```@docs
AbstractMRControl
MRControl
optimcon
LindbladMRControl
mr_relaxation
density_matrix
grape_state_kernel
grape_lindblad_kernel
grape_tracking_kernel
fidelity_forward
TrackingPoint
```

## MR domain extensions

```@docs
optimcon_dnp
```

## Quantum-computing — gates

```@docs
single_qubit_gate_set
two_qubit_gate_set
NativeGateSet
native_gate_set
zyz_decompose
zyz_sequence
gate_infidelity
```

## Quantum-computing — noise models

```@docs
QuasiStaticNoise
quasi_static_ensemble
robust_optimcon_qs
evaluate_qs_robustness
MarkovianNoise
markovian_noise
amplitude_damping
phase_damping
depolarizing_channel
lindblad_optimcon
NoiseSpectrum
pink_noise_spectrum
white_noise_spectrum
ohmic_noise_spectrum
compute_filter_function
filter_function_infidelity
optimcon_ff
```

## Verification

```@docs
RBResult
rb_sequence
rb_survival_probability
fit_rb_decay
estimate_epc
interleaved_rb
```
