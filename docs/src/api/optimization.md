# API — Optimization

All optimizers, configurations, and result types. Source:
[`src/Optimization/`](https://github.com/DaJo2025/PULSAR.jl/tree/main/src/Optimization).

## Result type

```@docs
OptimizationResult
```

## GRAPE family

```@docs
grape_optimize
grape_cg_optimize
grape_lbfgsb_optimize
GRAPEConfig
```

## Second-order

```@docs
bfgs_optimize
lbfgs_optimize
lbfgsb_optimize
newton_optimize
gauss_newton_optimize
lm_optimize
trust_region_newton_optimize
projected_gradient_optimize
BFGSConfig
LBFGSConfig
NewtonConfig
```

## Direct search

```@docs
nelder_mead_optimize
hooke_jeeves_optimize
compass_search_optimize
powell_dirset_optimize
uobyqa_optimize
newuoa_optimize
bobyqa_optimize
cobyla_optimize
lincoa_optimize
NelderMeadConfig
```

## Metaheuristic

```@docs
ga_optimize
sa_optimize
mcsa_optimize
ssmc_optimize
pso_optimize
de_optimize
cmaes_optimize
pscmaes_optimize
mc_random_search
grid_search
basin_hopping_optimize
CMAESConfig
PSOConfig
```

## QOC-specific

```@docs
krotov_optimize
krotov_second_order_optimize
group_optimize
goat_optimize
oc_trust_region_newton_optimize
oc_semismooth_newton_optimize
```

## Constrained

```@docs
constrained_optimize
ConstrainedConfig
BoundConstraint
PowerConstraint
BandwidthConstraint
EnergyConstraint
CustomConstraint
```

## Robust

```@docs
robust_optimize
RobustConfig
```

## Multi-objective

```@docs
multi_objective_optimize
MultiObjectiveConfig
MultiObjectiveResult
energy_objective
smoothness_objective
peak_amplitude_objective
fidelity_objective
```

## Adaptive step size

```@docs
AdamConfig
AdamState
```

## Generic gradient methods

```@docs
gd_optimize
sgd_optimize
momentum_optimize
nag_optimize
adagrad_optimize
rmsprop_optimize
adam_optimize
cg_optimize
```

## Analytic pulses

```@docs
CompositePulseSegment
AnalyticPulse
bb1
scrofulous
sk1
corpse
short_corpse
f1
g1
corpse_in_bb1
drag_pulse
sta_fourier_1d
slr_1d
verse
verse_min_time
verse_acoustic_noise
```
