"""
    comparisons/Translator/Translator.jl

Entry point that loads the full Translator layer:
  - `PhysicsAnnotation.jl`  — solver-agnostic problem description + matrix rebuild
  - `Capabilities.jl`       — per-solver capability checks
  - `EmitterHelpers.jl`     — Pulsar ↔ native token mapping
  - `Subprocess.jl`         — shared subprocess + waveform I/O

Emitters live under `Emitters/` and are loaded by the individual drivers that
need them.
"""

include(joinpath(@__DIR__, "PhysicsAnnotation.jl"))
include(joinpath(@__DIR__, "TransmonAnnotation.jl"))
include(joinpath(@__DIR__, "Capabilities.jl"))
include(joinpath(@__DIR__, "EmitterHelpers.jl"))
include(joinpath(@__DIR__, "Subprocess.jl"))
