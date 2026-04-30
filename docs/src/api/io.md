# API — IO

Pulse export, checkpointing, and the `Output` submodule.
Source: [`src/IO/`](https://github.com/DaJo2025/PULSAR.jl/tree/main/src/IO).

## Output / pulse export

```@docs
PULSAR.Output.OptimizedPulse
PULSAR.Output.export_pulse
PULSAR.Output.register_exporter
PULSAR.Output.replace_exporter
PULSAR.Output.list_exporters
PULSAR.Output.BrukerShape
PULSAR.Output.JEOLShape
PULSAR.Output.EPRShape
PULSAR.Output.PulseqSequence
PULSAR.Output.QiskitWaveformExport
PULSAR.Output.QuilTExport
PULSAR.Output.QUAExport
PULSAR.Output.PulserExport
```

## Bruker direct save

```@docs
save_bruker_shape
```

## Checkpointing

```@docs
Checkpoint
save_checkpoint
load_checkpoint
resume_optimization
create_checkpoint
auto_checkpoint_callback
checkpoint_compatible
list_checkpoints
checkpoint_summary
```
