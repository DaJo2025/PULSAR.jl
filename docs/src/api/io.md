# API — IO

Pulse export, checkpointing, and the `Output` submodule.
Source: [`src/IO/`](https://github.com/DaJo2025/Pulsar.jl/tree/main/src/IO).

## Output / pulse export

```@docs
Pulsar.Output.OptimizedPulse
Pulsar.Output.export_pulse
Pulsar.Output.register_exporter
Pulsar.Output.replace_exporter
Pulsar.Output.list_exporters
Pulsar.Output.BrukerShape
Pulsar.Output.JEOLShape
Pulsar.Output.EPRShape
Pulsar.Output.PulseqSequence
Pulsar.Output.QiskitWaveformExport
Pulsar.Output.QuilTExport
Pulsar.Output.QUAExport
Pulsar.Output.PulserExport
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
