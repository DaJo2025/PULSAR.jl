"""
    comparisons/Translator/Capabilities.jl

Per-solver capability declarations and generic `check_supported(caps, ann)`.
"""

"""
    SolverCapabilities

Boolean flags describing which physics features a given solver supports.
Checked against an [`AbstractPhysicsAnnotation`](@ref) before the driver
emits any native script, so unsupported problems short-circuit with a
specific reason.

`qc_transmon` gates the transmon-style QC benchmarks (BM07+); NMR emitters
leave it at the default `false` and decline those problems cleanly.
"""
Base.@kwdef struct SolverCapabilities
    multi_spin        :: Bool = true
    ensemble          :: Bool = true
    multi_state_pair  :: Bool = true
    lindblad          :: Bool = false
    multichannel      :: Bool = true   # more than 2 control channels
    nonuniform_dt     :: Bool = true
    heteronuclear     :: Bool = true
    csa               :: Bool = false
    dipolar           :: Bool = false
    j_coupling        :: Bool = true
    amplitude_bounds  :: Bool = true
    qc_transmon       :: Bool = false  # superconducting-qubit (BM07+)
end

"""
    check_supported(caps, ann) -> Union{Nothing,String}

Return `nothing` when the annotation fits the capabilities; otherwise a
one-line reason string naming the missing capability.  Dispatches on the
annotation type — NMR (`PhysicsAnnotation`) and QC (`TransmonAnnotation`)
take different code paths.
"""
function check_supported(caps::SolverCapabilities,
                          ann::PhysicsAnnotation)::Union{Nothing,String}
    if length(ann.spins) > 1 && !caps.multi_spin
        return "Driver does not support multi-spin problems (annotation has $(length(ann.spins)) spins)."
    end
    if ann.sweep !== nothing && length(ann.sweep.offsets_hz) > 1 && !caps.ensemble
        return "Driver does not support ensemble problems (annotation has $(length(ann.sweep.offsets_hz)) ensemble members)."
    end
    if length(ann.target.initial_states) > 1 && !caps.multi_state_pair
        return "Driver does not support multi-state-pair objectives (annotation has $(length(ann.target.initial_states)) pairs)."
    end
    if ann.target.kind == :lindblad_state_transfer && !caps.lindblad
        return "Driver does not support open-system (Lindblad) dynamics."
    end
    if length(ann.controls) > 2 && !caps.multichannel
        return "Driver does not support more than 2 control channels (annotation has $(length(ann.controls)))."
    end
    isos = Set(isotopes_vec(ann))
    if length(isos) > 1 && !caps.heteronuclear
        return "Driver does not support heteronuclear problems (annotation has isotopes $(collect(isos)))."
    end
    if any(s -> s.csa !== nothing, ann.spins) && !caps.csa
        return "Driver does not support chemical-shift anisotropy (CSA)."
    end
    for c in ann.couplings
        c.kind == :dipolar && !caps.dipolar &&
            return "Driver does not support dipolar couplings."
        c.kind == :j_isotropic && !caps.j_coupling &&
            return "Driver does not support J-couplings."
    end
    return nothing
end

function check_supported(caps::SolverCapabilities,
                          ann::TransmonAnnotation)::Union{Nothing,String}
    caps.qc_transmon ||
        return "Driver does not support transmon-style QC problems (qc_transmon=false)."
    # Gate objectives are handled natively by Quandary's own basis-state scan;
    # we do not require multi_state_pair = true at the capability level.
    return nothing
end
