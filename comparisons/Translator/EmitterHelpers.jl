"""
    comparisons/Translator/EmitterHelpers.jl

Small utilities shared by every native-script emitter.  The single place
where Pulsar's spin-state symbols (`:Iz`, `:mIy`, ...) are mapped to the
string/operator primitives of each target program.
"""

# ─── Spin-state name mapping ──────────────────────────────────────────────────

"""
    state_symbol_to(symbol::Symbol, target::Symbol, spin_idx::Int=1) -> String

Convert a Pulsar single-spin state symbol to the corresponding native token
for one target program.  `target` is one of `:spinach`, `:qutip`, `:qopt`,
`:simpson`, `:krotov_jl`, `:quantumcontrol`.  `spin_idx` is used by
targets that embed the spin index in the token (Spinach `"Lz",1`, SIMPSON
`"I1z"`, etc.).
"""
function state_symbol_to(symbol::Symbol, target::Symbol, spin_idx::Int = 1)::String
    _assert_known_symbol(symbol)
    target === :spinach        && return _to_spinach(symbol, spin_idx)
    target === :simpson        && return _to_simpson(symbol, spin_idx)
    target === :qutip          && return _to_qutip(symbol, spin_idx)
    target === :qopt           && return _to_qopt(symbol, spin_idx)
    target === :krotov_jl      && return _to_krotov_jl(symbol, spin_idx)
    target === :quantumcontrol && return _to_qc(symbol, spin_idx)
    throw(ArgumentError("Unknown emission target :$target"))
end

const _KNOWN_STATE_SYMS = (:Iz, :mIz, :Ix, :mIx, :Iy, :mIy)

_assert_known_symbol(sym::Symbol) =
    sym ∈ _KNOWN_STATE_SYMS ||
        throw(ArgumentError(
            "Unknown state symbol :$sym. Supported: $(_KNOWN_STATE_SYMS)"))

# ─── Spinach (MATLAB) ─────────────────────────────────────────────────────────

function _to_spinach(sym::Symbol, spin_idx::Int)::String
    sym === :Iz  && return "state(spin_system,'Lz',$(spin_idx))"
    sym === :mIz && return "-state(spin_system,'Lz',$(spin_idx))"
    sym === :Ix  && return "state(spin_system,'Lx',$(spin_idx))"
    sym === :mIx && return "-state(spin_system,'Lx',$(spin_idx))"
    sym === :Iy  && return "state(spin_system,'Ly',$(spin_idx))"
    sym === :mIy && return "-state(spin_system,'Ly',$(spin_idx))"
    error("unreachable")
end

# ─── SIMPSON (Tcl) ────────────────────────────────────────────────────────────

function _to_simpson(sym::Symbol, spin_idx::Int)::String
    # SIMPSON tokens: I1z, I1x, I1y, with "-" prefix for negation.
    base = "I$(spin_idx)"
    sym === :Iz  && return "$(base)z"
    sym === :mIz && return "-$(base)z"
    sym === :Ix  && return "$(base)x"
    sym === :mIx && return "-$(base)x"
    sym === :Iy  && return "$(base)y"
    sym === :mIy && return "-$(base)y"
    error("unreachable")
end

# ─── QuTiP (Python) ───────────────────────────────────────────────────────────

# QuTiP single-spin operators at the single-spin (2-level) level.
# For multi-spin problems emitters should wrap each per-spin state in a kron.
function _to_qutip(sym::Symbol, spin_idx::Int)::String
    # State vectors, not operators — emitters use these to build |ψ⟩.
    sym === :Iz  && return "qutip.basis(2,0)"
    sym === :mIz && return "qutip.basis(2,1)"
    sym === :Ix  && return "(qutip.basis(2,0) + qutip.basis(2,1)).unit()"
    sym === :mIx && return "(qutip.basis(2,0) - qutip.basis(2,1)).unit()"
    sym === :Iy  && return "(qutip.basis(2,0) + 1j*qutip.basis(2,1)).unit()"
    sym === :mIy && return "(qutip.basis(2,0) - 1j*qutip.basis(2,1)).unit()"
    error("unreachable")
end

# ─── qopt (Python) ────────────────────────────────────────────────────────────

function _to_qopt(sym::Symbol, spin_idx::Int)::String
    sym === :Iz  && return "np.array([1.0, 0.0], dtype=complex)"
    sym === :mIz && return "np.array([0.0, 1.0], dtype=complex)"
    sym === :Ix  && return "np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)"
    sym === :mIx && return "np.array([1.0,-1.0], dtype=complex) / np.sqrt(2)"
    sym === :Iy  && return "np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2)"
    sym === :mIy && return "np.array([1.0,-1.0j], dtype=complex) / np.sqrt(2)"
    error("unreachable")
end

# ─── Krotov.jl (Julia subprocess script) ──────────────────────────────────────

function _to_krotov_jl(sym::Symbol, spin_idx::Int)::String
    sym === :Iz  && return "ComplexF64[1, 0]"
    sym === :mIz && return "ComplexF64[0, 1]"
    sym === :Ix  && return "ComplexF64[1, 1] / sqrt(2)"
    sym === :mIx && return "ComplexF64[1,-1] / sqrt(2)"
    sym === :Iy  && return "ComplexF64[1, im] / sqrt(2)"
    sym === :mIy && return "ComplexF64[1,-im] / sqrt(2)"
    error("unreachable")
end

# ─── QuantumControl.jl (in-process) ───────────────────────────────────────────

# QuantumControl is in-process Julia; emitter calls `build_state_vector` on
# the Pulsar MRSpinSystem rather than emitting a string.  This function exists
# for API symmetry — it returns the Julia source string a reader would write.
function _to_qc(sym::Symbol, spin_idx::Int)::String
    sym === :Iz  && return "ComplexF64[1, 0]"
    sym === :mIz && return "ComplexF64[0, 1]"
    sym === :Ix  && return "ComplexF64[1, 1] / sqrt(2)"
    sym === :mIx && return "ComplexF64[1,-1] / sqrt(2)"
    sym === :Iy  && return "ComplexF64[1, im] / sqrt(2)"
    sym === :mIy && return "ComplexF64[1,-im] / sqrt(2)"
    error("unreachable")
end

# ─── Axis → operator name ─────────────────────────────────────────────────────

"""
    axis_symbol_to(axis::Symbol, target::Symbol, spin_idx::Int=1) -> String

Convert a control axis (`:x`, `:y`, `:z`) to the target program's operator
token for the corresponding spin.
"""
function axis_symbol_to(axis::Symbol, target::Symbol, spin_idx::Int = 1)::String
    axis ∈ (:x, :y, :z) ||
        throw(ArgumentError("axis must be :x, :y, :z (got :$axis)"))
    letter = String(axis)
    target === :spinach &&
        return "operator(spin_system,{'L$(letter)'},{$(spin_idx)})"
    target === :simpson && return "I$(spin_idx)$(letter)"
    target === :qutip   && return "qutip.sigma$(letter)()/2"
    target === :qopt    && return "0.5 * sigma_$(letter)"
    target === :krotov_jl &&
        return "0.5 * ComplexF64" *
               (axis === :x ? "[0 1; 1 0]" :
                axis === :y ? "[0 -im; im 0]" :
                              "[1 0; 0 -1]")
    throw(ArgumentError("axis_symbol_to: unknown target :$target"))
end

# ─── Isotope → Larmor frequency (Hz at B0) ────────────────────────────────────

"""
    larmor_hz(isotope, B0_tesla) -> Float64

Return the Larmor frequency (Hz) for `isotope` at field `B0_tesla`.
"""
function larmor_hz(isotope::String, B0_tesla::Float64)::Float64
    haskey(GYRO_MHZ_PER_T, isotope) ||
        throw(ArgumentError("Unknown isotope '$isotope' for larmor_hz"))
    return abs(GYRO_MHZ_PER_T[isotope]) * 1e6 * B0_tesla
end

# ─── Universal-rotation helpers ───────────────────────────────────────────────

"""
    is_universal_rotation(ann) -> Bool

True if the target is a universal rotation (3 orthogonal state pairs encoded
by `TargetAnnotation.kind == :universal_rotation`).
"""
is_universal_rotation(ann::PhysicsAnnotation) = ann.target.kind == :universal_rotation

# ─── Waveform file conventions (documentation) ────────────────────────────────
# Every emitter writes a waveform file with rows = time steps, columns = control
# channels, values in one of the conventions below.  Drivers decode with
# `parse_waveform_file(path, n_ctrl, n_t; convention=..., pwr_levels=...)`.
#
#   :rad_per_sec   — raw H_ctrl coefficient, divide by pwr_levels to normalise
#   :hz            — same as :rad_per_sec but in Hz; driver multiplies by 2π
#   :normalised    — already in [l_bound, u_bound]; no transform
