"""
    comparisons/Translator/tests/test_annotation_roundtrip.jl

Round-trip check: for every benchmark problem that has a `PhysicsAnnotation`,
`build_ctrl_from_annotation(p.physics)` must produce drifts, operators,
state vectors, pwr_levels and pulse_dt equal (to 1e-12) to the manually
authored `p.ctrl`.  This is the anchor that keeps emitters and canonical
re-evaluation consistent.
"""

using Test
using LinearAlgebra
using PULSAR

include(joinpath(@__DIR__, "..", "..", "Problems", "all_problems.jl"))

const RTOL = 1e-12

function _same_matrix_list(a, b; rtol=RTOL)
    length(a) == length(b) || return false
    for k in eachindex(a)
        norm(a[k] .- b[k]) <= rtol * max(1.0, norm(a[k])) || return false
    end
    return true
end

function _same_vec_list(a, b; rtol=RTOL)
    length(a) == length(b) || return false
    for k in eachindex(a)
        norm(a[k] .- b[k]) <= rtol * max(1.0, norm(a[k])) || return false
    end
    return true
end

@testset "PhysicsAnnotation round-trip" begin
    for p in ALL_PROBLEMS
        p.physics === nothing && continue
        @testset "$(p.id)" begin
            rebuilt = build_ctrl_from_annotation(p.physics)

            @test _same_matrix_list(rebuilt.drifts, p.ctrl.drifts)
            @test _same_matrix_list(rebuilt.operators, p.ctrl.operators)
            @test _same_vec_list(rebuilt.rho_init, p.ctrl.rho_init)
            @test _same_vec_list(rebuilt.rho_targ, p.ctrl.rho_targ)

            @test length(rebuilt.pwr_levels) == length(p.ctrl.pwr_levels)
            @test maximum(abs.(rebuilt.pwr_levels .- p.ctrl.pwr_levels)) <= RTOL
            @test length(rebuilt.pulse_dt) == length(p.ctrl.pulse_dt)
            @test maximum(abs.(rebuilt.pulse_dt .- p.ctrl.pulse_dt)) <= RTOL

            if p.ctrl isa LindbladMRControl
                @test rebuilt isa LindbladMRControl
                @test _same_matrix_list(rebuilt.jump_ops, p.ctrl.jump_ops)
                @test maximum(abs.(rebuilt.decay_rates .- p.ctrl.decay_rates)) <= RTOL
            else
                @test rebuilt isa MRControl
            end
        end
    end
end
