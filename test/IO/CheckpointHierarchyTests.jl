# test/IO/CheckpointHierarchyTests.jl
# ====================================
# Exercises the unified Checkpoint type.
#
#   • Checkpoint <: AbstractCheckpoint.
#   • Round-trips losslessly via Julia Serialization (.jls).
#   • Auto-converts legacy MRCheckpoint/QCCheckpoint payloads on disk
#     so existing checkpoint files keep loading after consolidation.

using Test
using PULSAR
using Serialization: serialize

@testset "Unified Checkpoint" begin
    w_opt = randn(2, 50)
    F_opt = 0.987

    @testset "Type hierarchy" begin
        @test Checkpoint <: AbstractCheckpoint
    end

    @testset "Kwargs constructor + warm-start defaults" begin
        c = Checkpoint(w_opt, F_opt, 2, 50)
        @test c.iteration == 0
        @test isempty(c.fidelity_history)
        @test isempty(c.gradient_norm_history)
        @test isempty(c.optimizer_state)
        @test c.system_dim == 0
        @test c.domain == :generic
        @test c.system_kind == :generic
        @test c.metadata isa Dict{String,Any}
        @test checkpoint_compatible(c, 2, 50)
        @test !checkpoint_compatible(c, 3, 50)
    end

    @testset "Round-trip (.jls)" begin
        fid_hist = collect(range(0.5, 0.987; length = 20))
        gnorm    = collect(range(0.2, 0.001; length = 20))
        c = Checkpoint(
            w_opt, F_opt, 2, 50;
            iteration             = 200,
            fidelity_history      = fid_hist,
            gradient_norm_history = gnorm,
            optimizer_state       = Dict{String,Any}("method" => "lbfgs"),
            system_dim            = 4,
            domain                = :qc,
            drive_max_hz          = 200e6,
            T_pulse               = 5e-8,
            system_kind           = :transmon,
            metadata              = Dict{String,Any}("note" => "test"),
        )
        fp = tempname() * ".jls"
        save_checkpoint(fp, c)
        loaded = load_checkpoint(fp)

        @test loaded isa Checkpoint
        @test loaded.w_opt == c.w_opt
        @test loaded.F_opt == c.F_opt
        @test loaded.n_controls == 2
        @test loaded.n_timesteps == 50
        @test loaded.iteration == 200
        @test loaded.fidelity_history == fid_hist
        @test loaded.gradient_norm_history == gnorm
        @test loaded.optimizer_state["method"] == "lbfgs"
        @test loaded.system_dim == 4
        @test loaded.domain == :qc
        @test loaded.drive_max_hz == 200e6
        @test loaded.T_pulse == 5e-8
        @test loaded.system_kind == :transmon
        @test loaded.metadata["note"] == "test"
        rm(fp)
    end

    @testset "load_checkpoint auto-converts legacy MR/QC payloads" begin
        mr = PULSAR.MRCheckpoint(copy(w_opt), F_opt, 50, 2, 25e3, 5e-3,
                                  "2026-04-29T00:00:00",
                                  Dict{String,Any}("source" => "legacy"))
        fp = tempname() * ".jls"
        open(fp, "w") do io; serialize(io, mr); end
        loaded = load_checkpoint(fp)
        @test loaded isa Checkpoint
        @test loaded.w_opt == mr.w_opt
        @test loaded.domain == :mr
        @test loaded.drive_max_hz == 25e3
        @test loaded.metadata["source"] == "legacy"
        rm(fp)

        qc = PULSAR.QCCheckpoint(copy(w_opt), F_opt, 50, 2, 200e6, 5e-8,
                                  :transmon, "2026-04-29T00:00:00",
                                  Dict{String,Any}())
        fp = tempname() * ".jls"
        open(fp, "w") do io; serialize(io, qc); end
        loaded = load_checkpoint(fp)
        @test loaded isa Checkpoint
        @test loaded.domain == :qc
        @test loaded.system_kind == :transmon
        rm(fp)
    end

    @testset "load_checkpoint missing file → ArgumentError" begin
        @test_throws ArgumentError load_checkpoint(tempname() * ".jls")
    end

    @testset "list_checkpoints" begin
        dir = mktempdir()
        try
            c = Checkpoint(w_opt, F_opt, 2, 50)
            save_checkpoint(joinpath(dir, "a.jls"), c)
            save_checkpoint(joinpath(dir, "b.jls"), c)
            files = list_checkpoints(dir)
            @test length(files) == 2
            @test all(endswith(f, ".jls") for f in files)
        finally
            rm(dir; recursive = true, force = true)
        end
    end
end
