@testset "Backend Fallback" begin

    # ─── CUDA fallback ───────────────────────────────────────────────────────
    @testset "CUDABackend graceful fallback" begin
        if !PULSAR._CUDA_LOADED[]
            # Requesting a CUDA backend without CUDA.jl should return cpu or warn.
            prev = get_device()
            try
                # set_device! with :cuda when unavailable should either fall back
                # silently or throw a descriptive error — not a hard crash.
                local caught = false
                try
                    set_device!(:cuda)
                catch e
                    caught = true
                    @test occursin("cuda", lowercase(string(e)))
                end
                # If it didn't throw, it should have fallen back to :cpu
                caught || @test get_device() == :cpu
            finally
                set_device!(prev)
            end
        else
            @info "CUDA.jl is loaded; skipping CUDABackend fallback test"
        end
    end

    # ─── Metal fallback ──────────────────────────────────────────────────────
    @testset "MetalBackend graceful fallback" begin
        if !PULSAR._METAL_LOADED[]
            prev = get_device()
            try
                local caught = false
                try
                    set_device!(:metal)
                catch e
                    caught = true
                    @test occursin("metal", lowercase(string(e)))
                end
                caught || @test get_device() == :cpu
            finally
                set_device!(prev)
            end
        else
            @info "Metal.jl is loaded; skipping MetalBackend fallback test"
        end
    end

    # ─── CPU backend always available ────────────────────────────────────────
    @testset "CPUBackend always available" begin
        prev = get_device()
        set_device!(:cpu)
        @test get_device() == :cpu
        set_device!(prev)
    end

    # ─── available_devices returns at least :cpu ─────────────────────────────
    @testset "available_devices includes :cpu" begin
        devs = available_devices()
        @test :cpu ∈ devs
    end

end  # @testset "Backend Fallback"
