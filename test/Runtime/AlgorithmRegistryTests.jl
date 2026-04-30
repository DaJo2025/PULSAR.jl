# test/Runtime/AlgorithmRegistryTests.jl
# ======================================
# Theme 8 — exercises the OptimizerEntry / OPTIMIZER_REGISTRY abstractions
# and the built-in registrations populated at module-load time.

using Test
using PULSAR

@testset "AlgorithmRegistry (Theme 8)" begin

    @testset "Built-in registrations are present" begin
        for name in (:grape, :bfgs, :lbfgs, :newton, :cmaes,
                     :nelder_mead, :pso, :constrained_grape,
                     :robust_grape, :trust_region)
            @test is_registered(name)
            e = get_optimizer(name)
            @test e isa OptimizerEntry
            @test e.name == name
            @test e.callable isa Function
            @test e.description isa String && !isempty(e.description)
        end
    end

    @testset "Capability filters" begin
        # gradient-based optimizers
        grad_opts = list_optimizers(gradient = true)
        @test :grape  in grad_opts
        @test :lbfgs  in grad_opts
        @test :newton in grad_opts
        @test !(:cmaes in grad_opts)
        @test !(:pso   in grad_opts)
        # derivative-free
        @test :cmaes        in list_optimizers(gradient = false)
        @test :nelder_mead  in list_optimizers(gradient = false)
        @test :pso          in list_optimizers(gradient = false)
        # bounds-aware
        @test :constrained_grape in list_optimizers(bounds = true)
        # noise-tolerant
        noise_opts = list_optimizers(noise = true)
        @test :cmaes        in noise_opts
        @test :pso          in noise_opts
        @test :robust_grape in noise_opts
    end

    @testset "Idempotent re-registration" begin
        e = get_optimizer(:lbfgs)
        @test register_optimizer!(e) === e        # no-op
    end

    @testset "Conflicting re-registration without replace=true" begin
        e0 = get_optimizer(:lbfgs)
        bogus = OptimizerEntry(:lbfgs, +,
            (gradient=true, hessian=false, bounds=false, noise=false,
             parallel=false, qoc=false, generic=true, open_system=false),
            "intentionally wrong")
        @test_throws ArgumentError register_optimizer!(bogus)
        # original still intact
        @test get_optimizer(:lbfgs).description == e0.description
        # replace=true succeeds, then restore for downstream tests
        register_optimizer!(bogus; replace = true)
        @test get_optimizer(:lbfgs).description == "intentionally wrong"
        register_optimizer!(e0; replace = true)
        @test get_optimizer(:lbfgs).description == e0.description
    end

    @testset "Unknown optimizer throws KeyError" begin
        @test_throws KeyError get_optimizer(:does_not_exist)
        @test !is_registered(:does_not_exist)
    end

    @testset "open_system filter (Theme 6b)" begin
        # Algorithms reachable on LindbladMRControl / LindbladQCControl via
        # the application-layer kernel dispatch.
        open_opts = list_optimizers(open_system = true)
        for name in (:grape, :lbfgs, :cmaes, :nelder_mead, :pso)
            @test name in open_opts
        end
        # Hilbert-only QOC wrappers must not be reported as open-system safe.
        closed_opts = list_optimizers(open_system = false)
        for name in (:bfgs, :newton, :constrained_grape,
                     :robust_grape, :trust_region)
            @test name in closed_opts
        end
        # Combined filter: gradient + open-system reachable.
        grad_open = list_optimizers(gradient = true, open_system = true)
        @test :grape in grad_open
        @test :lbfgs in grad_open
        @test !(:cmaes in grad_open)        # gradient-free
    end
end
