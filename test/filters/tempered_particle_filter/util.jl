using StateSpaceRoutines, Test, Distributions, LinearAlgebra

Random.seed!(47)

@testset "bisection" begin

    @testset "Finds root of simple function" begin
        f = x -> x^2 - 4.0
        root = StateSpaceRoutines.bisection(f, 0.0, 3.0; xtol = 1e-8)
        @test abs(root - 2.0) < 1e-7
    end

    @testset "Finds root of linear function" begin
        root = StateSpaceRoutines.bisection(x -> x - 0.5, 0.0, 1.0; xtol = 1e-10)
        @test abs(root - 0.5) < 1e-9
    end

    @testset "Tighter xtol gives more accurate result" begin
        f = x -> x^3 - 2.0
        root_loose = StateSpaceRoutines.bisection(f, 0.0, 2.0; xtol = 1e-2)
        root_tight = StateSpaceRoutines.bisection(f, 0.0, 2.0; xtol = 1e-8)
        @test abs(root_tight - 2.0^(1/3)) < abs(root_loose - 2.0^(1/3))
    end

    @testset "Throws when no root in interval" begin
        @test_throws String StateSpaceRoutines.bisection(x -> x^2 + 1.0, 0.0, 2.0)
    end

end

@testset "fast_mvnormal_pdf" begin

    @testset "Scalar standard normal matches Distributions.jl" begin
        for x in [-2.0, -1.0, 0.0, 0.5, 1.5]
            @test StateSpaceRoutines.fast_mvnormal_pdf(x) ≈ pdf(Normal(), x)
        end
    end

    @testset "Vector standard normal matches Distributions.jl" begin
        for x in [[0.0], [1.0, -1.0], [0.5, 0.5, -0.5]]
            @test StateSpaceRoutines.fast_mvnormal_pdf(x) ≈ pdf(MvNormal(zeros(length(x)), I), x)
        end
    end

    @testset "Vector with Σ matches Distributions.jl" begin
        Σ = [2.0 0.5; 0.5 1.0]
        inv_Σ = inv(Σ)
        det_Σ = det(Σ)
        x = [1.0, -0.5]
        @test StateSpaceRoutines.fast_mvnormal_pdf(x, det_Σ, inv_Σ) ≈ pdf(MvNormal(zeros(2), Σ), x)
    end

    @testset "Scalar with Σ matches Distributions.jl" begin
        σ² = 3.0
        inv_Σ = fill(1.0/σ², 1, 1)
        det_Σ = σ²
        for x in [-1.0, 0.0, 2.0]
            @test StateSpaceRoutines.fast_mvnormal_pdf(x, det_Σ, inv_Σ) ≈ pdf(Normal(0.0, sqrt(σ²)), x)
        end
    end

    @testset "pdf is non-negative and peaks at zero" begin
        @test StateSpaceRoutines.fast_mvnormal_pdf(0.0) ≥ StateSpaceRoutines.fast_mvnormal_pdf(1.0)
        @test StateSpaceRoutines.fast_mvnormal_pdf([0.0, 0.0]) ≥ StateSpaceRoutines.fast_mvnormal_pdf([1.0, 1.0])
    end

end
