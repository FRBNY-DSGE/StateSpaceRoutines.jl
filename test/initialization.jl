using StateSpaceRoutines, Test, Distributions, Random, LinearAlgebra

Random.seed!(47)

# Simple 1D AR(1): s_t = 0.9*s_{t-1} + ε_t, ε ~ N(0,1)
s0_1d  = [0.0]
F_ϵ_1d = Normal(0.0, 1.0)
Φ_1d(s, ϵ) = [0.9 * s[1] + ϵ]
n_parts = 200

draws_1d = initialize_state_draws(s0_1d, F_ϵ_1d, Φ_1d, n_parts; burn = 500, thin = 2)

# 2D system: s_t = A*s_{t-1} + ε_t
A      = [0.8 0.1; 0.0 0.5]
F_ϵ_2d = MvNormal(zeros(2), I)
Φ_2d(s, ϵ) = A * s + ϵ
s0_2d  = [0.0, 0.0]

draws_2d = initialize_state_draws(s0_2d, F_ϵ_2d, Φ_2d, n_parts; burn = 500, thin = 2)

@testset "initialize_state_draws" begin

    @testset "Output dimensions" begin
        @test size(draws_1d) == (1, n_parts)
        @test eltype(draws_1d) == Float64
        @test size(draws_2d) == (2, n_parts)
    end

    @testset "Chain evolved from initial state" begin
        @test !all(draws_1d .== 0.0)
        @test length(unique(draws_1d[1, :])) > 1
    end

    @testset "Stationary distribution: mean and variance close to analytic" begin
        # AR(1) stationary: mean=0, var=1/(1-0.9^2) ≈ 5.26
        @test abs(mean(draws_1d[1, :])) < 1.0
        @test abs(var(draws_1d[1, :]) - 1.0 / (1.0 - 0.9^2)) < 2.0
    end

    @testset "Multivariate: both state dimensions have variation" begin
        @test std(draws_2d[1, :]) > 0.0
        @test std(draws_2d[2, :]) > 0.0
    end

    @testset "Thinning: output shape respects n_parts and thin" begin
        draws_thin = initialize_state_draws(s0_1d, F_ϵ_1d, Φ_1d, 50; burn = 100, thin = 10)
        @test size(draws_thin) == (1, 50)
    end

end
