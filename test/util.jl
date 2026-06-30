using StateSpaceRoutines, Test

Random.seed!(47)

@testset "scalar_reduce" begin

    @testset "Two iterations, two outputs" begin
        a1 = [1.0, 2.0]; b1 = [10.0]
        a2 = [3.0, 4.0]; b2 = [20.0]
        r = StateSpaceRoutines.scalar_reduce([copy(a1), copy(b1)], [copy(a2), copy(b2)])
        @test r[1] == vcat(a1, a2)
        @test r[2] == vcat(b1, b2)
    end

    @testset "Three iterations" begin
        iters = [[Float64[i], Float64[i*10]] for i in 1:3]
        r = StateSpaceRoutines.scalar_reduce(iters...)
        @test r[1] == [1.0, 2.0, 3.0]
        @test r[2] == [10.0, 20.0, 30.0]
    end

    @testset "Single iteration returns unchanged" begin
        v = [[1.0, 2.0], [3.0]]
        r = StateSpaceRoutines.scalar_reduce(v)
        @test r[1] == [1.0, 2.0]
        @test r[2] == [3.0]
    end

end

@testset "vector_reduce" begin

    @testset "Two iterations, two matrix outputs" begin
        # Elements must be Matrix{Float64} so hcat result fits back into return_arg
        A1 = reshape([1.0; 2.0], 2, 1); A2 = reshape([3.0; 4.0], 2, 1)
        B1 = reshape([10.0; 20.0], 2, 1); B2 = reshape([30.0; 40.0], 2, 1)
        r = StateSpaceRoutines.vector_reduce([copy(A1), copy(B1)], [copy(A2), copy(B2)])
        @test r[1] == hcat(A1, A2)
        @test r[2] == hcat(B1, B2)
    end

    @testset "Three iterations build correct column count" begin
        iters = [Matrix{Float64}[reshape([Float64(i); Float64(i+1)], 2, 1)] for i in 1:3]
        r = StateSpaceRoutines.vector_reduce(iters...)
        @test size(r[1]) == (2, 3)
        @test r[1][:, 1] == [1.0, 2.0]
        @test r[1][:, 3] == [3.0, 4.0]
    end

end

@testset "vec_scal_reduce" begin

    @testset "Two iterations: first two outputs hcatted, last appended" begin
        A1 = reshape([1.0; 2.0], 2, 1); A2 = reshape([3.0; 4.0], 2, 1)
        B1 = reshape([10.0; 20.0], 2, 1); B2 = reshape([30.0; 40.0], 2, 1)
        c1 = [5.0]; c2 = [6.0]
        r = StateSpaceRoutines.vec_scal_reduce(
            [copy(A1), copy(B1), copy(c1)],
            [copy(A2), copy(B2), copy(c2)])
        @test r[1] == hcat(A1, A2)
        @test r[2] == hcat(B1, B2)
        @test r[3] == [5.0, 6.0]
    end

end

@testset "scalar_reshape" begin

    @testset "Scalar inputs are wrapped in vectors" begin
        r = StateSpaceRoutines.scalar_reshape(1.0, 2.0, 3.0)
        @test r == [[1.0], [2.0], [3.0]]
        @test eltype(r[1]) == Float64
    end

    @testset "Vector inputs are passed through" begin
        v1 = [1.0, 2.0]; v2 = [3.0, 4.0, 5.0]
        r = StateSpaceRoutines.scalar_reshape(v1, v2)
        @test r[1] == v1
        @test r[2] == v2
    end

    @testset "Returns Vector{Vector{Float64}}" begin
        r = StateSpaceRoutines.scalar_reshape(1.0, [2.0, 3.0])
        @test r isa Vector{Vector{Float64}}
        @test length(r) == 2
    end

end

@testset "vector_reshape" begin

    @testset "Vector inputs become column matrices" begin
        v = [1.0, 2.0, 3.0]
        r = StateSpaceRoutines.vector_reshape(v)
        @test r[1] == reshape(v, 3, 1)
        @test size(r[1]) == (3, 1)
    end

    @testset "Scalar inputs become 1x1 matrices" begin
        r = StateSpaceRoutines.vector_reshape(5.0)
        @test r[1] == reshape([5.0], 1, 1)
    end

    @testset "Returns Vector{Matrix{Float64}}" begin
        r = StateSpaceRoutines.vector_reshape([1.0, 2.0], [3.0])
        @test r isa Vector{Matrix{Float64}}
        @test length(r) == 2
    end

end
