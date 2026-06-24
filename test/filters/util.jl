using StateSpaceRoutines, Test, LinearAlgebra

Random.seed!(47)

T = 10
Nt0 = 3
ns = 2

loglh  = collect(Float64, 1:T)
s_pred = reshape(Float64.(1:(ns*T)), ns, T)
P_pred = reshape(Float64.(1:(ns*ns*T)), ns, ns, T)
s_filt = reshape(Float64.(1:(ns*T)), ns, T)
P_filt = reshape(Float64.(1:(ns*ns*T)), ns, ns, T)

@testset "remove_presample! (full)" begin

    loglh2, sp2, Pp2, sf2, Pf2 = StateSpaceRoutines.remove_presample!(
        Nt0, copy(loglh), copy(s_pred), copy(P_pred), copy(s_filt), copy(P_filt))

    @testset "Removes first Nt0 periods" begin
        @test length(loglh2) == T - Nt0
        @test size(sp2, 2)   == T - Nt0
        @test size(Pp2, 3)   == T - Nt0
        @test size(sf2, 2)   == T - Nt0
        @test size(Pf2, 3)   == T - Nt0
    end

    @testset "Remaining values are from period Nt0+1 onward" begin
        @test loglh2[1] ≈ loglh[Nt0 + 1]
        @test sp2[:, 1] ≈ s_pred[:, Nt0 + 1]
    end

    @testset "Nt0=0 returns unchanged arrays" begin
        loglh3, sp3, Pp3, sf3, Pf3 = StateSpaceRoutines.remove_presample!(
            0, copy(loglh), copy(s_pred), copy(P_pred), copy(s_filt), copy(P_filt))
        @test length(loglh3) == T
        @test size(sp3, 2)   == T
    end

    @testset "outputs kwarg: :loglh only leaves pred/filt unchanged" begin
        loglh_in = copy(loglh)
        sp_in = copy(s_pred); Pp_in = copy(P_pred)
        sf_in = copy(s_filt); Pf_in = copy(P_filt)
        loglh4, sp4, Pp4, sf4, Pf4 = StateSpaceRoutines.remove_presample!(
            Nt0, loglh_in, sp_in, Pp_in, sf_in, Pf_in; outputs = [:loglh])
        @test length(loglh4) == T - Nt0
        @test size(sp4, 2)   == T   # pred not trimmed
        @test size(sf4, 2)   == T   # filt not trimmed
    end

end

@testset "remove_presample! (loglh only)" begin

    loglh2 = StateSpaceRoutines.remove_presample!(Nt0, copy(loglh))
    @test length(loglh2) == T - Nt0
    @test loglh2[1] ≈ loglh[Nt0 + 1]

    loglh3 = StateSpaceRoutines.remove_presample!(0, copy(loglh))
    @test length(loglh3) == T
    @test loglh3 == loglh

end

@testset "solve_discrete_lyapunov" begin

    @testset "Identity case: AXA' - X + B = 0 satisfied" begin
        A = 0.5 * I(3) |> Matrix
        B = Matrix(I(3) * 1.0)
        X = StateSpaceRoutines.solve_discrete_lyapunov(A, B)
        residual = A * X * A' - X + B
        @test maximum(abs, residual) < 1e-10
    end

    @testset "Scalar-like 1x1 case" begin
        A = fill(0.5, 1, 1)
        B = fill(1.0, 1, 1)
        X = StateSpaceRoutines.solve_discrete_lyapunov(A, B)
        # Analytic: 0.25*X - X + 1 = 0 => X = 4/3
        @test X[1,1] ≈ 4/3 atol=1e-10
    end

    @testset "Solution is symmetric when B is symmetric" begin
        A = [0.8 0.1; 0.0 0.5]
        B = [1.0 0.2; 0.2 1.0]
        X = StateSpaceRoutines.solve_discrete_lyapunov(A, B)
        @test maximum(abs, X - X') < 1e-10
    end

    @testset "Residual AXA' - X + B ≈ 0 for general case" begin
        A = [0.6 0.2; 0.1 0.4]
        B = [2.0 0.5; 0.5 3.0]
        X = StateSpaceRoutines.solve_discrete_lyapunov(A, B)
        residual = A * X * A' - X + B
        @test maximum(abs, residual) < 1e-10
    end

end
