using StateSpaceRoutines, Test, LinearAlgebra

Random.seed!(47)

# Single-regime system: Nz=3 states, Ne=2 shocks, Ny=2 observables
Nz, Ne, Ny, T = 3, 2, 2, 5

TTT = [0.8 0.1 0.0; 0.0 0.5 0.2; 0.0 0.0 0.3]
RRR = [1.0 0.0; 0.0 1.0; 0.5 0.5]
CCC = [0.1, 0.2, 0.3]
QQ  = Matrix(0.5 * I(Ne))
ZZ  = [1.0 0.0 0.0; 0.0 1.0 0.0]
z0  = ones(Nz)
P0  = Matrix(2.0 * I(Nz))

regime_indices = [1:T]
TTTs = [TTT]; RRRs = [RRR]; CCCs = [CCC]; QQs = [QQ]; ZZs = [ZZ]

newTTTs, newRRRs, newCCCs, newZZs, z0_aug, P0_aug =
    StateSpaceRoutines.augment_states_with_shocks(
        regime_indices, TTTs, RRRs, CCCs, QQs, ZZs, copy(z0), copy(P0))

@testset "augment_states_with_shocks" begin

    @testset "Output dimensions (single regime)" begin
        @test size(newTTTs[1]) == (Nz + Ne, Nz + Ne)
        @test size(newRRRs[1]) == (Nz + Ne, Ne)
        @test length(newCCCs[1]) == Nz + Ne
        @test size(newZZs[1])  == (Ny, Nz + Ne)
        @test length(z0_aug)   == Nz + Ne
        @test size(P0_aug)     == (Nz + Ne, Nz + Ne)
    end

    @testset "newTTT structure: original TTT in top-left, zeros elsewhere" begin
        @test newTTTs[1][1:Nz, 1:Nz] ≈ TTT
        @test newTTTs[1][1:Nz, Nz+1:end]    ≈ zeros(Nz, Ne)
        @test newTTTs[1][Nz+1:end, 1:Nz]    ≈ zeros(Ne, Nz)
        @test newTTTs[1][Nz+1:end, Nz+1:end] ≈ zeros(Ne, Ne)
    end

    @testset "newRRR structure: original RRR on top, identity on bottom" begin
        @test newRRRs[1][1:Nz, :] ≈ RRR
        @test newRRRs[1][Nz+1:end, :] ≈ Matrix(I(Ne) * 1.0)
    end

    @testset "newCCC structure: original CCC on top, zeros on bottom" begin
        @test newCCCs[1][1:Nz] ≈ CCC
        @test newCCCs[1][Nz+1:end] ≈ zeros(Ne)
    end

    @testset "newZZ structure: original ZZ on left, zeros on right" begin
        @test newZZs[1][:, 1:Nz] ≈ ZZ
        @test newZZs[1][:, Nz+1:end] ≈ zeros(Ny, Ne)
    end

    @testset "z0 augmented with zeros" begin
        @test z0_aug[1:Nz] ≈ z0
        @test z0_aug[Nz+1:end] ≈ zeros(Ne)
    end

    @testset "P0 augmented with QQ block in bottom-right, zeros off-diagonal" begin
        @test P0_aug[1:Nz, 1:Nz]       ≈ P0
        @test P0_aug[Nz+1:end, Nz+1:end] ≈ QQ
        @test P0_aug[1:Nz, Nz+1:end]    ≈ zeros(Nz, Ne)
        @test P0_aug[Nz+1:end, 1:Nz]    ≈ zeros(Ne, Nz)
    end

    @testset "Multiple regimes: each regime augmented independently" begin
        TTT2 = 0.5 * TTT
        QQ2  = Matrix(I(Ne) * 1.0)
        regime_indices2 = [1:3, 4:T]
        TTTs2 = [TTT, TTT2]; RRRs2 = [RRR, RRR]
        CCCs2 = [CCC, CCC];  QQs2  = [QQ, QQ2]; ZZs2 = [ZZ, ZZ]

        newTTTs2, newRRRs2, newCCCs2, newZZs2, _, _ =
            StateSpaceRoutines.augment_states_with_shocks(
                regime_indices2, TTTs2, RRRs2, CCCs2, QQs2, ZZs2, copy(z0), copy(P0))

        @test length(newTTTs2) == 2
        @test newTTTs2[1][1:Nz, 1:Nz] ≈ TTT
        @test newTTTs2[2][1:Nz, 1:Nz] ≈ TTT2
    end

end
