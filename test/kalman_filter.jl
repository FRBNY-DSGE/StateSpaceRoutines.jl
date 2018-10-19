path = dirname(@__FILE__)

# Initialize arguments to function
file = jldopen("$path/reference/kalman_filter_args.jld2", "r")
data = read(file, "data")
TTT, RRR, CCC    = read(file, "TTT"), read(file, "RRR"), read(file, "CCC")
QQ, ZZ, DD, EE = read(file, "QQ"), read(file, "ZZ"), read(file, "DD"), read(file, "EE")
z0, P0   = read(file, "z0"), read(file, "P0")
close(file)

# Method with all arguments provided
out = kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0)

@testset "Test Kalman filter output with all arguments" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
        @test z0                         ≈ out[6]
        @test P0                         ≈ out[7]
    end
end

# Method with initial conditions omitted
out = kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE)

@testset "Test Kalman filter output with initial conditions omitted" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
        @test z0                         ≈ out[6]
        @test P0                         ≈ out[7]
    end
end

nothing
