path = dirname(@__FILE__)

# Initialize arguments to function
@load "$path/reference/kalman_filter_args.jld2" y T R C Q Z D E z0 P0

# Kalman Filter (all arguments and no presample)
out = kalman_filter(y, T, R, C, Q, Z, D, E, z0, P0)
@testset "Basic Kalman Filter (all arguments, no presample)" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
    end
end

# Kalman Filter (no initial conditions and no presample)
out = kalman_filter(y, T, R, C, Q, Z, D, E)
@testset "Kalman Filter (no initial conditions)" begin
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

# Kalman filter with presample
out = kalman_filter(y, T, R, C, Q, Z, D, E, Nt0=4)
@testset "Kalman Filter (presample)" begin
    h5open("$path/reference/kalman_filter_out_presample.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
    end
end

# Initialize arguments to for multi-regime Kalman Filter (ZLB)
@load "$path/reference/kalman_filter_args_zlb.jld2" y Ts Rs Cs Qs Zs Ds Es regime_inds

out = kalman_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es)
@testset "Kalman Filter (Multi-regime/ZLB)" begin
    h5open("$path/reference/kalman_filter_out_zlb.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
    end
end


nothing
