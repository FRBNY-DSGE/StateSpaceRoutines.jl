path = dirname(@__FILE__)

# Initialize arguments to function
file = h5open("$path/reference/kalman_filter_args.h5", "r")
y = read(file, "data")
T, R, C    = read(file, "TTT"), read(file, "RRR"), read(file, "CCC")
Q, Z, D, E = read(file, "QQ"), read(file, "ZZ"), read(file, "DD"), read(file, "EE")
s_0, P_0   = read(file, "z0"), read(file, "P0")
close(file)

# Kalman Filter (all arguments and no presample)
out = kalman_filter(y, T, R, C, Q, Z, D, E, s_0, P_0)
@testset "Basic Kalman Filter (all arguments, no presample)" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
        @test s_0                        ≈ out[6]
        @test P_0                        ≈ out[7]
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
        @test s_0                        ≈ out[6]
        @test P_0                        ≈ out[7]
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
file = jldopen("$path/reference/kalman_filter_args_zlb.jld")
y = read(file, "data")
Ts, Rs, Cs    = read(file, "Ts"), read(file, "Rs"), read(file, "Cs")
Qs, Zs, Ds, Es = read(file, "Qs"), read(file, "Zs"), read(file, "Ds"), read(file, "Es")
regime_inds = read(file, "regime_inds")
close(file)

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

# Fast Kalman filter i.e. set a tolerance and stop computing P'Z'V if the max difference between PZV and the last one is less than tolerance.

file = h5open("$path/reference/kalman_filter_args.h5", "r")
y = read(file, "data")
T, R, C    = read(file, "TTT"), read(file, "RRR"), read(file, "CCC")
Q, Z, D, E = read(file, "QQ"), read(file, "ZZ"), read(file, "DD"), read(file, "EE")
s_0, P_0   = read(file, "z0"), read(file, "P0")
close(file)

# Kalman Filter (all arguments and no presample)
out = kalman_filter(y, T, R, C, Q, Z, D, E, s_0, P_0, tol = 1e-5)
@testset "Basic Kalman Filter (all arguments, no presample)" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
        @test s_0                        ≈ out[6]
        @test P_0                        ≈ out[7]
    end
end


# Kalman Filter (no initial conditions and no presample)
out = kalman_filter(y, T, R, C, Q, Z, D, E, tol = 1e-5)
@testset "Kalman Filter (no initial conditions)" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
        @test s_0                        ≈ out[6]
        @test P_0                        ≈ out[7]
    end
end

# Kalman filter with presample
out = kalman_filter(y, T, R, C, Q, Z, D, E, Nt0=4, tol = 1e-5)
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
file = jldopen("$path/reference/kalman_filter_args_zlb.jld")
y = read(file, "data")
Ts, Rs, Cs    = read(file, "Ts"), read(file, "Rs"), read(file, "Cs")
Qs, Zs, Ds, Es = read(file, "Qs"), read(file, "Zs"), read(file, "Ds"), read(file, "Es")
regime_inds = read(file, "regime_inds")
close(file)

out = kalman_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, tol = 1e-5)
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
