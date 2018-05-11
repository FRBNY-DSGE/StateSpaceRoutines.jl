path = dirname(@__FILE__)

# Initialize arguments to function
h5 = h5open("$path/reference/kalman_filter_args.h5", "r")
for arg in ["data", "TTT", "RRR", "CCC", "QQ", "ZZ", "DD", "EE", "z0", "P0"]
    eval(parse("$arg = read(h5, \"$arg\")"))
end
close(h5)

# Method with all arguments provided
out = kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0)

h5open("$path/reference/kalman_filter_out.h5", "r") do h5
    @test read(h5, "log_likelihood") ≈ sum(out.loglh)
    @test read(h5, "marginal_loglh") ≈ out.loglh
    @test read(h5, "pred")           ≈ out.s_pred
    @test read(h5, "vpred")          ≈ out.P_pred
    @test read(h5, "filt")           ≈ out.s_filt
    @test read(h5, "vfilt")          ≈ out.P_filt
    @test z0                         ≈ out.s_0
    @test P0                         ≈ out.P_0
end

# Method with initial conditions omitted
out = kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE)

h5open("$path/reference/kalman_filter_out.h5", "r") do h5
    @test read(h5, "log_likelihood") ≈ sum(out.loglh)
    @test read(h5, "marginal_loglh") ≈ out.loglh
    @test read(h5, "pred")           ≈ out.s_pred
    @test read(h5, "vpred")          ≈ out.P_pred
    @test read(h5, "filt")           ≈ out.s_filt
    @test read(h5, "vfilt")          ≈ out.P_filt
    @test z0                         ≈ out.s_0
    @test P0                         ≈ out.P_0
end


nothing
