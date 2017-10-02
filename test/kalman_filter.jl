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
    @test read(h5, "log_likelihood") ≈ out[1]
    @test read(h5, "pred")           ≈ out[4]
    @test read(h5, "vpred")          ≈ out[5]
    @test read(h5, "filt")           ≈ out[6]
    @test read(h5, "vfilt")          ≈ out[7]
    @test read(h5, "yprederror")     ≈ out[8]
    @test read(h5, "ystdprederror")  ≈ out[9]
    @test read(h5, "rmse")           ≈ out[10]
    @test read(h5, "rmsd")           ≈ out[11]
    @test z0                         ≈ out[12]
    @test P0                         ≈ out[13]
    @test read(h5, "marginal_loglh") ≈ out[14]
end

# Method with initial conditions omitted
out = kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE)

# Pend, vpred, and vfilt matrix entries are especially large, averaging 1e5, so
# we allow greater ϵ
h5open("$path/reference/kalman_filter_out.h5", "r") do h5
    @test read(h5, "log_likelihood") ≈ out[1]
    @test read(h5, "pred")           ≈ out[4]
    @test read(h5, "vpred")          ≈ out[5]
    @test read(h5, "filt")           ≈ out[6]
    @test read(h5, "vfilt")          ≈ out[7]
    @test read(h5, "yprederror")     ≈ out[8]
    @test read(h5, "ystdprederror")  ≈ out[9]
    @test read(h5, "rmse")           ≈ out[10]
    @test read(h5, "rmsd")           ≈ out[11]
    @test z0                         ≈ out[12]
    @test P0                         ≈ out[13]
    @test read(h5, "marginal_loglh") ≈ out[14]
end


nothing
