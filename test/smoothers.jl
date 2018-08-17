path = dirname(@__FILE__)

# Initialize arguments to function
h5 = h5open("$path/reference/kalman_filter_args.h5")
for arg in ["data", "TTT", "RRR", "CCC", "QQ", "ZZ", "DD", "EE", "z0", "P0"]
    eval(parse("$arg = read(h5, \"$arg\")"))
end
close(h5)

h5 = h5open("$path/reference/kalman_filter_out.h5")
for arg in ["pred", "vpred", "filt", "vfilt"]
    eval(parse("$arg = read(h5, \"$arg\")"))
end
close(h5)

# Run smoothers
states = Dict{Symbol, Matrix{Float64}}()
shocks = Dict{Symbol, Matrix{Float64}}()

states[:hamilton], shocks[:hamilton] = hamilton_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0)
states[:koopman], shocks[:koopman] = koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0, pred, vpred)
states[:carter_kohn], shocks[:carter_kohn] = carter_kohn_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0; draw_states = false)
states[:durbin_koopman], shocks[:durbin_koopman] = durbin_koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0; draw_states = false)

# Check that last-period smoothed states equal last-period filtered states
@testset "Ensure last smoothed state = the last filtered state" begin
    for smoother in [:hamilton, :koopman, :carter_kohn, :durbin_koopman]
        @test filt[:, end] ≈ states[smoother][:, end]
    end
end

# Compare to expected output
exp_states, exp_shocks = h5open("$path/reference/smoothers_out.h5", "r") do file
    read(file, "exp_states"), read(file, "exp_shocks")
end

@testset "Check expected output for all smoothers" begin
    for smoother in [:hamilton, :koopman, :carter_kohn, :durbin_koopman]
        @test exp_states ≈ states[smoother]
        @test exp_shocks ≈ shocks[smoother]
    end
end

# Make sure that simulation smoothers run with `draw_states` on
carter_kohn_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0; draw_states = true)
durbin_koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0; draw_states = true)


nothing
