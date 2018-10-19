path = dirname(@__FILE__)

# Initialize arguments to function
file = jldopen("$path/reference/kalman_filter_args.jld2", "r")
data            = read(file, "data")
TTT, RRR, CCC   = read(file, "TTT"), read(file, "RRR"), read(file, "CCC")
QQ, ZZ, DD, EE  = read(file, "QQ"), read(file, "ZZ"), read(file, "DD"), read(file, "EE")
z0, P0          = read(file, "z0"), read(file, "P0")
close(file)

file  = h5open("$path/reference/kalman_filter_out.h5", "r")
pred  = read(file, "pred")
vpred = read(file, "vpred")
filt  = read(file, "filt")
vfilt = read(file, "vfilt")
close(file)

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
