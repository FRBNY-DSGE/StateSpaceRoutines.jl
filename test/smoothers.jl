path = dirname(@__FILE__)

# Initialize arguments to function
@load "$path/reference/kalman_filter_args.jld2" y T R C Q Z D E z0 P0

file  = h5open("$path/reference/kalman_filter_out.h5", "r")
pred  = read(file, "pred")
vpred = read(file, "vpred")
filt  = read(file, "filt")
vfilt = read(file, "vfilt")
close(file)

# Run smoothers
states = Dict{Symbol, Matrix{Float64}}()
shocks = Dict{Symbol, Matrix{Float64}}()

states[:hamilton], shocks[:hamilton] = hamilton_smoother(y, T, R, C, Q, Z, D, E, z0, P0)
states[:koopman], shocks[:koopman] = koopman_smoother(y, T, R, C, Q, Z, D, E, z0, P0, pred, vpred)
states[:carter_kohn], shocks[:carter_kohn] = carter_kohn_smoother(y, T, R, C, Q, Z, D, E, z0, P0; draw_states = false)
states[:durbin_koopman], shocks[:durbin_koopman] = durbin_koopman_smoother(y, T, R, C, Q, Z, D, E, z0, P0; draw_states = false)

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
carter_kohn_smoother(y, T, R, C, Q, Z, D, E, z0, P0; draw_states = true)
durbin_koopman_smoother(y, T, R, C, Q, Z, D, E, z0, P0; draw_states = true)

nothing
