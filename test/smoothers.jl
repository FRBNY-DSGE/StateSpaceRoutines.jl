using StateSpaceRoutines
using HDF5

path = dirname(@__FILE__)

# Initialize arguments to function
h5 = h5open("$path/reference/kalman_filter_args.h5")
for arg in ["data", "TTT", "RRR", "CCC", "QQ", "ZZ", "DD", "MM", "EE", "z0", "P0"]
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

states[:hamilton], shocks[:hamilton] = hamilton_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, MM, EE, z0, P0)
states[:koopman], shocks[:koopman] = koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, z0, P0, pred, vpred)
states[:carter_kohn], shocks[:carter_kohn] = carter_kohn_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, MM, EE, z0, P0; draw_states = false)
states[:durbin_koopman], shocks[:durbin_koopman] = durbin_koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, MM, EE, z0, P0; draw_states = false)

# Check that last-period smoothed states equal last-period filtered states
for smoother in [:hamilton, :koopman, :carter_kohn, :durbin_koopman]
    @test_approx_eq filt[:, end] states[smoother][:, end]
end

# Compare to expected output
for smoother in [:hamilton, :koopman, :carter_kohn, :durbin_koopman]
    @test_approx_eq_eps states[:koopman] states[smoother] 1e-8
    @test_approx_eq_eps shocks[:koopman] shocks[smoother] 1e-8
end