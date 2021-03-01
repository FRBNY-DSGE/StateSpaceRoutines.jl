using StateSpaceRoutines, JLD2, HDF5, Random
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

# TODO: tests for regime-switching and make sure they match on states across smoothers too, including where the data is missing
# Furthermore, you should test that the implied observables are matching.
nreg = 7
TTTs = Vector{Matrix{Float64}}(undef, nreg)
RRRs = Vector{Matrix{Float64}}(undef, nreg)
CCCs = Vector{Vector{Float64}}(undef, nreg)
ZZs = Vector{Matrix{Float64}}(undef, nreg)
DDs = Vector{Vector{Float64}}(undef, nreg)
QQs = Vector{Matrix{Float64}}(undef, nreg)
EEs = Vector{Matrix{Float64}}(undef, nreg)
regime_inds0 = Vector{Vector{Int64}}(undef, nreg)
regime_inds = Vector{UnitRange{Int64}}(undef, nreg)

for i in 1:nreg
    TTTs[i] = h5read("reference/time_varying_system.h5", "T$(i)")
    RRRs[i] = h5read("reference/time_varying_system.h5", "R$(i)")
    CCCs[i] = h5read("reference/time_varying_system.h5", "C$(i)")
    ZZs[i] = h5read("reference/time_varying_system.h5", "Z$(i)")
    DDs[i] = h5read("reference/time_varying_system.h5", "D$(i)")
    QQs[i] = h5read("reference/time_varying_system.h5", "Q$(i)")
    EEs[i] = h5read("reference/time_varying_system.h5", "E$(i)")
    regime_inds0[i] = h5read("reference/time_varying_system.h5", "regime_inds$(i)")
    regime_inds[i] = regime_inds0[i][1]:regime_inds0[i][end]
end

y = h5read("reference/time_varying_system.h5", "data")

# Run smoothers
states = Dict{Symbol, Matrix{Float64}}()
shocks = Dict{Symbol, Matrix{Float64}}()

z0 = zeros(size(TTTs[1], 1))
P0 = StateSpaceRoutines.solve_discrete_lyapunov(TTTs[1], RRRs[1] * QQs[1] * RRRs[1]')
_, pred, vpred, ~ = kalman_filter(regime_inds, y, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0)

states[:hamilton], shocks[:hamilton] =
    hamilton_smoother(regime_inds, y, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0)
states[:koopman], shocks[:koopman] =
    koopman_smoother(regime_inds, y, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0, pred, vpred)
states[:carter_kohn], shocks[:carter_kohn] =
    carter_kohn_smoother(regime_inds, y, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0; draw_states = false)
states[:durbin_koopman], shocks[:durbin_koopman] =
    durbin_koopman_smoother(regime_inds, y, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0; draw_states = false)

Random.seed!(1793)
states[:carter_kohn_draw], shocks[:carter_kohn_draw] =
    carter_kohn_smoother(regime_inds, y, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0; draw_states = true)
states[:durbin_koopman_draw], shocks[:durbin_koopman_draw] =
    durbin_koopman_smoother(regime_inds, y, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0; draw_states = true)

obs = Dict{Symbol, Matrix{Float64}}()
for k in [:hamilton, :koopman, :carter_kohn, :durbin_koopman, :carter_kohn_draw, :durbin_koopman_draw]
    obs[k] = zeros(size(y, 1), size(states[:hamilton], 2))
    for (i, inds) in enumerate(regime_inds)
        obs[k][:, inds] = ZZs[i] * states[k][:, inds] .+ DDs[i]
    end
end

@testset "Regime-switching in T, R, etc., with smoothers" begin
    @test states[:hamilton] ≈ states[:carter_kohn]
    @test states[:koopman] ≈ states[:durbin_koopman]
    @test shocks[:hamilton] ≈ shocks[:carter_kohn]
    @test shocks[:koopman] ≈ shocks[:durbin_koopman]
    @test maximum(abs.(states[:hamilton] - states[:koopman])) < 5e-3
    @test maximum(abs.(shocks[:hamilton] - shocks[:koopman])) < 1.5e-3
    for i in 1:size(y, 1)
        not_nan = findall(.!isnan.(y[i, :]))
        if !isempty(not_nan)
            for k in [:hamilton, :koopman, :carter_kohn, :durbin_koopman]
                @test obs[k][i, not_nan] ≈ y[i, not_nan] atol=6e-6
            end
            if i in [1, 3, 7, 8, 13]
                @test obs[:carter_kohn_draw][i, not_nan] ≈ y[i, not_nan] atol=1e-2
            elseif i in [4, 5]
                @test obs[:carter_kohn_draw][i, not_nan] ≈ y[i, not_nan] atol=5e-4
            else
                @test obs[:carter_kohn_draw][i, not_nan] ≈ y[i, not_nan] atol=5e-5
            end
            @test obs[:durbin_koopman_draw][i, not_nan] ≈ y[i, not_nan] atol=5e-6
        end
    end
end


nothing
