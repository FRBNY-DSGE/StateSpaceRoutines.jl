@everywhere using JLD, DSGE, DSGEModels, StateSpaceRoutines
@everywhere using QuantEcon: solve_discrete_lyapunov
using Base.Test, BenchmarkTools, DataFrames, Plots

@everywhere model = :AnSchorfheide
do_setup = false
overwrite_jld = true

if do_setup
    # Set up model and data
    if model ==:AnSchorfheide
        @everywhere m = AnSchorfheide()
        para = [2.09, 0.98, 2.25, 0.65, 0.34, 3.16, 0.51, 0.81, 0.98, 0.93, 0.19, 0.65, 0.24,
                0.115985, 0.294166, 0.447587] # θ^m on page 24 of the paper
        df = readtable("us.txt", header = false, separator = ' ') # Great Moderation sample
        data = convert(Matrix{Float64}, df)'
    elseif model ==:SmetsWouters
        @everywhere m = SmetsWoutersOrig()
        para = vec(readdlm("/data/dsge_data_dir/dsgejl/smc/schorfheide/models/sw/theta_m.txt",
                           comments = true, comment_char = '%'))
        data = readdlm("/data/dsge_data_dir/dsgejl/smc/schorfheide/models/sw/us.txt")'
    end
    update!(m, para)
    @everywhere system = compute_system(m)
    if model == :SmetsWouters
        @everywhere system.measurement.EE = diagm([0.1731, 0.1394, 0.4515, 0.1128, 0.5838, 0.1230, 0.1653].^2)
    end

    # Generate initial state draws
    nstates = n_states_augmented(m)
    s0 = zeros(nstates)
    P0 = solve_discrete_lyapunov(system[:TTT], system[:RRR]*system[:QQ]*system[:RRR]')
    U, E, V = svd(P0)
    srand(10)
    s_init = s0 .+ U*diagm(sqrt.(E))*randn(nstates, 1000)
    @everywhere F_ϵ = Distributions.MvNormal(system[:QQ])
    @everywhere F_u = Distributions.MvNormal(system[:EE])

    # Write to JLD
    if overwrite_jld
        jldopen("reference/input_args_$model.jld", "w") do file
            write(file, "data",      data)
            write(file, "system",    system)
            write(file, "F_epsilon", F_ϵ)
            write(file, "F_u",       F_u)
            write(file, "s_init",    s_init)
        end
        println("Wrote input_args_$model.jld")
    end
else
    # Read in from JLD
    println("Reading from input_args_$model.jld...")
    @everywhere data, system, F_ϵ, F_u, s_init = jldopen("reference/input_args_$model.jld", "r") do file
        read(file, "data"),
        read(file, "system"),
        read(file, "F_epsilon"),
        read(file, "F_u"),
        read(file, "s_init")
    end
end

# Tune algorithm
@everywhere tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
                          :resampling_method => :systematic, :n_mh_steps => 1,
                          :n_particles => 1000, :n_presample_periods => 0,
                          :allout => true)
#@everywhere tuning[:n_particles] = 1000 # THIS CAN BE CHANGED
@assert mod(tuning[:n_particles], 1000) == 0

# Adjust s_init for number of particles
@everywhere reptimes = convert(Int, tuning[:n_particles] / 1000)
@everywhere s_init = repeat(s_init, outer = (1, reptimes))

# Define Φ and Ψ (can't be saved to JLD)
@everywhere const TTT = system[:TTT]
@everywhere const RRR = system[:RRR]
@everywhere const CCC = system[:CCC]
@everywhere const ZZ = system[:ZZ]
@everywhere const DD = system[:DD]

@everywhere Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
@everywhere Ψ(s_t::AbstractVector{Float64}) = ZZ*s_t + DD

srand(47)
out_no_parallel = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false)
