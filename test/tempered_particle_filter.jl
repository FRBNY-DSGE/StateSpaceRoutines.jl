@everywhere using DSGE, DSGEModels, StateSpaceRoutines, JLD
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
        df = readtable("reference/us.txt", header = false, separator = ' ') # Great Moderation sample
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

test_file_inputs = load("reference/tpf_aux_inputs.jld")
test_file_outputs = load("reference/tpf_aux_outputs.jld")
test_file_outputs_mutation = load("reference/tpf_aux_mutation_outputs.jld")
coeff_terms = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
s_t_nontemp = test_file_inputs["s_t_nontemp"]
φ_old = test_file_inputs["phi_old"]
inc_weights = test_file_inputs["inc_weights"]
norm_weights = test_file_inputs["norm_weights"]
HH = cov(F_u)
QQ = cov(F_ϵ)
s_t1_temp = test_file_inputs["s_t1_temp"]
ϵ_t = test_file_inputs["eps_t"]
accept_rate = test_file_inputs["accept_rate"]
c = test_file_inputs["c"]

## Correction and Associated Auxiliary Function Tests
weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, data[:, 47], s_t_nontemp, det(HH), inv(HH); initialize=false, parallel=false)
φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)
correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]))

@testset "Corection and Auxiliary Tests" begin
    @test coeff_terms[1] ≈ test_file_outputs["coeff_terms"][1]
    @test log_e_1_terms[1] ≈ test_file_outputs["log_e_1_terms"][1]
    @test log_e_2_terms[1] ≈ test_file_outputs["log_e_2_terms"][1]
    @test φ_new ≈ test_file_outputs["phi_new"]
    @test inc_weights[1] ≈ test_file_outputs["inc_weights"][1]
end

## Selection Tests
srand(47)
selection!(norm_weights, s_t1_temp, s_t_nontemp,ϵ_t, resampling_method = tuning[:resampling_method])
@testset "Selection Tests" begin
    @test s_t1_temp[1] ≈ test_file_outputs["s_t1_temp"][1]
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp"][1]
    @test ϵ_t[1] ≈ test_file_outputs["eps_t"][1]
end

c = update_c(c, accept_rate, tuning[:target_accept_rate])

## Mutation Tests
srand(47)
mutation!(Φ, Ψ, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c, tuning[:n_mh_steps])

@testset "Mutation Tests" begin
    @test s_t_nontemp[1] ≈ test_file_outputs_mutation["s_t_nontemp"][1]
    @test ϵ_t[1] ≈ test_file_outputs_mutation["eps_t"][1]
end

## Whole TPF Tests
srand(47)
out_no_parallel = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false)
srand(47)
out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true)
@testset "TPF tests" begin
    @test out_no_parallel[1] ≈ -302.99967306704133
    @test out_parallel_one_worker[1] ≈ -306.8211172094595
end


#### OLD TEST FILE
#=
using Base.Test

path = dirname(@__FILE__)

args_file = h5open("$path/reference/tempered_particle_filter_args.h5")
for arg in ["data", "TTT", "RRR", "QQ", "ZZ", "DD", "HH", "data", "s0", "s_init"]
    eval(parse("$arg = read(args_file, \"$arg\")"))
end
close(args_file)

sqrtS2 = RRR*chol(QQ)'

Φ(s_t::Vector{Float64}, ϵ_t::Vector{Float64}) = TTT*s_t + sqrtS2*ϵ_t
Ψ(s_t::Vector{Float64}, u_t::Vector{Float64}) = ZZ*s_t + DD + u_t

F_ϵ = Distributions.MvNormal(zeros(size(QQ, 1)), eye(size(QQ, 1)))
F_u = Distributions.MvNormal(zeros(size(HH, 1)), HH)

fixed_sched = [0.2, 0.5, 1.0]

test_data = reshape(data[:, 1], 3, 1)
loglik, lik, _ = tempered_particle_filter(test_data, Φ, Ψ, F_ϵ, F_u, s_init; r_star = 2., c = 0.3,
                         accept_rate = 0.4, target = 0.4, xtol = 0., resampling_method = :multinomial,
                         N_MH = 1, n_particles = 500, n_presample_periods = 0, verbose = :none,
                         adaptive = false, fixed_sched = fixed_sched, allout = true, parallel = false,
                         testing = true)


file = h5open("$path/reference/tempered_particle_filter_out.h5", "r")
test_loglik = read(file, "log_lik")
test_lik    = read(file, "incr_lik")
close(file)

@test test_loglik ≈ (loglik)
@test test_lik ≈ lik

tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; n_particles = 500, verbose = :none);

nothing
=#
