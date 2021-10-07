using JLD, JLD2, Test, Distributions, Random, StateSpaceRoutines, BenchmarkTools, ParallelDataTransfer, FLoops, DistributedArrays, OffsetArrays

@show "Parallel 12"

# Read in from JLD
tpf_main_input = load("reference/tpf_main_inputs.jld2")
data = tpf_main_input["data"]
TTT = tpf_main_input["TTT"]
RRR = tpf_main_input["RRR"]
CCC = tpf_main_input["CCC"]
ZZ = tpf_main_input["ZZ"]
DD = tpf_main_input["DD"]
F_ϵ = tpf_main_input["F_epsilon"]
F_u = tpf_main_input["F_u"]
s_init = tpf_main_input["s_init"]

# Tune algorithm
tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
              :resampling_method => :systematic, :n_mh_steps => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true, :parallel => true)

# Define Φ and Ψ (can't be saved to JLD)
Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
Ψ(s_t::AbstractVector{Float64}) = ZZ*s_t + DD

# Load in test inputs and outputs
test_file_inputs = load("reference/tpf_aux_inputs.jld2")
test_file_outputs = load("reference/tpf_aux_outputs.jld2")

φ_old = test_file_inputs["phi_old"]
norm_weights = test_file_inputs["norm_weights"]
coeff_terms = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
inc_weights = test_file_inputs["inc_weights"]
HH = cov(F_u)
s_t_nontemp = test_file_inputs["s_t_nontemp"]

ENV["frbnyjuliamemory"] = "5G"
myprocs = addprocs_frbny(12)
@everywhere using JLD, JLD2, Test, Distributions, Random, StateSpaceRoutines, BenchmarkTools, ParallelDataTransfer


@everywhere tpf_main_input = load("reference/tpf_main_inputs.jld2")
@everywhere TTT = tpf_main_input["TTT"]
@everywhere RRR = tpf_main_input["RRR"]
@everywhere CCC = tpf_main_input["CCC"]
@everywhere ZZ = tpf_main_input["ZZ"]
@everywhere DD = tpf_main_input["DD"]
@everywhere Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
@everywhere Ψ(s_t::AbstractVector{Float64}) = ZZ*s_t + DD

#= StateSpaceRoutines.sendto(workers(), Φ = Φ)
StateSpaceRoutines.sendto(workers(), Ψ = Ψ)
StateSpaceRoutines.sendto(workers(), coeff_terms = coeff_terms)
StateSpaceRoutines.sendto(workers(), log_e_1_terms = log_e_1_terms)
StateSpaceRoutines.sendto(workers(), log_e_2_terms = log_e_2_terms)
StateSpaceRoutines.sendto(workers(), data = data)
StateSpaceRoutines.sendto(workers(), s_t_nontemp = s_t_nontemp)
StateSpaceRoutines.sendto(workers(), HH = HH)=#

s_t_nontemp = distribute(s_t_nontemp)

@btime weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, data[:, 47], s_t_nontemp, det(HH), inv(HH);
               initialize = false, parallel = true)
weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, data[:, 47], s_t_nontemp, det(HH), inv(HH);
               initialize = false, parallel = true)
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
s_t1_temp = test_file_inputs["s_t1_temp"]
ϵ_t = test_file_inputs["eps_t"]
#=
ENV["frbnyjuliamemory"] = "5G"
myprocs = addprocs_frbny(48)
@everywhere using JLD2, Test, Distributions, Random, StateSpaceRoutines, BenchmarkTools, ParallelDataTransfer
#@everywhere include("../src/filters/tempered_particle_filter/correction.jl")
#@everywhere include("../src/util.jl")
=#
Random.seed!(47)
s_t_nontemp_std = convert(Array, s_t_nontemp)
selection!(norm_weights, s_t1_temp, s_t_nontemp_std,ϵ_t, resampling_method = tuning[:resampling_method])
s_t_nontemp = distribute(s_t_nontemp_std)
@testset "Selection Tests" begin
    @test s_t1_temp[1] ≈ test_file_outputs["s_t1_temp"][1]
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp"][1]
    @test ϵ_t[1] ≈ test_file_outputs["eps_t"][1]
end

s_t1_temp = distribute(s_t1_temp)
#ϵ_t = distribute(ϵ_t)
## Mutation Tests
QQ = cov(F_ϵ)
accept_rate = test_file_inputs["accept_rate"]
c = test_file_inputs["c"]

c = update_c(c, accept_rate, tuning[:target_accept_rate])
Random.seed!(47)

StateSpaceRoutines.mutation!(Φ, Ψ, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c, tuning[:n_mh_steps],
                             parallel = true)
#=
@testset "Mutation Tests" begin
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp_mutation"][1]
    @test ϵ_t[1] ≈ test_file_outputs["eps_t_mutation"][1]
end
=#
@btime StateSpaceRoutines.mutation!(Φ, Ψ, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c,
                                    tuning[:n_mh_steps], parallel = true)
@btime out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true)

## Whole TPF Tests
Random.seed!(47)
out_no_parallel = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false)
Random.seed!(47)
out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true)
@testset "TPF tests" begin
    @test out_no_parallel[1] ≈ -302.99967306704133
    @test out_parallel_one_worker[1] ≈ -302.99967306704133
    ## Note when using more than 1 worker, this is not true because there is still a random step in mh_steps
#=
    # In Julia 1.5, seeding appears to be different
    if VERSION >=  v"1.5"
        @test out_parallel_one_worker[1] ≈ -307.9285106003482
    elseif VERSION >= v"1.0" # This is different than in Julia6 because @distributed seeds differently than @parallel
        @test out_parallel_one_worker[1] ≈ -303.64727963725073 # Julia 6 was: -306.8211172094595
    end
=#
end

for i in workers()
    rmprocs(i)
end