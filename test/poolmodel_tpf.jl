using Statistics, DSGE, DSGEModels, CSV, Dates
# using JLD2, FileIO, Statistics, StateSpaceRoutines, Test, PDMats
# using LinearAlgebra, Distributions, Random, DSGE, DSGEModels, CSV, Dates

# Read in from JLD
m805 = Model805()
m904 = Model904()
# The following stuff should be saved into tpf_main_input
lppd_805 = load("reference/m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2")["logscores"]
# lppd_805_4 = load("reference/m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=4_est=2_hor=4_samp=SMC.jld2")["logscores"]
lppd_904 = load("reference/m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2")["logscores"]
# lppd_904_4 = load("reference/m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=4_est=2_hor=4_samp=SMC.jld2")["logscores"]
lppd_805 = vec(mean(lppd_805, dims = 1))
# lppd_805_4 = vec(mean(lppd_805_4, dims = 1))
lppd_904 = vec(mean(lppd_904, dims = 1))
# lppd_904_4 = vec(mean(lppd_904_4, dims = 1))
y805 = CSV.read("reference/realtime_spec=m805_hp=true_vint=170410.csv")
datevec = Vector{Date}(y805[y805.date .>= Date("1991-12-31"),1])
y805 = Matrix{Float64}(Matrix(y805[y805.date .>= Date("1991-12-31"),:])[:,2:end]')
y904 = CSV.read("reference/realtime_spec=m904_hp=true_vint=170410.csv")
y904 = Matrix{Float64}(Matrix(y904[y904.date .>= Date("1991-12-31"),:])[:,2:end]')
m = PoolModel(Dict(:Model805 => y805, :Model904 => y904), 4,
               Dict(:Model805 => lppd_805, :Model904 => lppd_904),
               [m805, m904])
data = zeros(1,get_periods(m))
# tempered_particle_filter(data, get_Φ(m), get_Ψ(m), get_F_ϵ(m),
#                          get_F_u(m), draw_prior(m); parallel = false,
#                          dynamic_measurement = true, poolmodel = true)

# tpf_main_input = load("reference/tpf_main_inputs_poolmodel.jld2")
# data     = tpf_main_input["data"]
# lppd_805 = tpf_main_input["lppd_805"]
# lppd_904 = tpf_main_input["lppd_904"]
# y805     = tpf_main_input["y805"]
# y904     = tpf_main_input["y904"]
# m        = tpf_main_input["pm"]
# s_init = tpf_main_input["s_init"]

# Define transition eq, measurement eq, and shock distributions
Φ = get_Φ(m)
Ψ = get_Ψ(m)
Ψ47(x) = get_Ψ(m)(x,47)
F_ϵ = get_F_ϵ(m)
F_u = get_F_u(m)

# Tune algorithm
tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
              :resampling_method => :systematic, :n_mh_steps => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true)

# Load in test inputs and outputs
test_file_inputs = load("reference/tpf_aux_inputs.jld2")
test_file_outputs = load("reference/tpf_aux_outputs.jld2")

φ_old = test_file_inputs["phi_old"]
# φ_old = (2 * pi)^(-1/2)
norm_weights = test_file_inputs["norm_weights"]
coeff_terms = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
inc_weights = test_file_inputs["inc_weights"]
HH = ones(1,1)
# s_t_nontemp = test_file_inputs["s_t_nontemp"]
s_t_nontemp = [.5; .5] * ones(1,1000) # this might be desirable since all particles same then

weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ47, data[:, 47], s_t_nontemp, det(HH), inv(HH); initialize = false, parallel = false,
               dynamic_measurement = true, poolmodel = true)
φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)
correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]))
true_inc_wt = φ_new^(1/2) * coeff_terms[1] * exp(log_e_1_terms[1]) * exp(φ_new * log_e_2_terms[1])
# true_inc_wts = test_file_outputs["phi_new"]^(1/2) * coeff_terms[1] * exp(log_e_1_terms[1]) *
#     exp(test_file_outputs["phi_new"] * log_e_2_terms[1])

@testset "Corection and Auxiliary Tests" begin
    @test coeff_terms[1] ≈ (φ_old)^(-1/2) # test_file_outputs["coeff_terms"][1]
    @test log_e_1_terms[1] ≈ get_Ψ(m)(s_t_nontemp[:,1], 47) * -φ_old # test_file_outputs["log_e_1_terms"][1]
    @test log_e_2_terms[1] ≈ get_Ψ(m)(s_t_nontemp[:,1], 47) # test_file_outputs["log_e_2_terms"][1]
    @test φ_new ≈ 1.0 # test_file_outputs["phi_new"]
    @test inc_weights[1] ≈ true_inc_wt # test_file_outputs["inc_weights"][1]
end

## Mutation Tests
# s_t1_temp = test_file_inputs["s_t1_temp"]
s_t1_temp = [.49; .51] * ones(1,1000)
ϵ_t = reshape(test_file_inputs["eps_t"][1,:], 1, 1000)
QQ = F_ϵ.σ * ones(1,1)
accept_rate = test_file_inputs["accept_rate"]
c = test_file_inputs["c"]

c = update_c(c, accept_rate, tuning[:target_accept_rate])
Random.seed!(47)
StateSpaceRoutines.mutation!(Φ, Ψ47, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c, tuning[:n_mh_steps]; dynamic_measurement = true, poolmodel = true)
@testset "Mutation Tests" begin
    @test s_t_nontemp[1] ≈ 0.2699446459556935 # test_file_outputs["s_t_nontemp_mutation"][1]
    @test ϵ_t[1] ≈ -0.693335270423147625246545 # test_file_outputs["eps_t_mutation"][1]
end

## Whole TPF Tests
Random.seed!(47)
s_init = reshape(rand(Uniform(.4, .6), 1000), 1, 1000)
s_init = [s_init; 1 .- s_init]
out_no_parallel = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none,fixed_sched = [1.], parallel = false, dynamic_measurement = true, poolmodel = true)
Random.seed!(47)
out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, fixed_sched = [1.], parallel = true, dynamic_measurement = true, poolmodel = true)
@testset "TPF tests" begin
    @test out_no_parallel[1] ≈ -423.17791042050027
    @test out_parallel_one_worker[1] ≈ -423.2233360059381
end

nothing
