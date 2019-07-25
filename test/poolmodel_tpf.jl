using DSGE, DSGEModels, CSV, Dates

# Set up and read in from JLD
m805 = Model805()
m904 = Model904()
# The following stuff should be saved into tpf_main_input
# lpd_805 = load("reference/m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2")["logscores"]
# lpd_904 = load("reference/m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2")["logscores"]
# lpd_805 = vec(mean(lpd_805, dims = 1))
# lpd_904 = vec(mean(lpd_904, dims = 1))

y805 = CSV.read("reference/realtime_spec=m805_hp=true_vint=170410.csv") # is this data confidential???
datevec = Vector{Date}(y805[y805.date .>= Date("1991-12-31"),1])
y805 = Matrix{Float64}(Matrix(y805[y805.date .>= Date("1991-12-31"),:])[:,2:end]')
y904 = CSV.read("reference/realtime_spec=m904_hp=true_vint=170410.csv")
y904 = Matrix{Float64}(Matrix(y904[y904.date .>= Date("1991-12-31"),:])[:,2:end]')
tpf_main_input = load("reference/tpf_poolmodel.jld2")
data = tpf_main_input["data"]
tuning = tpf_main_input["tuning"]
lpd_805 = tpf_main_input["lpd_805"]
lpd_904 = tpf_main_input["lpd_904"]
m = PoolModel(Dict(:Model805 => y805, :Model904 => y904), 4,
               Dict(:Model805 => exp.(lpd_805), :Model904 => exp.(lpd_904)),
               [m805, m904])

# Define transition eq, measurement eq, and shock distributions
Ψ47_pm(x) = get_Ψ(m)(x,47)

# Load in test inputs and outputs
test_file_inputs = load("reference/tpf_aux_inputs.jld2")
test_file_outputs_pm = load("reference/tpf_aux_outputs_poolmodel.jld2")

φ_old = test_file_inputs["phi_old"]
norm_weights = test_file_inputs["norm_weights"]
coeff_terms = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
inc_weights = test_file_inputs["inc_weights"]
HH = ones(1,1)

s_t_nontemp = [.5; .5] * ones(1,1000) # this might be desirable since all particles same then
Random.seed!(1793)
weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ47_pm, data[:, 47],
               s_t_nontemp, det(HH), inv(HH);
               initialize = false, parallel = false,
               dynamic_measurement = true, poolmodel = true)
φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)
correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]))
true_inc_wt = φ_new^(1/2) * coeff_terms[1] * exp(log_e_1_terms[1]) * exp(φ_new * log_e_2_terms[1])

@testset "Corection and Auxiliary Tests" begin
    @test coeff_terms[1] ≈ (φ_old)^(-1/2)
    @test log_e_1_terms[1] ≈ log(get_Ψ(m)(s_t_nontemp[:,1], 47)) * -φ_old
    @test log_e_2_terms[1] ≈ log(get_Ψ(m)(s_t_nontemp[:,1], 47))
    @test φ_new ≈ 1.0
    @test inc_weights[1] ≈ true_inc_wt
end

φ_old = test_file_inputs["phi_old"]
norm_weights = test_file_inputs["norm_weights"]
coeff_terms = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
inc_weights = test_file_inputs["inc_weights"]
s_t_nontemp0 = [.5; .5] * ones(1, 900)
s_t_nontemp = hcat(s_t_nontemp0, repeat([.49; .51], 1, 100))
weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ47_pm, data[:, 47],
               s_t_nontemp, det(HH), inv(HH);
               initialize = false, parallel = false,
               dynamic_measurement = true, poolmodel = true)
φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)
correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]))
true_inc_wt = φ_new^(1/2) * (coeff_terms[1] * exp(log_e_1_terms[1])
                             * exp(φ_new * log_e_2_terms[1]) * 900/1000
                             + coeff_terms[end] * exp(log_e_1_terms[end])
                             * exp(φ_new * log_e_2_terms[end]) * 100/1000)

@testset "Ensure different states lead to different measurement errors" begin
    @test norm_weights[1] != norm_weights[end]
    @test log_e_1_terms[1] != log_e_1_terms[end]
    @test log_e_2_terms[1] != log_e_2_terms[end]
    @test inc_weights[1] != inc_weights[end]
    @test true_inc_wt != mean(inc_weights)
end

## Mutation Tests
s_t1_temp = [.49; .51] * ones(1,1000)
ϵ_t = reshape(test_file_inputs["eps_t"][1,:], 1, 1000)
QQ = get_F_ϵ(m).σ * ones(1,1)
accept_rate = test_file_inputs["accept_rate"]
c = test_file_inputs["c"]

c = update_c(c, accept_rate, tuning[:target_accept_rate])
Random.seed!(47)
StateSpaceRoutines.mutation!(get_Φ(m), Ψ47_pm, QQ, det(HH), inv(HH), φ_new, data[:,47],
                             s_t_nontemp, s_t1_temp, ϵ_t, c, tuning[:n_mh_steps];
                             dynamic_measurement = true, poolmodel = true)
@testset "Mutation Tests" begin
    @test s_t_nontemp[1] ≈ test_file_outputs_pm["s_t_nontemp"][1]
    @test ϵ_t[1] ≈ test_file_outputs_pm["eps_t"][1]
end

## Whole TPF Tests
Random.seed!(47)
s_init = reshape(rand(Uniform(.4, .6), 1000), 1, 1000)
s_init = [s_init; 1 .- s_init]
out_no_parallel = tempered_particle_filter(data, get_Φ(m), get_Ψ(m), get_F_ϵ(m), get_F_u(m), s_init;
                                           tuning..., verbose = :none, fixed_sched = [1.],
                                           parallel = false, dynamic_measurement = true, poolmodel = true)
Random.seed!(47)
out_parallel_one_worker = tempered_particle_filter(data, get_Φ(m), get_Ψ(m), get_F_ϵ(m), get_F_u(m), s_init;
                                                   tuning..., verbose = :none, fixed_sched = [1.],
                                                   parallel = true, dynamic_measurement = true, poolmodel = true)
@testset "TPF tests" begin
    @test out_no_parallel[1] ≈ test_file_outputs_pm["out_no_parallel"][1]
    @test out_parallel_one_worker[1] ≈ test_file_outputs_pm["out_parallel_one_worker"][1]
end

nothing
