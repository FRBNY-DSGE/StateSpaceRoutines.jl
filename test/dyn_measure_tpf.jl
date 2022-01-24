using JLD2

# Read in from JLD
tpf_main_input = load("reference/tpf_main_inputs.jld2")
data   = tpf_main_input["data"]
TTT    = tpf_main_input["TTT"]
RRR    = tpf_main_input["RRR"]
CCC    = tpf_main_input["CCC"]
ZZ     = tpf_main_input["ZZ"]
DD     = tpf_main_input["DD"]
F_ϵ    = tpf_main_input["F_epsilon"]
F_u    = tpf_main_input["F_u"]
s_init = tpf_main_input["s_init"]

# Tune algorithm
tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
              :resampling_method => :systematic, :n_mh_steps => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true)

# Define Φ and Ψ (can't be saved to JLD)
Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
Ψt(s_t::AbstractVector{Float64}, t) = t < 100 ? ZZ*s_t + DD : DD
Ψ(x) = Ψt(x, 47)

# Load in test inputs and outputs
test_file_inputs = load("reference/tpf_aux_inputs.jld2")
test_file_outputs = load("reference/tpf_aux_outputs.jld2")

φ_old         = test_file_inputs["phi_old"]
norm_weights  = test_file_inputs["norm_weights"]
coeff_terms   = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
inc_weights   = test_file_inputs["inc_weights"]
s_t_nontemp   = test_file_inputs["s_t_nontemp"]
HH            = cov(F_u)

weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, data[:, 47], s_t_nontemp, det(HH), inv(HH);
               initialize = false)
φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)

correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]))

@testset "Correction and Auxiliary Tests" begin
    @test coeff_terms[1]   ≈ test_file_outputs["coeff_terms"][1]
    @test log_e_1_terms[1] ≈ test_file_outputs["log_e_1_terms"][1]
    @test log_e_2_terms[1] ≈ test_file_outputs["log_e_2_terms"][1]
    @test φ_new            ≈ test_file_outputs["phi_new"]
    @test inc_weights[1]   ≈ test_file_outputs["inc_weights"][1]
end

# Incorrect measurement equation
Ψtinc(s_t::AbstractVector{Float64}, t) = t < 46 ? ZZ*s_t + DD : ZZ*s_t + DD + [0.; 1; 0.];
Ψinc(x) = Ψtinc(x, 47)

φ_old         = test_file_inputs["phi_old"]
norm_weights  = test_file_inputs["norm_weights"]
coeff_terms   = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
inc_weights   = test_file_inputs["inc_weights"]
s_t_nontemp   = test_file_inputs["s_t_nontemp"]
HH            = cov(F_u)

weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψinc, data[:, 47], s_t_nontemp, det(HH), inv(HH);
               initialize = false)
φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)

correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]))
@testset "Correction and Auxiliary Tests with Dynamically Wrong Measurement Equation" begin
    @test !(log_e_1_terms[1] ≈ test_file_outputs["log_e_1_terms"][1])
    @test !(log_e_2_terms[1] ≈ test_file_outputs["log_e_2_terms"][1])
    @test !(φ_new            ≈ test_file_outputs["phi_new"])
    @test !(inc_weights[1]   ≈ test_file_outputs["inc_weights"][1])
end

## Selection Tests
Ψt(s_t::AbstractVector{Float64}, t) = t < 100 ? ZZ*s_t + DD : DD
Ψ(x) = Ψt(x, 47)
s_t1_temp = test_file_inputs["s_t1_temp"]
ϵ_t = test_file_inputs["eps_t"]

Random.seed!(47)
selection!(norm_weights, s_t1_temp, s_t_nontemp,ϵ_t, resampling_method = tuning[:resampling_method])
@testset "Selection Tests" begin
    @test s_t1_temp[1]   ≈ test_file_outputs["s_t1_temp"][1]
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp"][1]
    @test ϵ_t[1]         ≈ test_file_outputs["eps_t"][1]
end

## Mutation Tests
QQ = cov(F_ϵ)
accept_rate = test_file_inputs["accept_rate"]
c = test_file_inputs["c"]

c = update_c(c, accept_rate, tuning[:target_accept_rate])
Random.seed!(47)
StateSpaceRoutines.mutation!(Φ, Ψ, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c, tuning[:n_mh_steps])

@testset "Mutation Tests" begin
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp_mutation"][1]
    @test ϵ_t[1] ≈ test_file_outputs["eps_t_mutation"][1]
end

## Whole TPF Tests
Random.seed!(47)
out_no_parallel = tempered_particle_filter(data, Φ, Ψt, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false, dynamic_measurement = true)
Random.seed!(47)
out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψt, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true, dynamic_measurement = true)
@testset "TPF tests" begin
    @test out_no_parallel[1] ≈ -305.2043197924593#-302.99967306704133
    # See equivalent test in tempered_particle_filter.jl
    if VERSION >= v"1.5"
        @test out_parallel_one_worker[1] ≈ -305.2043197924593#-307.9285106003482
    elseif VERSION >= v"1.0"
        @test out_parallel_one_worker[1] ≈ -303.64727963725073 # Julia 6 was: -306.8211172094595
    end
end

nothing
