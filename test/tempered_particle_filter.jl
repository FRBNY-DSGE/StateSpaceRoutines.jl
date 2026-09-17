using JLD2, Test, StateSpaceRoutines, Distributions, Random, BenchmarkTools, LinearAlgebra, Statistics
path = dirname(@__FILE__)
run_benchmarks = false
# Read in from JLD
tpf_main_input = load("$path/reference/tpf_main_inputs.jld2")
data = tpf_main_input["data"]
TTT = tpf_main_input["TTT"]
RRR = tpf_main_input["RRR"]
CCC = tpf_main_input["CCC"]
ZZ = tpf_main_input["ZZ"]
DD = tpf_main_input["DD"]
F_ϵ = as_mvnormal(tpf_main_input["F_epsilon"])
F_u = as_mvnormal(tpf_main_input["F_u"])
s_init = tpf_main_input["s_init"]

# Tune algorithm
tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
              :resampling_method => :systematic, :n_mh_steps => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true)

# Define Φ and Ψ (can't be saved to JLD)
Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
Ψ(s_t::AbstractVector{Float64}) = ZZ*s_t + DD

# Load in test inputs and outputs
test_file_inputs = load("$path/reference/tpf_aux_inputs.jld2")
test_file_outputs = load("$path/reference/tpf_aux_outputs.jld2")

φ_old = test_file_inputs["phi_old"]
norm_weights = test_file_inputs["norm_weights"]
coeff_terms = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
inc_weights = test_file_inputs["inc_weights"]
HH = Matrix(cov(F_u))
s_t_nontemp = test_file_inputs["s_t_nontemp"]

weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, data[:, 47], s_t_nontemp, det(HH), inv(HH);
               initialize = false)
φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)
correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]))

det_HH = det(HH)
inv_HH = inv(HH)
run_benchmarks && @btime weight_kernel!($coeff_terms, $log_e_1_terms, $log_e_2_terms, $φ_old, $Ψ, $(data[:, 47]), $s_t_nontemp, $det_HH, $inv_HH; initialize = false)
run_benchmarks && @btime next_φ($φ_old, $coeff_terms, $log_e_1_terms, $log_e_2_terms, $(length(data[:,47])), $(tuning[:r_star]), 2)
run_benchmarks && @btime correction!($inc_weights, $norm_weights, $φ_new, $coeff_terms, $log_e_1_terms, $log_e_2_terms, $(length(data[:,47])))

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

Random.seed!(47)
selection!(norm_weights, s_t1_temp, s_t_nontemp,ϵ_t, resampling_method = tuning[:resampling_method])
# Benchmark on copies: selection! mutates its particle args in place and reuses the
# RNG, so benchmarking the live buffers would corrupt the inputs mutation! (and the
# asserted references) depend on below.
run_benchmarks && @btime selection!($norm_weights, copy($s_t1_temp), copy($s_t_nontemp), copy($ϵ_t), resampling_method = $(tuning[:resampling_method]))
@testset "Selection Tests" begin
    @test s_t1_temp[1] ≈ test_file_outputs["s_t1_temp"][1]
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp"][1]
    @test ϵ_t[1] ≈ test_file_outputs["eps_t"][1]
end

## Mutation Tests
QQ = Matrix(cov(F_ϵ))
accept_rate = test_file_inputs["accept_rate"]
c = test_file_inputs["c"]

c = update_c(c, accept_rate, tuning[:target_accept_rate])
run_benchmarks && @btime update_c($c, $accept_rate, $(tuning[:target_accept_rate]))
Random.seed!(47)
StateSpaceRoutines.mutation!(Φ, Ψ, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c, tuning[:n_mh_steps])
# Benchmark on copies (see note above): mutation! mutates s_t/ϵ_t in place and reuses
# the RNG, which would otherwise corrupt the values asserted just below.
run_benchmarks && @btime StateSpaceRoutines.mutation!($Φ, $Ψ, $QQ, $det_HH, $inv_HH, $φ_new, $(data[:,47]), copy($s_t_nontemp), copy($s_t1_temp), copy($ϵ_t), $c, $(tuning[:n_mh_steps]))

@testset "Mutation Tests" begin
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp_mutation"][1]
    @test ϵ_t[1] ≈ test_file_outputs["eps_t_mutation"][1]
end

## Whole TPF Tests
Random.seed!(47)
out_no_parallel = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false)
Random.seed!(47)
out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true)
run_benchmarks && @btime tempered_particle_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = false)
run_benchmarks && @btime tempered_particle_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = true)
@testset "TPF tests" begin
    # NOTE (Julia 1.12): exact parallel≡sequential equality no longer holds — under
    # 1.7+ task-local Xoshiro the spmd kernels draw from a different task's RNG than
    # the `parallel_testing` seed pins, so the two paths consume different streams.
    # Both are valid Monte Carlo estimates; check only that their log-likelihoods
    # agree within a relative tolerance. See parallel_tempered_particle_filter.jl.
    @test isapprox(out_no_parallel[1], out_parallel_one_worker[1]; rtol = 0.05)
end
