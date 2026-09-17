using JLD2, BenchmarkTools, Test, StateSpaceRoutines, Random, LinearAlgebra, Statistics, Distributions
path = dirname(@__FILE__)
run_benchmarks = false
# Set true to regenerate the RNG-dependent mutation references in
# tpf_aux_outputs.jld2 against the current Julia/dependency stack, then set back
# to false to test against them.
write_output = false

# Read in from JLD
tpf_main_input = load("$path/reference/tpf_main_inputs.jld2")
data   = tpf_main_input["data"]
TTT    = tpf_main_input["TTT"]
RRR    = tpf_main_input["RRR"]
CCC    = tpf_main_input["CCC"]
ZZ     = tpf_main_input["ZZ"]
DD     = tpf_main_input["DD"]
F_ϵ    = as_mvnormal(tpf_main_input["F_epsilon"])
F_u    = as_mvnormal(tpf_main_input["F_u"])
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
test_file_inputs = load("$path/reference/tpf_aux_inputs.jld2")
test_file_outputs = load("$path/reference/tpf_aux_outputs.jld2")

φ_old         = test_file_inputs["phi_old"]
norm_weights  = test_file_inputs["norm_weights"]
coeff_terms   = test_file_inputs["coeff_terms"]
log_e_1_terms = test_file_inputs["log_e_1_terms"]
log_e_2_terms = test_file_inputs["log_e_2_terms"]
inc_weights   = test_file_inputs["inc_weights"]
s_t_nontemp   = test_file_inputs["s_t_nontemp"]
HH            = Matrix(cov(F_u))

det_HH = det(HH)
inv_HH = inv(HH)
weight_kernel!(coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, data[:, 47], s_t_nontemp, det_HH, inv_HH;
               initialize = false)
φ_new = next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)
correction!(inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]))

run_benchmarks && @btime weight_kernel!($coeff_terms, $log_e_1_terms, $log_e_2_terms, $φ_old, $Ψ, $(data[:, 47]), $s_t_nontemp, $det_HH, $inv_HH; initialize = false)
run_benchmarks && @btime next_φ($φ_old, $coeff_terms, $log_e_1_terms, $log_e_2_terms, $(length(data[:,47])), $(tuning[:r_star]), 2)
run_benchmarks && @btime correction!($inc_weights, $norm_weights, $φ_new, $coeff_terms, $log_e_1_terms, $log_e_2_terms, $(length(data[:,47])))

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
HH            = Matrix(cov(F_u))

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
# (Ψt/Ψ already defined above and unchanged since — no need to redefine)
s_t1_temp = test_file_inputs["s_t1_temp"]
ϵ_t = test_file_inputs["eps_t"]

Random.seed!(47)
selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t, resampling_method = tuning[:resampling_method])
# Benchmark on copies: selection! mutates its particle args in place and reuses the
# RNG, so benchmarking the live buffers would corrupt the inputs that mutation! (and
# the saved/asserted references) depend on below.
run_benchmarks && @btime selection!($norm_weights, copy($s_t1_temp), copy($s_t_nontemp), copy($ϵ_t), resampling_method = $(tuning[:resampling_method]))
@testset "Selection Tests" begin
    @test s_t1_temp[1]   ≈ test_file_outputs["s_t1_temp"][1]
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp"][1]
    @test ϵ_t[1]         ≈ test_file_outputs["eps_t"][1]
end

## Mutation Tests
QQ = Matrix(cov(F_ϵ))
accept_rate = test_file_inputs["accept_rate"]
c = test_file_inputs["c"]

c = update_c(c, accept_rate, tuning[:target_accept_rate])
run_benchmarks && @btime update_c($c, $accept_rate, $(tuning[:target_accept_rate]))
Random.seed!(47)
StateSpaceRoutines.mutation!(Φ, Ψ, QQ, det_HH, inv_HH, φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c, tuning[:n_mh_steps])
# Benchmark on copies (see note above): mutation! mutates s_t/ϵ_t in place and reuses
# the RNG, which would otherwise corrupt the values saved/asserted just below.
run_benchmarks && @btime StateSpaceRoutines.mutation!($Φ, $Ψ, $QQ, $det_HH, $inv_HH, $φ_new, $(data[:,47]), copy($s_t_nontemp), copy($s_t1_temp), copy($ϵ_t), $c, $(tuning[:n_mh_steps]))

if write_output
    # Overwrite only the two RNG-dependent mutation keys, preserving every other
    # (deterministic) key in the shared reference file.
    refs = load("$path/reference/tpf_aux_outputs.jld2")
    refs["s_t_nontemp_mutation"] = copy(s_t_nontemp)
    refs["eps_t_mutation"]       = copy(ϵ_t)
    jldopen("$path/reference/tpf_aux_outputs.jld2", "w") do f
        for (k, v) in refs
            f[k] = v
        end
    end
    test_file_outputs = load("$path/reference/tpf_aux_outputs.jld2")
end

@testset "Mutation Tests" begin
    @test s_t_nontemp[1] ≈ test_file_outputs["s_t_nontemp_mutation"][1]
    @test ϵ_t[1] ≈ test_file_outputs["eps_t_mutation"][1]
end

## Whole TPF Tests
Random.seed!(47)
out_no_parallel = tempered_particle_filter(data, Φ, Ψt, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false, dynamic_measurement = true)
Random.seed!(47)
out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψt, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true, dynamic_measurement = true)
run_benchmarks && @btime tempered_particle_filter($data, $Φ, $Ψt, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = false, dynamic_measurement = true)
run_benchmarks && @btime tempered_particle_filter($data, $Φ, $Ψt, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = true, dynamic_measurement = true)

if write_output
    # Only the non-parallel (sequential) loglh is reproducible and worth pinning. The
    # parallel run is NOT reproducible on 1.12 — its spmd path draws from task-local
    # Xoshiro RNGs the seed never pins — so we don't store it (we MC-compare it below).
    refs = load("$path/reference/tpf_aux_outputs.jld2")
    refs["loglh_no_parallel_dyn"] = out_no_parallel[1]
    jldopen("$path/reference/tpf_aux_outputs.jld2", "w") do f
        for (k, v) in refs
            f[k] = v
        end
    end
    test_file_outputs = load("$path/reference/tpf_aux_outputs.jld2")
end

@testset "TPF tests" begin
    # Sequential loglh is deterministic → pin it against the regenerated reference.
    @test out_no_parallel[1] ≈ test_file_outputs["loglh_no_parallel_dyn"]
    # Parallel loglh is not reproducible run-to-run under task-local Xoshiro (see
    # NOTE in parallel_tempered_particle_filter.jl); only check it agrees with the
    # sequential estimate within Monte Carlo tolerance.
    @test isapprox(out_parallel_one_worker[1], out_no_parallel[1]; rtol = 0.05)
end

nothing
