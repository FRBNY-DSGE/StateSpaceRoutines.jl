using JLD, JLD2, Test, Distributions, Random, StateSpaceRoutines, BenchmarkTools, ParallelDataTransfer, DistributedArrays, OffsetArrays, DistributedArrays.SPMD

addproc_num = 0 ## Set to 0 to not add workers
nparts_mill = false
nparts_mult = nparts_mill ? 10 : 1
run_timing  = false
only_tpf    = true
n_states1   = false
n_shocks1   = false

@show n_states1, n_shocks1
@show "Parallel " * string(addproc_num)
@show "No of particles (in thousands): " * string(nparts_mult)

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

if n_states1
    s_init = vec(s_init[1,:])
    TTT = n_shocks1 ? 0.5 : TTT[1,1]
    RRR = RRR[1,:]'
    CCC = CCC[1]
    ZZ = ZZ[:,1]

    if n_shocks1
        RRR = 0.9#RRR[1,1]
        F_ϵ = Normal(F_ϵ.μ[1], F_ϵ.Σ[1,1])
    end
end

if n_shocks1 && !n_states1
    RRR = RRR[:,1]
    F_ϵ = Normal(F_ϵ.μ[1], F_ϵ.Σ[1,1])
end

if nparts_mill
    s_init = repeat(s_init, outer=(1,nparts_mult))
end

# Tune algorithm
tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
              :resampling_method => :systematic, :n_mh_steps => 1,
              :n_particles => 1000 * nparts_mult, :n_presample_periods => 0,
              :allout => true, :parallel => true)

# Define Φ and Ψ (can't be saved to JLD)
if !n_states1 && n_shocks1
    Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t .+ RRR .* ϵ_t .+ CCC
else
    Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
end
Ψ(s_t::AbstractVector{Float64}) = ZZ .* s_t .+ DD

Φ(s_t::Float64, ϵ_t::Float64) = TTT*s_t + RRR*ϵ_t + CCC
Ψ(s_t::Float64) = ZZ*s_t + DD

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

if n_states1
    s_t_nontemp = s_t_nontemp[1,:]#reshape(s_t_nontemp[1,:], (1,size(s_t_nontemp,2)))
end

if nparts_mill
    s_t_nontemp = repeat(s_t_nontemp, outer=(1,nparts_mult))
    coeff_terms = repeat(coeff_terms,nparts_mult)
    log_e_1_terms = repeat(log_e_1_terms, nparts_mult)
    log_e_2_terms = repeat(log_e_2_terms, nparts_mult)
    inc_weights = repeat(inc_weights, nparts_mult)
    norm_weights = repeat(norm_weights, nparts_mult)
end

ENV["frbnyjuliamemory"] = "1G"
if addproc_num > 0
    myprocs = addprocs_frbny(addproc_num)
end
@everywhere using JLD, JLD2, Test, Distributions, Random, StateSpaceRoutines, BenchmarkTools, ParallelDataTransfer, DistributedArrays, DistributedArrays.SPMD

@everywhere tpf_main_input = load("reference/tpf_main_inputs.jld2")
@everywhere TTT = tpf_main_input["TTT"]
@everywhere RRR = tpf_main_input["RRR"]
@everywhere CCC = tpf_main_input["CCC"]
@everywhere ZZ = tpf_main_input["ZZ"]
@everywhere DD = tpf_main_input["DD"]

@everywhere Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
@everywhere Ψ(s_t::AbstractVector{Float64}) = ZZ*s_t + DD
@everywhere Φ(s_t::Float64, ϵ_t::Float64) = TTT*s_t + RRR*ϵ_t + CCC
@everywhere Ψ(s_t::Float64) = ZZ*s_t + DD

if n_states1
    if n_shocks1
        @everywhere TTT = 0.5
        @everywhere RRR = 0.9
        @everywhere ZZ = ZZ[:,1]
    else
        @everywhere TTT = TTT[1,1]
        @everywhere RRR = RRR[1,:]'
        @everywhere ZZ = ZZ[:,1]
    end
    @everywhere CCC = CCC[1]

    #=if n_shocks1
        @everywhere F_ϵ = Normal(F_ϵ.μ[1], F_ϵ.Σ[1,1])
    end=#
end

if n_shocks1 && !n_states1
    @everywhere RRR = RRR[:,1]
    # @everywhere F_ϵ = Normal(F_ϵ.μ[1], F_ϵ.Σ[1,1])
end

if n_shocks1 || n_states1
    if !n_states1
        @everywhere Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t .+ RRR .* ϵ_t .+ CCC
        @everywhere Ψ(s_t::AbstractVector{Float64}) = ZZ*s_t + DD
    else
        @everywhere Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t .+ RRR*ϵ_t .+ CCC
        @everywhere Ψ(s_t::AbstractVector{Float64}) = ZZ .* s_t .+ DD
    end
    @everywhere Φ(s_t::Float64, ϵ_t::Float64) = TTT*s_t + RRR*ϵ_t + CCC
    @everywhere Ψ(s_t::Float64) = ZZ*s_t + DD
end

if !only_tpf
#= StateSpaceRoutines.sendto(workers(), Φ = Φ)
StateSpaceRoutines.sendto(workers(), Ψ = Ψ)
StateSpaceRoutines.sendto(workers(), coeff_terms = coeff_terms)
StateSpaceRoutines.sendto(workers(), log_e_1_terms = log_e_1_terms)
StateSpaceRoutines.sendto(workers(), log_e_2_terms = log_e_2_terms)
StateSpaceRoutines.sendto(workers(), data = data)
StateSpaceRoutines.sendto(workers(), s_t_nontemp = s_t_nontemp)
StateSpaceRoutines.sendto(workers(), HH = HH)=#

s_t_nontemp = distribute(s_t_nontemp, dist = [1, nworkers()])
coeff_terms = distribute(coeff_terms)
log_e_1_terms = distribute(log_e_1_terms)
log_e_2_terms = distribute(log_e_2_terms)
inc_weights = distribute(inc_weights)
norm_weights = distribute(norm_weights)

coeff_vec = convert(Vector, coeff_terms)
log_e_1_vec = convert(Vector, log_e_1_terms)
log_e_2_vec = convert(Vector, log_e_2_terms)
s_t_nontemp_vec = convert(Matrix, s_t_nontemp)

if run_timing
    @show "Weight_kernel with vectors in sequential"
    @btime weight_kernel!(coeff_vec, log_e_1_vec,
                          log_e_2_vec, φ_old, Ψ, data[:, 47], s_t_nontemp_vec,
                          det(HH), inv(HH), initialize = false)

    @show "Weight_kernel with DistributedArrays in parallel"
    @btime spmd(weight_kernel!,coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, data[:, 47], s_t_nontemp, det(HH), inv(HH);
                pids = workers())
end
spmd(weight_kernel!,coeff_terms, log_e_1_terms, log_e_2_terms, φ_old, Ψ, data[:, 47], s_t_nontemp, det(HH), inv(HH);
            pids = workers())
#=weight_kernel!(coeff_vec, log_e_1_vec,
                      log_e_2_vec, φ_old, Ψ, data[:, 47], s_t_nontemp_vec,
                      det(HH), inv(HH), initialize = false)
# Not needed b/c weight_kernel run earlier
=#
#=φ_new = @sync @distributed (+) for p in workers()
    next_φ(φ_old, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]), tuning[:r_star], 2)
end
φ_new /= nworkers()
## Note above is incorrect since we can't just take the mean - just wanted to confirm that it runs
=#
φ_new = next_φ(φ_old, convert(Vector, coeff_terms), convert(Vector, log_e_1_terms), convert(Vector, log_e_2_terms),
               length(data[:,47]), tuning[:r_star], 2)
spmd(correction!, inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]);
     pids = workers())

# Reference run
#=φ_new_ref = next_φ(φ_old, coeff_vec, log_e_1_vec, log_e_2_vec,
               length(data[:,47]), tuning[:r_star], 2)
spmd(correction!, inc_weights, norm_weights, φ_new, coeff_terms, log_e_1_terms, log_e_2_terms, length(data[:,47]);
     pids = workers())=#

@testset "Corection and Auxiliary Tests" begin
    @test convert(Vector, coeff_terms)[1] ≈ test_file_outputs["coeff_terms"][1]
    @test convert(Vector, log_e_1_terms)[1] ≈ test_file_outputs["log_e_1_terms"][1]
    @test convert(Vector, log_e_2_terms)[1] ≈ test_file_outputs["log_e_2_terms"][1]
    @test φ_new ≈ test_file_outputs["phi_new"]
    @test convert(Vector, inc_weights)[1] ≈ test_file_outputs["inc_weights"][1]
end

## Selection Tests
s_t1_temp = test_file_inputs["s_t1_temp"]
ϵ_t = test_file_inputs["eps_t"]

if nparts_mill
    s_t1_temp = repeat(s_t1_temp, outer=(1,nparts_mult))
    ϵ_t = repeat(ϵ_t, outer=(1,nparts_mult))
end

# ENV["frbnyjuliamemory"] = "5G"
# myprocs = addprocs_frbny(48)
# @everywhere using JLD2, Test, Distributions, Random, StateSpaceRoutines, BenchmarkTools, ParallelDataTransfer
#@everywhere include("../src/filters/tempered_particle_filter/correction.jl")
#@everywhere include("../src/util.jl")

Random.seed!(47)
@everywhere Random.seed!(47)
s_t1_temp = distribute(s_t1_temp, dist = [1, nworkers()])
ϵ_t = distribute(ϵ_t, dist = [1, nworkers()])
# s_t_nontemp_std = convert(Array, s_t_nontemp)
selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t, resampling_method = tuning[:resampling_method])
# s_t_nontemp = distribute(s_t_nontemp_std)
s_t_nontemp_vec = copy(convert(Matrix, s_t_nontemp))
s_t1_temp_vec = copy(convert(Matrix, s_t1_temp))
ϵ_t_vec = copy(convert(Matrix, ϵ_t))
@testset "Selection Tests" begin
    @test s_t1_temp_vec[1] ≈ test_file_outputs["s_t1_temp"][1]
    @test s_t_nontemp_vec[1] ≈ test_file_outputs["s_t_nontemp"][1]
    @test ϵ_t_vec[1] ≈ test_file_outputs["eps_t"][1]
end

## Mutation Tests
QQ = n_shocks1 ? var(F_ϵ) : cov(F_ϵ)
accept_rate = test_file_inputs["accept_rate"]
c = test_file_inputs["c"]

c = update_c(c, accept_rate, tuning[:target_accept_rate])

#@assert false

# No mutation in BSPF, although I will implement this later.
#=if run_timing
    @show "Sequential mutation"
    Random.seed!(47)
    @everywhere Random.seed!(47)
    @btime StateSpaceRoutines.mutation!($Φ, $Ψ, $QQ, det($HH), inv($HH), $φ_new, $data[:,47], $s_t_nontemp_vec, $s_t1_temp_vec, $ϵ_t_vec, $c, $tuning[:n_mh_steps])
    @show "Parallel mutation"
    Random.seed!(47)
    @everywhere Random.seed!(47)
    @btime spmd(StateSpaceRoutines.mutation!, $Φ, $Ψ, $QQ, det($HH), inv($HH), $φ_new, $data[:,47], $s_t_nontemp, $s_t1_temp, $ϵ_t, $c, $tuning[:n_mh_steps])
end
=#
# @assert false

Random.seed!(47)
@everywhere Random.seed!(47)
StateSpaceRoutines.mutation!(Φ, Ψ, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp_vec, s_t1_temp_vec, ϵ_t_vec, c, tuning[:n_mh_steps])
@everywhere Random.seed!(47)
Random.seed!(47)
spmd(StateSpaceRoutines.mutation!, Φ, Ψ, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c, tuning[:n_mh_steps])
@testset "Mutation Tests" begin
    @test convert(Matrix, s_t_nontemp)[1] ≈ s_t_nontemp_vec[1]#test_file_outputs["s_t_nontemp_mutation"][1]
    @test convert(Matrix, ϵ_t)[1] ≈ ϵ_t_vec[1]#test_file_outputs["eps_t_mutation"][1]
end
end
# @btime StateSpaceRoutines.mutation!(Φ, Ψ, QQ, det(HH), inv(HH), φ_new, data[:,47], s_t_nontemp, s_t1_temp, ϵ_t, c,
#                                     tuning[:n_mh_steps], parallel = true)
# @btime out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true)

## Whole TPF Tests

# BSPF Test
@everywhere seed_val = 47

tuning[:fixed_sched] = [1.0]
@everywhere Random.seed!(seed_val)
out_no_parallel = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false, parallel_testing = true)
@everywhere Random.seed!(seed_val)
out_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true, parallel_testing = true)

if run_timing
    @show "BSPF in sequential"
    @btime out_no_parallel = tempered_particle_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = false)
    @show "BSPF in parallel"
    @btime out_parallel_one_worker = tempered_particle_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = true)
end
@show out_no_parallel[1], out_parallel_one_worker[1]

# Adaptive TPF Test
if !n_shocks1 # Bisection doesn't have real roots in this case
    delete!(tuning, :fixed_sched)
    @everywhere Random.seed!(seed_val)
    adapt_no_parallel = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false, parallel_testing = true)
    @everywhere Random.seed!(seed_val)
    adapt_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true, parallel_testing = true)

if run_timing
    @show "Adaptive TPF in sequential"
    @btime adapt_no_parallel = tempered_particle_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = false)
    @show "Adaptive TPF in parallel"
    @btime adapt_parallel_one_worker = tempered_particle_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = true)
end
@show adapt_no_parallel[1], adapt_parallel_one_worker[1]
end

# Fixed Schedule TPF Test
tuning[:fixed_sched] = [0.1,0.15,0.3,1.0]
@everywhere Random.seed!(seed_val)
fixed_no_parallel = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = false, parallel_testing = true)
@everywhere Random.seed!(seed_val)
fixed_parallel_one_worker = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning..., verbose = :none, parallel = true, parallel_testing = true)

if run_timing
    @show "Fixed schedule TPF in sequential"
    @btime fixed_no_parallel = tempered_particle_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = false)
    @show "Fixed schedule TPF in parallel"
    @btime fixed_parallel_one_worker = tempered_particle_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init; $tuning..., verbose = :none, parallel = true)
end
@show fixed_no_parallel[1], fixed_parallel_one_worker[1]

@testset "TPF tests" begin
    if addproc_num <= 1
        @test all(out_no_parallel[2] .≈ out_parallel_one_worker[2])
        @test (n_shocks1) || all(adapt_no_parallel[2] .≈ adapt_parallel_one_worker[2])
        @test all(fixed_no_parallel[2] .≈ fixed_parallel_one_worker[2])
    else
        @test abs(out_no_parallel[1] - out_parallel_one_worker[1]) ≤ 10.0
        @test (n_shocks1) || abs(adapt_no_parallel[1] - adapt_parallel_one_worker[1]) ≤ 10.0
        @test abs(fixed_no_parallel[1] - fixed_parallel_one_worker[1]) ≤ 10.0
    end
    ## Note when using more than 1 worker, equality is not true because there is still a random step in mh_steps
end

for i in workers()
    rmprocs(i)
end
