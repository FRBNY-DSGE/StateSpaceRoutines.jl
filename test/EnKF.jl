using JLD2, Distributions, StateSpaceRoutines, BenchmarkTools
test_parallel = false
# Read in from JLD
data, TTT, RRR, CCC, ZZ, DD, F_ϵ, F_u, s_init = JLD2.jldopen("reference/tpf_main_inputs.jld2") do tpf_main_input
    tpf_main_input["data"],
    tpf_main_input["TTT"],
    tpf_main_input["RRR"],
    tpf_main_input["CCC"],
    tpf_main_input["ZZ"],
    tpf_main_input["DD"],
    tpf_main_input["F_epsilon"],
    tpf_main_input["F_u"],
    tpf_main_input["s_init"]
end

# Tune algorithm
n_particles = 1000
n_presample_periods = 0
allout = true
get_t_particle_dist = true

# Define Φ and Ψ (can't be saved to JLD)
Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
Ψ(s_t::AbstractVector{Float64}) = ZZ*s_t + DD

s0_mean = vec(mean(s_init, dims = 2))
P0_mean = cov(s_init, dims = 2)

kalman_out = kalman_filter(data, TTT, RRR, CCC, cov(F_ϵ), ZZ, DD, cov(F_u), s0_mean, P0_mean)#, s_init, Matrix{Float64}(undef,0,0))

#s_0 = rand(DegenerateMvNormal(kalman_out[6], kalman_out[7]), n_particles)
# repeat(kalman_out[6], outer = [1,n_particles])
iters100 = zeros(50) # Run EnKF 100 times to get loglh close to truth
tpf_iters = zeros(50)
s_init5 = repeat(s_init, outer = [1, 10])
for i in 1:length(iters100)
    if i % 20 == 0
        @show i
    end
    out = ensemble_kalman_filter(data, Φ, Ψ, F_ϵ, F_u, s_init;
                                 n_particles = n_particles, n_presample_periods = n_presample_periods,
                                 allout = allout, get_t_particle_dist = get_t_particle_dist,
                                 verbose = :none)
    iters100[i] = out[1]

    temp_out = tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; #fixed_sched = [1.0],
                       n_particles = n_particles, n_presample_periods = n_presample_periods,
                       allout = allout, verbose = :none)

    tpf_iters[i] = temp_out[1]
end

#=@show "Timing"
@btime ensemble_kalman_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init;
                       n_particles = $n_particles, n_presample_periods = $n_presample_periods,
                       allout = $allout, verbose = :none) ## 472.4 ms
=#
@assert abs(mean(iters100) - sum(kalman_out[1])) < 0.75 ## Randomness from the particles + initial states
# @assert abs(mean(tpf_iters) - sum(kalman_out[1])) < 0.75 ## Generally not true


# Testing that EnKF works when n_particles < n_states (this is the classic case for EnKF)
## Result doesn't matter so much - just testing that it runs.
n_particles = 4

iters_low = zeros(10) # Run EnKF 100 times to get loglh close to truth
s_init2 = s_init[:,sample(1:size(s_init,2), n_particles, replace = false)]
for i in 1:length(iters_low)
    if i % 20 == 0
        @show i
    end
    out = ensemble_kalman_filter(data, Φ, Ψ, F_ϵ, F_u, s_init2;
                                 n_particles = n_particles, n_presample_periods = n_presample_periods,
                                 allout = allout, get_t_particle_dist = get_t_particle_dist,
                                 verbose = :none)
    iters_low[i] = out[1]
end

# Testing the EnKF run when n_states = 1
TTT1 = TTT[1,1]
RRR1 = RRR[1,1]
CCC1 = CCC[1]
ZZ1 = ZZ[:,1]
DD1 = DD[1]

one_Φ(s_t::Float64, ϵ_t::Float64) = TTT1*s_t + RRR1*ϵ_t + CCC1
one_Ψ(s_t::Float64) = ZZ1 .* s_t .+ DD
one_one_Ψ(s_t::Float64) = (ZZ1 .* s_t .+ DD)[1]

F_u1 = Normal(var(F_u)[1])
F_ϵ1 = Normal(var(F_ϵ)[1])
s_init1 = s_init[1,:]
n_particles = 1000

one_outs = ensemble_kalman_filter(data, one_Φ, one_Ψ, F_ϵ1, F_u, s_init1;
                                  n_particles = n_particles, n_presample_periods = n_presample_periods,
                                  allout = allout, get_t_particle_dist = get_t_particle_dist,
                                  verbose = :none)

obs_one_outs = ensemble_kalman_filter(vec(data[1,:]), one_Φ, one_one_Ψ, F_ϵ1, F_u1, s_init1;
                                      n_particles = n_particles, n_presample_periods = n_presample_periods,
                                      allout = allout, get_t_particle_dist = get_t_particle_dist,
                                      verbose = :none)

# Case when n_obs = 1 but n_states > 1
one_two_Ψ(s_t::Vector{Float64}) = (ZZ * s_t .+ DD)[1]

obs_two_outs = ensemble_kalman_filter(vec(data[1,:]), Φ, one_two_Ψ, F_ϵ, F_u1, s_init;
                                      n_particles = n_particles, n_presample_periods = n_presample_periods,
                                      allout = allout, get_t_particle_dist = get_t_particle_dist,
                                      verbose = :none)

# Test Parallel with 1 worker
para1 = zeros(50) # Run EnKF 100 times to get loglh close to truth
for i in 1:length(para1)
    if i % 20 == 0
        @show i
    end
    out = ensemble_kalman_filter(data, Φ, Ψ, F_ϵ, F_u, s_init;
                                 n_particles = n_particles, n_presample_periods = n_presample_periods,
                                 allout = allout, get_t_particle_dist = get_t_particle_dist,
                                 verbose = :none, parallel = true)
    para1[i] = out[1]
end
@assert abs(mean(para1) - sum(kalman_out[1])) < 0.75

#=@show "Parallel Timing"
@btime ensemble_kalman_filter($data, $Φ, $Ψ, $F_ϵ, $F_u, $s_init;
                       n_particles = $n_particles, n_presample_periods = $n_presample_periods,
                       allout = $allout, verbose = :none, parallel = true) ##
=#
