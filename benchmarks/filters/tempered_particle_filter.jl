using StateSpaceRoutines
using QuantEcon: solve_discrete_lyapunov
using BenchmarkTools

include("../model_setup.jl")

# Loading the state-space components

const TTT = system[:TTT]
const RRR = system[:RRR]
const QQ  = system[:QQ]
Φ, Ψ, F_ϵ, F_u = compute_system_function(system)

# Tuning of the tempered particle filter algorithm

tuning = Dict(:r_star => 2.,
              :xtol => 1e-3, :resampling_method => :systematic,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true, :parallel => false, :verbose => :none)

# Generation of the initial state draws
n_states = n_states_augmented(m)
const s0 = zeros(n_states)
const P0 = solve_discrete_lyapunov(TTT, RRR*QQ*RRR')
U, E, V = svd(P0)
s_init = s0 .+ U*diagm(sqrt.(E))*randn(n_states, tuning[:n_particles])

tpf_trial = @benchmark tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; $(tuning...)) gcsample = true

# Optionally re-write the reference trial
# write_ref_trial(tpf_trial, "tempered_particle_filter")
print_all_benchmarks(tpf_trial, "../reference/tempered_particle_filter.jld", "tempered_particle_filter")
