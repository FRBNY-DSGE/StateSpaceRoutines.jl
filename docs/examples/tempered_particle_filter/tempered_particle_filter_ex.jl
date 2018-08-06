using DSGE, StateSpaceRoutines
using QuantEcon: solve_discrete_lyapunov
using DataFrames

# Setup the model and data

m = AnSchorfheide()
df = readtable("us.txt", header = false, separator = ' ')
data = convert(Matrix{Float64}, df)'

params = [2.09, 0.98, 2.25, 0.65, 0.34, 3.16, 0.51, 0.81, 0.98, 0.93, 0.19, 0.65, 0.24,
          0.115985, 0.294166, 0.447587]
update!(m, params)

# Solution to a Linear DSGE Model w/ IID Gaussian Errors

system  = compute_system(m)
TTT     = system[:TTT]
RRR     = system[:RRR]
CCC     = system[:CCC]
HH      = system[:EE]
DD      = system[:DD]
ZZ      = system[:ZZ]
QQ      = system[:QQ]

#Φ(s_t::Vector{Float64}, ϵ_t::Vector{Float64}) = TTT*s_t + RRR*ϵ_t + CCC
#Ψ(s_t::Vector{Float64}, u_t::Vector{Float64}) = ZZ*s_t + DD + u_t

#F_ϵ = Distributions.MvNormal(zeros(size(QQ, 1)), QQ)
#F_u = Distributions.MvNormal(zeros(size(HH, 1)), HH)

Φ, Ψ, F_ϵ, F_u = compute_system_function(system)

# Tuning of the tempered particle filter algorithm

## Old tuning parameters
#=tuning = Dict(:r_star => 2., :c => 0.3, :accept_rate => 0.4, :target => 0.4,
              :xtol => 0., :resampling_method => :systematic, :N_MH => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :adaptive => true, :allout => true, :parallel => false)
=#
tuning = Dict(:r_star => 2.,
              :xtol => 0., :resampling_method => :systematic,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true, :parallel => false)

# Generation of the initial state draws

n_states = n_states_augmented(m)
s0 = zeros(n_states)
P0 = solve_discrete_lyapunov(TTT, RRR*QQ*RRR')
U, E, V = svd(P0)
s_init = s0 .+ U*diagm(sqrt.(E))*randn(n_states, tuning[:n_particles])
tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning...)
