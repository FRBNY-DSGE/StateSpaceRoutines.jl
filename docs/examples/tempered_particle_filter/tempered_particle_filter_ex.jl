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
TTT = system[:TTT]
RRR = system[:RRR]
QQ  = system[:QQ]
Φ, Ψ, F_ϵ, F_u = compute_system_function(system)

# Tuning of the tempered particle filter algorithm

tuning = Dict(:r_star => 2.,
              :xtol => 1e-3, :resampling_method => :systematic,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true, :parallel => false)

# Generation of the initial state draws

n_states = n_states_augmented(m)
s0 = zeros(n_states)
P0 = solve_discrete_lyapunov(TTT, RRR*QQ*RRR')
U, E, V = svd(P0)
s_init = s0 .+ U*diagm(sqrt.(E))*randn(n_states, tuning[:n_particles])
tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning...)
