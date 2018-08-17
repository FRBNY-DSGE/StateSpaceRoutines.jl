using StateSpaceRoutines
using QuantEcon: solve_discrete_lyapunov
using BenchmarkTools

include("../model_setup.jl")

# Loading the state-space components
const TTT = system[:TTT]
const RRR = system[:RRR]
const CCC = system[:CCC]
const QQ  = system[:QQ]
const ZZ  = system[:ZZ]
const DD  = system[:DD]
const EE  = system[:EE]

# Generate initial state/state-covariance matrix
n_states = n_states_augmented(m)
const s_0 = zeros(n_states)
const P_0 = solve_discrete_lyapunov(TTT, RRR*QQ*RRR')

trial = @benchmark hamilton_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE,
                                     s_0, P_0) gcsample = true

# Optionally re-write the reference trial
# write_ref_trial(trial, "hamilton_smoother")
print_all_benchmarks(trial, "../reference/hamilton_smoother.jld", "hamilton_smoother")
