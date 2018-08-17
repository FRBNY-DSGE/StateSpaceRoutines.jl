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

ck_drawstates_trial = @benchmark carter_kohn_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE,
                                                      s_0, P_0, draw_states = true) gcsample = true
ck_nodrawstates_trial = @benchmark carter_kohn_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, s_0,
                                                        P_0, draw_states = false) gcsample = true

trials = [ck_drawstates_trial, ck_nodrawstates_trial]
trial_names = [:carter_kohn_smoother_drawstates, :carter_kohn_smoother_nodrawstates]
group = construct_trial_group(trials, trial_names)
group_name = "carter_kohn_smoother"

# Optionally re-write the reference trial
# write_ref_trial_group(group, trial_names, group_name)
print_all_benchmarks(group, "../reference/$group_name.jld", group_name, trial_names)
