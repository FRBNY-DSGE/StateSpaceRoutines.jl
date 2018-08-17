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

kf_loglh_trial = @benchmark kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, s_0, P_0,
                                          outputs = [:loglh]) gcsample = true
kf_all_trial = @benchmark kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, s_0, P_0,
                                        outputs = [:loglh, :pred, :filt]) gcsample = true

trials = [kf_loglh_trial, kf_all_trial]
trial_names = [:kalman_filter_loglh, :kalman_filter_all]
group = construct_trial_group(trials, trial_names)
group_name = "kalman_filter"

# Optionally re-write the reference trial
# write_ref_trial_group(group, trial_names, group_name)
print_all_benchmarks(group, "../reference/$group_name.jld", group_name, trial_names)
