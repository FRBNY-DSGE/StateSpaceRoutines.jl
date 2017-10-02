m = AnSchorfheide()
df = readtable("/data/dsge_data_dir/dsgejl/mcai/TPF/save/input_data/data/us.txt", header=false, separator=' ')
data = convert(Matrix{Float64}, df)'

params = [2.09, 0.98, 2.25, 0.65, 0.34, 3.16, 0.51, 0.81, 0.98, 0.93, 0.19, 0.65, 0.24,
          0.115985, 0.294166, 0.447587]

update!(m, params)

system  = compute_system(m)

s0 = zeros(n_states_augmented(m))
RRR     = system[:RRR]
TTT     = system[:TTT]
HH      = system[:EE]
DD      = system[:DD]
ZZ      = system[:ZZ]
QQ      = system[:QQ]
sqrtS2 = RRR*chol(QQ)'

Φ(s_t::Vector{Float64}, ϵ_t::Vector{Float64}) = TTT*s_t + sqrtS2*ϵ_t
Ψ(s_t::Vector{Float64}, u_t::Vector{Float64}) = ZZ*s_t + DD + u_t

F_ϵ = Distributions.MvNormal(zeros(size(QQ, 1)), eye(size(QQ, 1)))
F_u = Distributions.MvNormal(zeros(size(HH, 1)), HH)

s_init = h5read("reference/tempered_particle_filter_args.h5", "s_init")
fixed_sched = [0.2, 0.5, 1.0]

test_data = reshape(data[:, 1], 3, 1)
loglik, lik, _ = tempered_particle_filter(test_data, Φ, Ψ, F_ϵ, F_u, s_init; r_star = 2., c = 0.3,
                         accept_rate = 0.4, target = 0.4, xtol = 0., resampling_method = :multinomial,
                         N_MH = 1, n_particles = 500, n_presample_periods = 0, verbose = :none,
                         adaptive = false, fixed_sched = fixed_sched, allout = true, parallel = false,
                         testing = true)

file = h5open("reference/tempered_particle_filter_out.h5", "r")
test_loglik = read(file, "log_lik")
test_lik    = read(file, "incr_lik")
close(file)

@test test_loglik ≈ (loglik)
@test test_lik ≈ lik

tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; n_particles = 500, verbose = :none);

nothing
