using JLD2, StateSpaceRoutines, Test, HDF5, Distributions, ModelConstructors

path = dirname(@__FILE__)

# Initialize arguments to for multi-regime Kalman Filter (ZLB)
@load "$path/reference/kalman_filter_args_zlb.jld2" y Ts Rs Cs Qs Zs Ds Es regime_inds

for i in 1:length(Es)
    for j in 1:size(Es[i],1)
        Es[i][j,j] = 0.5
    end
end
out = kalman_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es)

# Remove zeros that create degeneracy
for i in 1:length(Qs)
    keep_qs = [sum(Qs[i][:,j] .!= 0.0) > 0 for j in 1:size(Qs[i],2)]
    Qs[i] = Qs[i][keep_qs, keep_qs]
    Rs[i] = Rs[i][:,keep_qs]
end

# Check Time-varying TPF now
Φ(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = Ts[1] * s_t + Cs[1] + Rs[1] * ϵ_t
Ψ(s_t::AbstractVector{Float64}) = Zs[1] * s_t + Ds[1]
Φ2(s_t::AbstractVector{Float64}, ϵ_t::AbstractVector{Float64}) = Ts[2] * s_t + Cs[2] + Rs[2] * ϵ_t
Ψ2(s_t::AbstractVector{Float64}) = Zs[2] * s_t + Ds[2]

Φ_vec = [Φ,Φ2]
Ψ_vec = [Ψ,Ψ2]

# Distributions
F_u1 = MvNormal(zeros(size(Es[1],1)),Es[1])
F_u2 = MvNormal(zeros(size(Es[2],1)),Es[2])
F_u_vec = AbstractVector{Distribution}([F_u1, F_u2])

F_ϵ1 = MvNormal(zeros(size(Qs[1],1)),Qs[1])
F_ϵ2 = MvNormal(zeros(size(Qs[2],1)),Qs[2])
F_ϵ_vec = AbstractVector{Distribution}([F_ϵ1, F_ϵ2])

# Regime indices
reg_ts = ones(Int64, length(regime_inds[1]))
for i in 2:length(regime_inds)
    append!(reg_ts, repeat([i],length(regime_inds[i])))
end

# Run TPF
n_particles = 10000
s_init = rand(DegenerateMvNormal(out[6], out[7]), n_particles)
tpf_out = tempered_particle_filter(y, Φ_vec, Ψ_vec, F_ϵ_vec, F_u_vec,
                         s_init; n_particles = n_particles,
                         Φ_regime_inds = reg_ts, Ψ_regime_inds = reg_ts, verbose = :none,
                         F_ϵ_regime_inds = reg_ts, F_u_regime_inds = reg_ts)

@assert abs(tpf_out[1] - sum(out[1])) < 20.0

nothing
