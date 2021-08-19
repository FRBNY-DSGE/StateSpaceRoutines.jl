"""
```
mutation!(Φ, Ψ, QQ, det_HH, inv_HH, φ_new, y_t, s_t, s_t1, ϵ_t, c, n_mh_steps;
    parallel = false, poolmodel = false)
```

Mutate particles by taking Metropolis-Hastings steps in the `ϵ_t` space. This
function modifies `s_t` and `ϵ_t` in place and returns `accept_rate`.
"""
function mutation!(Φ::Function, Ψ::Function, QQ::Matrix{Float64},
                   det_HH::Float64, inv_HH::Matrix{Float64}, φ_new::Float64,
                   y_t::Vector{Float64}, s_t, s_t1,
                   ϵ_t, c::Float64, n_mh_steps::Int;
                   parallel::Bool = false,
                   poolmodel::Bool = false) where M<:AbstractMatrix{Float64}
    # Sizes
    n_obs = size(y_t, 1)
    n_particles = size(ϵ_t, 2)

    # Initialize vector of acceptances
    #accept_vec = parallel ? SharedVector{Int}(n_particles) : Vector{Int}(undef, n_particles)
    accept_vec = Vector{Int}(undef, n_particles)

    # Used to generate new draws of ϵ
    dist_ϵ = MvNormal(c^2 * diag(QQ))

    # Used to calculate posteriors
    scaled_det_HH = det_HH/(φ_new^n_obs)
    scaled_inv_HH = inv_HH*φ_new

    # Take Metropolis-Hastings steps
    if parallel
        combined_mat_inds = (size(s_t,1) + size(ϵ_t,1) + 1, n_particles)
        combined_mat = DArray(combined_mat_inds, workers(), [1, nworkers()]) do inds
            arr = zeros(inds)
            s_t1_d = OffsetArray(DistributedArrays.localpart(s_t1), DistributedArrays.localindices(s_t1))
            s_t_d = OffsetArray(DistributedArrays.localpart(s_t), DistributedArrays.localindices(s_t))
            # ϵ_t_d = OffsetArray(localpart(ϵ_t), DistributedArrays.localindices(ϵ_t))

            for i in inds[2]
                ss, ϵs, accepts =
                    mh_steps(Φ, Ψ, dist_ϵ, y_t, parent(s_t1_d[:,i]), parent(s_t_d[:,i]), ϵ_t[:,i],
                             scaled_det_HH, scaled_inv_HH, n_mh_steps;
                             poolmodel = poolmodel)
                arr[1:size(s_t,1),i] .= ss
                arr[size(s_t,1)+1:end-1,i] .= ϵs
                arr[end,i] = accepts
            end
            parent(arr)
        end
        s_t = combined_mat[1:size(s_t,1),:]
        ϵ_t .= combined_mat[size(s_t,1)+1:end-1,:]
        accept_vec .= combined_mat[end,:]
    else
         for i in 1:n_particles
            s_t[:,i], ϵ_t[:,i], accept_vec[i] =
                mh_steps(Φ, Ψ, dist_ϵ, y_t, s_t1[:,i], s_t[:,i], ϵ_t[:,i],
                         scaled_det_HH, scaled_inv_HH, n_mh_steps;
                         poolmodel = poolmodel)
        end
    end

    # Calculate and return acceptance rate
    accept_rate = sum(accept_vec) / (n_mh_steps*n_particles)
    return accept_rate
end

"""
```
mh_steps(Φ, Ψ, dist_ϵ, y_t, s_t1, s_t, ϵ_t, scaled_det_HH,
    scaled_inv_HH, n_mh_steps; poolmodel = false)
```

Take `n_mh_steps` many steps in the `ϵ_t` space for a single particle. Returns
the new `s_t`, `ϵ_t`, and the number of acceptances `accept`.
"""
function mh_steps(Φ::Function, Ψ::Function, dist_ϵ::MvNormal, y_t::Vector{Float64},
                  s_t1::Vector{Float64}, s_t::Vector{Float64}, ϵ_t::Vector{Float64},
                  scaled_det_HH::Float64, scaled_inv_HH::Matrix{Float64}, n_mh_steps::Int;
                  poolmodel::Bool = false)
    accept = 0

    # Compute posterior at initial ϵ_t
    if poolmodel
        post_1 = Ψ(s_t) # PoolModel measurement eq is the likelihood
    else
        post_1 = fast_mvnormal_pdf(y_t - Ψ(s_t), scaled_det_HH, scaled_inv_HH)
    end
    post_2 = fast_mvnormal_pdf(ϵ_t)
    post   = post_1 * post_2

    for j = 1:n_mh_steps
        # Draw ϵ_new and s_new
        ϵ_new = ϵ_t + rand(dist_ϵ)
        s_new = Φ(s_t1, ϵ_new)

        # Calculate posterior
        post_new_1 = poolmodel ? Ψ(s_new) : fast_mvnormal_pdf(y_t - Ψ(s_new),
                                                              scaled_det_HH, scaled_inv_HH)
        post_new_2 = fast_mvnormal_pdf(ϵ_new)
        post_new   = post_new_1 * post_new_2

        # Calculate α, probability of accepting the new particle
        α = post_new / post
        if rand() < α
            s_t  = s_new
            ϵ_t  = ϵ_new
            post = post_new
            accept += 1
        end
    end
    return s_t, ϵ_t, accept
end

"""
```
update_c(c, accept_rate, target_rate)
```

Return the new proposal covariance matrix scaling `c`, adaptively chosen given
`accept_rate` to match `target_rate`.
"""
@inline function update_c(c::Float64, accept_rate::Float64, target_rate::Float64)
    c*(0.95 + 0.1*exp(20*(accept_rate - target_rate))/(1 + exp(20*(accept_rate - target_rate))))
end
