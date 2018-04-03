# """
# ```
# correction_selection!(φ_new::Float64, φ_old::Float64, y_t::Vector{Float64}, p_error::Matrix{Float64},
# s_lag_tempered::Matrix{Float64}, ε::Matrix{Float64}, HH::Matrix{Float64}, n_particles::Int;
    # initialize::Bool=false)
# ```
# Calculate densities, normalize and reset weights, call multinomial resampling, update state and
# error vectors, reset error vectors to 1,and calculate new log likelihood.

# ### Inputs
# - `φ_new::Float64`: current φ
# - `φ_old::Float64`: φ from last tempering iteration
# - `y_t::Vector{Float64}`: (`n_observables` x 1) vector of observables at time t
# - `p_error::Vector{Float64}`: A single particle's error: y_t - Ψ(s_t)
# - `HH::Matrix{Float64}`: measurement error covariance matrix, ∑ᵤ
# - `n_particles::Int`: number of particles

# ### Keyword Arguments
# - `initialize::Bool`: Flag indicating whether one is solving for incremental weights during
    # the initialization of weights; default is `false`.

# ### Outputs
# - `loglik`: incremental log likelihood
# - `id`: vector of indices corresponding to resampled particles
# """

function correction(φ_new::Float64, coeff_terms::Vector{Float64}, log_e_1_terms::Vector{Float64},
                    log_e_2_terms::Vector{Float64}, n_obs::Int64)
    n_particles = length(coeff_terms)
    incremental_weights = Vector{Float64}(n_particles)
    for i = 1:n_particles
        incremental_weights[i] = incremental_weight(φ_new, coeff_terms[i], log_e_1_terms[i], log_e_2_terms[i], n_obs)
    end

    normalized_weights = incremental_weights ./ mean(incremental_weights)

    # Calculate likelihood
    loglik = log(mean(incremental_weights))

    return normalized_weights, loglik
end

function selection(normalized_weights::Vector{Float64}, s_lag_tempered::Matrix{Float64},
                   s_t_nontempered::Matrix{Float64}, ϵ::Matrix{Float64};
                   resampling_method::Symbol = :multinomial)
    # Resampling
    id = resample(normalized_weights, method = resampling_method)

    # Update arrays for resampled indices
    s_lag_tempered  = s_lag_tempered[:,id]
    s_t_nontempered = s_t_nontempered[:,id]
    ϵ               = ϵ[:,id]

    return s_lag_tempered, s_t_nontempered, ϵ
end
