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

function next_φ!(Ψ::Function, stage::Int, φ_old::Float64, det_HH::Float64, inv_HH::Matrix{Float64},
                 y_t::Vector{Float64}, s_t_nontemp::AbstractMatrix{Float64},
                 coeff_terms::AbstractVector{Float64}, log_e_1_terms::AbstractVector{Float64},
                 log_e_2_terms::AbstractVector{Float64}, r_star::Float64;
                 adaptive::Bool = true, findroot::Function = bisection, xtol::Float64 = 1e-3,
                 fixed_sched::Vector{Float64} = zeros(0), parallel::Bool = false)
    # Sizes
    n_particles = length(coeff_terms)
    n_obs = length(y_t)

    # Compute weight kernel terms
    @mypar parallel for i in 1:n_particles
        p_err = y_t - Ψ(s_t_nontemp[:, i])
        coeff_terms[i], log_e_1_terms[i], log_e_2_terms[i] =
            weight_kernel(φ_old, y_t, p_err, det_HH, inv_HH, initialize = stage == 1)
    end

    # Determine φ_new
    φ_new = if adaptive
        # Compute interval
        solve_ineff_func(φ) =
            solve_inefficiency(φ, coeff_terms, log_e_1_terms, log_e_2_terms, n_obs) - r_star

        if stage == 1
            findroot(solve_ineff_func, φ_old, 1.0, xtol = xtol)
        else
            fphi_interval = [solve_ineff_func(φ_old) solve_ineff_func(1.0)]

            # Look for optimal φ within the interval
            if prod(sign.(fphi_interval)) == -1
                findroot(solve_ineff_func, φ_old, 1.0, xtol = xtol)
            else
                1.0
            end
        end
    else
        fixed_sched[stage]
    end
end

function correction!(φ_new::Float64, coeff_terms::AbstractVector{Float64},
                     log_e_1_terms::AbstractVector{Float64}, log_e_2_terms::AbstractVector{Float64},
                     n_obs::Int, inc_weights::Vector{Float64}, norm_weights::Vector{Float64})
    # Compute incremental weights
    n_particles = length(inc_weights)
    for i = 1:n_particles
        inc_weights[i] =
            incremental_weight(φ_new, coeff_terms[i], log_e_1_terms[i], log_e_2_terms[i], n_obs)
    end

    # Normalize weights
    norm_weights .= inc_weights ./ mean(inc_weights)

    return nothing
end

function selection!(norm_weights::Vector{Float64}, s_t1_temp::AbstractMatrix{Float64},
                    s_t_nontemp::AbstractMatrix{Float64}, ϵ::AbstractMatrix{Float64};
                    resampling_method::Symbol = :multinomial)
    # Resampling
    id = resample(norm_weights, method = resampling_method)

    # Update arrays for resampled indices
    s_t_nontemp .= s_t_nontemp[:, id]
    s_t1_temp   .= s_t1_temp[:, id]
    ϵ           .= ϵ[:, id]

    return nothing
end
