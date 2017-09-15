"""
```
correction_selection!(φ_new::Float64, φ_old::Float64, y_t::Vector{Float64}, p_error::Matrix{Float64},
s_lag_tempered::Matrix{Float64}, ε::Matrix{Float64}, HH::Matrix{Float64}, n_particles::Int;
    initialize::Bool=false)
```
Calculate densities, normalize and reset weights, call multinomial resampling, update state and
error vectors, reset error vectors to 1,and calculate new log likelihood.

### Inputs
- `φ_new::Float64`: current φ
- `φ_old::Float64`: φ from last tempering iteration
- `y_t::Vector{Float64}`: (`n_observables` x 1) vector of observables at time t
- `p_error::Vector{Float64}`: A single particle's error: y_t - Ψ(s_t)
- `HH::Matrix{Float64}`: measurement error covariance matrix, ∑ᵤ
- `n_particles::Int`: number of particles

### Keyword Arguments
- `initialize::Bool`: Flag indicating whether one is solving for incremental weights during
    the initialization of weights; default is `false`.

### Outputs
- `loglik`: incremental log likelihood
- `id`: vector of indices corresponding to resampled particles
"""
function correction_selection!(φ_new::Float64, φ_old::Float64, y_t::Vector{Float64},
                               p_error::Matrix{Float64}, HH::Matrix{Float64}, n_particles::Int;
                               initialize::Bool = false, parallel::Bool = false,
                               resampling_method::Symbol = :systematic)
    # Initialize vector
    incremental_weights = zeros(n_particles)

    # Calculate initial weights
    if parallel
        incremental_weights = @sync @parallel (vcat) for n = 1:n_particles
            incremental_weight(φ_new, φ_old, y_t, p_error[:,n], HH, initialize=initialize)
        end
    else
        for n = 1:n_particles
            incremental_weights[n] = incremental_weight(φ_new, φ_old, y_t, p_error[:,n], HH,
                                                        initialize = initialize)
        end
    end

    # Normalize weights
    normalized_weights = incremental_weights ./ mean(incremental_weights)

    # Resampling
    id = resample(normalized_weights, method = resampling_method, parallel = parallel)

    # Calculate likelihood
    loglik = log(mean(incremental_weights))

    return loglik, id
end

"""
```
incremental_weight(φ_new::Float64, φ_old::Float64, y_t::Vector{Float64}, p_error::Vector{Float64},
HH::Matrix{Float64}; initialize::Bool=false)
```
### Inputs
- `φ_new::Float64`: current φ
- `φ_old::Float64`: φ value before last
- `y_t::Vector{Float64}`: Vector of observables for time t
- `p_error::Vector{Float64}`: A single particle's error: y_t - Ψ(s_t)
- `HH::Matrix{Float64}`: Measurement error covariance matrix

### Keyword Arguments
- `initialize::Bool`: Flag indicating whether one is solving for incremental weights during
    the initialization of weights; default is `false`.

### Output
- Returns the incremental weight of single particle
"""
@inline function incremental_weight(φ_new::Float64, φ_old::Float64, y_t::Vector{Float64},
                                    p_error::Vector{Float64}, HH::Matrix{Float64}; initialize::Bool = false)

    # Initialization step (using 2π instead of φ_old)
    if initialize
        return (φ_new/(2*pi))^(length(y_t)/2) * (det(HH)^(-1/2)) *
            exp(-1/2 * p_error' * φ_new * inv(HH) * p_error)[1]

    # Non-initialization step (tempering and final iteration)
    else
        return (φ_new/φ_old)^(length(y_t)/2) *
            exp(-1/2 * p_error' * (φ_new - φ_old) * inv(HH) * p_error)[1]
    end
end
