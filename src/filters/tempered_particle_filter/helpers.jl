"""
```
solve_inefficiency{S<:AbstractFloat}(φ_new::S, φ_old::S, y_t::Vector{S}, p_error::Matrix{S},
inv_HH::Matrix{S}, det_HH::S; initialize::Bool=false)
```
Returns the value of the ineffeciency function InEff(φₙ), where:

        InEff(φₙ) = (1/M) ∑ᴹ (W̃ₜʲ(φₙ))²

Where ∑ is over j=1...M particles, and for a particle j:

        W̃ₜʲ(φₙ) = w̃ₜʲ(φₙ) / (1/M) ∑ᴹ w̃ₜʲ(φₙ)

Where ∑ is over j=1...M particles, and incremental weight is:

        w̃ₜʲ(φₙ) = pₙ(yₜ|sₜʲ'ⁿ⁻¹) / pₙ₋₁(yₜ|sₜ^{j,n-1})
                = (φₙ/φₙ₋₁)^(d/2) exp{-1/2 [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' (φₙ-φₙ₋₁) ∑ᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}

### Inputs

- `φ_new`: φₙ
- `φ_old`: φₙ₋₁
- `y_t`: vector of observables for time t
- `p_error`: (`n_states` x `n_particles`) matrix of particles' errors yₜ - Ψ(sₜʲ'ⁿ⁻¹) in columns
- `inv_HH`: The inverse of the measurement error covariance matrix, ∑ᵤ
- `det_HH`: The determinant of the measurement error covariance matrix, ∑ᵤ

### Keyword Arguments

- `initialize::Bool`: flag to indicate whether this is being used in initialization stage,
    in which case one instead solves the formula for w̃ₜʲ(φₙ) as:

    w̃ₜʲ(φ₁) = (φ₁/2π)^(d/2)|∑ᵤ|^(1/2) exp{-1/2 [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' φ₁ ∑ᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}

"""
function solve_inefficiency{S<:AbstractFloat}(φ_new::S, coeff_terms::Vector{Float64}, exp_1_terms::Vector{Float64},
                                              exp_2_terms::Vector{Float64}, n_obs::Int64; parallel::Bool = false)

    n_particles = length(coeff_terms)

    if parallel
        w = @parallel (vcat) for i = 1:n_particles
            incremental_weight(φ_new, coeff_terms[i], exp_1_terms[i], exp_2_terms[i], n_obs)
        end
    else
        w = Vector{Float64}(n_particles)
        for i = 1:n_particles
            w[i] = incremental_weight(φ_new, coeff_terms[i], exp_1_terms[i], exp_2_terms[i], n_obs)
        end
    end

    W = w/mean(w)
    return sum(W.^2)/n_particles
end

function incremental_weight(φ_new::Float64, coeff_term::Float64, log_e_term_1::Float64,
                            log_e_term_2::Float64, n_obs::Int64)
    return φ_new^(n_obs/2)*coeff_term * exp(log_e_term_1) * exp(φ_new*log_e_term_2)
end

# The outputs of the weight_kernel function are meant to make calculating
# the incremental weight much more efficient to speed up the adaptive φ finding
# that way we're not doing the same matrix multiplication step (dot(p_error, inv_HH*p_error))
# for every iteration of the root-solving algorithm
# Also, the exponential terms are logged first and then exponentiated in the
# incremental_weight calculation so the problem is well-conditioned (i.e. not exponentiating
# very large negative numbers)
function weight_kernel(φ_old::Float64, y_t::Vector{Float64},
                       p_error::Vector{Float64}, det_HH::Float64, inv_HH::Matrix{Float64};
                       initialize::Bool = false)

    # Initialization step (using 2π instead of φ_old)
    if initialize
        coeff_term = (2*pi)^(-length(y_t)/2) * det_HH^(-1/2)
        log_e_term_1   = 0.
        log_e_term_2   = -1/2 * dot(p_error, inv_HH * p_error)
        return coeff_term, log_e_term_1, log_e_term_2
    # Non-initialization step (tempering and final iteration)
    else
        coeff_term = (φ_old)^(-length(y_t)/2)
        log_e_term_1   = -1/2 * (-φ_old) * dot(p_error, inv_HH * p_error)
        log_e_term_2   = -1/2 * dot(p_error, inv_HH * p_error)
        return coeff_term, log_e_term_1, log_e_term_2
    end
end

"""
```
resample(weights::AbstractArray; method::Symbol=:systematic,
         parallel::Bool=false, testing::Bool=false)
```

Reindexing and reweighting samples from a degenerate distribution

### Arguments:

- `weights`: The weights of each of the particles.
- `method`: :systematic or :multinomial
        the method for resampling.
- `parallel`: to indicate whether to resample using multiple workers (if available).
- `testing`: to indicate whether to give test output.

### Output:

- `vec(indx)`: id
        the newly assigned indices of parameter draws.
"""
function resample(weights::AbstractArray; method::Symbol = :multinomial,
                  parallel::Bool = false, testing::Bool = false)
    if method == :systematic
        n_parts = length(weights)
        # Stores cumulative weights until given index
        cumulative_weights = cumsum(weights./sum(weights))
        uu = Vector{Float64}(n_parts)

        # Random part of algorithm - choose offset of first index by some u~U[0,1)
        rand_offset = rand()

        # Set "spokes" at the position of the random offset
        for j = 1:n_parts
            uu[j] = (j - 1) + rand_offset
        end

        # Function solves where an individual "spoke" lands
        function subsys(i::Int)
            findfirst(j -> cumulative_weights[j] > uu[i]/n_parts, 1:length(cumulative_weights))
        end

        # Map function if parallel
        if parallel
            indx =
            @sync @parallel (vcat) for j in 1:n_parts
                subsys(j)
            end
        else
            indx = [subsys(j) for j = 1:n_parts]
        end

        return indx

    elseif method == :multinomial
        n_parts = length(weights)
        weights = Weights(weights./sum(weights))

        return sample(1:n_parts, weights, n_parts, replace = true)
    else
        throw("Invalid resampler. Set tuning field :resampling_method to either :systematic or :multinomial")
    end
end

