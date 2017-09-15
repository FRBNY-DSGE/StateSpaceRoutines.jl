"""
```
solve_inefficiency{S<:AbstractFloat}(φ_new::S, φ_old::S, y_t::Vector{S}, p_error::Matrix{S},
HH::Matrix{S}; initialize::Bool=false)
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
- `HH`: measurement error covariance matrix, ∑ᵤ

### Keyword Arguments

- `initialize::Bool`: flag to indicate whether this is being used in initialization stage,
    in which case one instead solves the formula for w̃ₜʲ(φₙ) as:

    w̃ₜʲ(φ₁) = (φ₁/2π)^(d/2)|∑ᵤ|^(1/2) exp{-1/2 [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' φ₁ ∑ᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}

"""
function solve_inefficiency{S<:AbstractFloat}(φ_new::S, φ_old::S, y_t::Vector{S},
                                              p_error::Matrix{S}, HH::Matrix{S}; initialize::Bool=false)

    n_particles = size(p_error, 2)
    n_obs       = length(y_t)
    w           = zeros(n_particles)
    inv_HH      = inv(HH)
    det_HH      = det(HH)

    # Inefficiency function during initialization
    if initialize
        for i=1:n_particles
            w[i] = ((φ_new/(2*pi))^(n_obs/2) * (det_HH^(-1/2)) * exp(-1/2 * p_error[:,i]' *
                                                φ_new * inv_HH * p_error[:,i]))[1]
        end

    # Inefficiency function during tempering steps
    else
        for i=1:n_particles
            w[i] = (φ_new/φ_old)^(n_obs/2) * exp(-1/2 * p_error[:,i]' *
                                                 (φ_new-φ_old) * inv_HH * p_error[:,i])[1]
        end
    end
    W = w/mean(w)
    return sum(W.^2)/n_particles
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
function resample(weights::AbstractArray; method::Symbol = :systematic,
                  parallel::Bool = false, testing::Bool = false)
    if method == :systematic
        n_parts = length(weights)
        weights = weights./sum(weights)
        # Stores cumulative weights until given index
        cumulative_weights = cumsum(weights)
        weights = weights'
        uu = zeros(n_parts, 1)

        # Random part of algorithm - choose offset of first index by some u~U[0,1)
        rand_offset = rand()

        # Set "spokes" at the position of the random offset
        for j = 1:n_parts
            uu[j] = (j - 1) + rand_offset
        end

        # Initialize output vector
        indx = zeros(n_parts, 1)

        # Function solves where an individual "spoke" lands
        function subsys(i)
            u = uu[i]/n_parts
            j = 1
            while j <= n_parts
                if (u < cumulative_weights[j])
                    break
                end
                j += 1
            end
            indx[i] = j
        end

        # Map function if parallel
        if parallel
            parindx =
            @sync @parallel (vcat) for j in 1:n_parts
                subsys(j)
            end
        else
            parindx = [subsys(j) for j = 1:n_parts]'
        end

        # Transpose and round output indices
        indx = parindx'
        indx = round(Int, indx)

        return vec(indx)
    elseif method == :multinomial
        n_parts = length(weights)
        weights = Weights(weights./sum(weights))

        return sample(1:n_parts, weights, n_parts, replace = true)
    else
        throw("Invalid resampler. Set tuning field :resampling_method to either :systematic or :multinomial")
    end
end
