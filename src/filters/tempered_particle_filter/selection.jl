"""
```
selection!(norm_weights, s_t1_temp, s_t_nontemp, ϵ_t;
    resampling_method = :multinomial)
```

Resample particles using `norm_weights`. This function modifies `s_t1_temp`,
`s_t_nontemp`, and `ϵ_t` in place.
"""
function selection!(norm_weights::Vector{Float64}, s_t1_temp::M, s_t_nontemp::M, ϵ_t::M;
                    resampling_method::Symbol = :multinomial) where M<:AbstractMatrix{Float64}
    # Resampling
    is = resample(norm_weights, method = resampling_method)

    # Update arrays using resampled indices
    s_t1_temp   .= s_t1_temp[:, is]
    s_t_nontemp .= s_t_nontemp[:, is]
    ϵ_t         .= ϵ_t[:, is]

    return nothing
end

"""
```
resample(weights; method = :systematic)
```

Return indices after resampling according to `method`, weighting by `weights`.
See https://xianblog.files.wordpress.com/2017/12/lawrenz.png for a helpful
visual comparison of multinomial and systematic resampling.
"""
function resample(weights::Vector{Float64}; method::Symbol = :systematic)
    n_particles = length(weights)

    # Normalize weights if necessary
    if sum(weights) != 1
        weights = weights ./ sum(weights)
    end

    if method == :systematic
        # Divide a circle of circumference 1 into n_particles segments, where
        # the jth segment has length weights[j] and lies on the half-closed
        # interval (cdf[j-1], cdf[j]]
        cdf = cumsum(weights)

        # Take a wheel with n_particles spokes, each spaced 1/n_particles
        # apart, and shift it from its initial position (where there is a spoke
        # at 0) by offset
        offset = rand() / n_particles

        # The ith spoke landed in the jth segment <=> set the ith new particle
        # equal to the jth old particle
        new_inds = zeros(Int, n_particles)
        j = 1
        for i in 1:n_particles
            spoke = (i-1)/n_particles + offset
            # Find j such that cdf[j-1] < spoke <= cdf[j]
            while spoke > cdf[j]
                j += 1
            end
            new_inds[i] = j
        end
        return new_inds

    elseif method == :multinomial
        return sample(1:n_particles, Weights(weights), n_particles, replace = true)

    else
        throw("Invalid resampling method: $method")
    end
end
