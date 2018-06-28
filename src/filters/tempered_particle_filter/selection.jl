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
"""
function resample(weights::Vector{Float64}; method::Symbol = :systematic)
    n_particles = length(weights)

    # Normalize weights if necessary
    if sum(weights) != 1
        weights = weights ./ sum(weights)
    end

    if method == :systematic

        cumulative_weights = cumsum(weights)
        offset = rand()

        new_inds = zeros(Int, n_particles)
        for i in 1:n_particles
            threshold = (i - 1 + offset) / n_particles
            start_ind = i == 1 ? 1 : new_inds[i-1]

            for j in start_ind:n_particles
                if cumulative_weights[j] > threshold
                    new_inds[i] = j
                    break
                end
            end
        end
        return new_inds

    elseif method == :multinomial
        return sample(1:n_particles, Weights(weights), n_particles, replace = true)

    else
        throw("Invalid resampling method: $method")
    end
end
