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
